"""3-stage multi-agent curriculum & historical league self-play PPO trainer."""

from __future__ import annotations

import argparse
import copy
import random
from dataclasses import dataclass

import torch

from packages.python_rl.export_onnx import export_actor_to_serialized_json
from packages.python_rl.ppo import ActorCriticMLP, PPOConfig, compute_gae, ppo_update
from packages.python_sim.contracts import ActionCode, WorldConfig
from packages.python_sim.env_vec import VectorizedLineupEnv


def rule_bot_vectorized_actions(obs: torch.Tensor) -> torch.Tensor:
    """Fast vectorized heuristic opponent policy on `[..., 16]` observations."""
    rel_peak = obs[..., 0]
    time_to_wave = obs[..., 1]
    rel_cross = obs[..., 2]
    stamina = obs[..., 3]
    pressure = obs[..., 4]
    status_norm = obs[..., 9]
    inside_threat = obs[..., 13]

    actions = torch.full(obs.shape[:-1], ActionCode.WAIT, dtype=torch.int64, device=obs.device)

    in_catch_win = (
        (time_to_wave >= -0.05)
        & (time_to_wave <= 0.075)
        & (torch.abs(rel_peak) <= 0.22)
        & (rel_cross >= -0.18)
        & (rel_cross <= 0.25)
        & (stamina >= 0.1)
    )
    yield_inside = (inside_threat > 0.55) | (pressure > 0.86)

    actions = torch.where(rel_peak < -0.05, ActionCode.LEFT, actions)
    actions = torch.where(rel_peak > 0.05, ActionCode.RIGHT, actions)
    actions = torch.where(rel_cross > 0.14, ActionCode.OUT, actions)
    actions = torch.where((rel_cross < -0.14) & (time_to_wave > 0.12), ActionCode.IN, actions)
    actions = torch.where(
        (time_to_wave > 0.075)
        & (time_to_wave <= 0.16)
        & (torch.abs(rel_peak) <= 0.09)
        & (stamina >= 0.18),
        ActionCode.IN,
        actions,
    )
    actions = torch.where(in_catch_win & (~yield_inside), ActionCode.CATCH, actions)
    actions = torch.where(in_catch_win & yield_inside, ActionCode.OUT, actions)
    # Riding or wiped out -> WAIT
    actions = torch.where(status_norm > 0.35, ActionCode.WAIT, actions)
    return actions


@dataclass
class TrainRunResult:
    updates: int
    total_transitions: int
    mean_episode_reward: float
    mean_catch_rate: float
    final_losses: dict[str, float]


def train_crowded_self_play(
    num_envs: int = 16,
    num_surfers: int = 16,
    rollout_steps: int = 32,
    updates: int = 6,
    device: str = "cpu",
    seed: int = 2026,
) -> tuple[ActorCriticMLP, TrainRunResult]:
    """Train an Actor-Critic policy with curriculum + historical league self-play."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    dev = torch.device(device)

    model = ActorCriticMLP().to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    ppo_cfg = PPOConfig(minibatch_size=min(256, max(32, (num_envs * rollout_steps) // 2)))

    league_snapshots: list[ActorCriticMLP] = []
    last_losses: dict[str, float] = {}
    mean_reward = 0.0
    mean_catch = 0.0
    total_transitions = 0

    for upd in range(updates):
        # Stage 1 (first 25%): solo simplified; Stage 2 (25-60%): crowded bots; Stage 3: league self-play
        frac = upd / max(1, updates)
        if frac < 0.25:
            stage_surfers = 1
            simplified = True
            width = 160.0
        elif frac < 0.6:
            stage_surfers = num_surfers
            simplified = False
            width = 320.0 if num_surfers >= 8 else 160.0
        else:
            stage_surfers = num_surfers
            simplified = False
            width = 320.0 if num_surfers >= 8 else 160.0

        env = VectorizedLineupEnv(
            config=WorldConfig(
                width=width,
                numSurfers=stage_surfers,
                maxWaves=10 if width >= 240 else 6,
                maxWhitewaters=40 if width >= 240 else 24,
                simplifiedWaves=simplified,
                maxSteps=rollout_steps + 5,
            ),
            num_envs=num_envs,
            device=dev,
            float_dtype=torch.float32,
        )
        obs_all = env.reset([seed + upd * 100 + i for i in range(num_envs)])

        # Select opponent policy for stage 3 (70% current copy, 20% historical snapshot, 10% rule bots)
        opponent_model: ActorCriticMLP | None = None
        if stage_surfers > 1 and frac >= 0.6 and league_snapshots:
            roll = rng.random()
            if roll < 0.7:
                opponent_model = copy.deepcopy(model).eval()
            elif roll < 0.9:
                opponent_model = rng.choice(league_snapshots)

        obs_buf = torch.zeros((rollout_steps, num_envs, model.obs_dim), device=dev)
        act_buf = torch.zeros((rollout_steps, num_envs), dtype=torch.int64, device=dev)
        logp_buf = torch.zeros((rollout_steps, num_envs), device=dev)
        rew_buf = torch.zeros((rollout_steps, num_envs), device=dev)
        done_buf = torch.zeros((rollout_steps, num_envs), dtype=torch.bool, device=dev)
        val_buf = torch.zeros((rollout_steps, num_envs), device=dev)

        for t in range(rollout_steps):
            obs_learner = obs_all[:, 0, :]
            with torch.no_grad():
                act0, logp0, val0 = model.act(obs_learner)

            full_actions = torch.zeros((num_envs, stage_surfers), dtype=torch.int64, device=dev)
            full_actions[:, 0] = act0

            if stage_surfers > 1:
                opp_obs = obs_all[:, 1:, :]
                if opponent_model is not None:
                    with torch.no_grad():
                        flat_opp = opp_obs.reshape(-1, model.obs_dim)
                        opp_act, _, _ = opponent_model.act(flat_opp)
                        full_actions[:, 1:] = opp_act.reshape(num_envs, stage_surfers - 1)
                else:
                    full_actions[:, 1:] = rule_bot_vectorized_actions(opp_obs)

            next_obs_all, rewards_all, dones = env.step(full_actions)
            obs_buf[t] = obs_learner
            act_buf[t] = act0
            logp_buf[t] = logp0
            rew_buf[t] = rewards_all[:, 0]
            done_buf[t] = dones
            val_buf[t] = val0
            obs_all = next_obs_all
            total_transitions += num_envs * stage_surfers

        with torch.no_grad():
            _, last_val = model(obs_all[:, 0, :])

        advantages, returns = compute_gae(
            rew_buf, val_buf, done_buf, last_val, ppo_cfg.gamma, ppo_cfg.gae_lambda
        )
        last_losses = ppo_update(
            model,
            optimizer,
            obs_buf.reshape(-1, model.obs_dim),
            act_buf.reshape(-1),
            logp_buf.reshape(-1),
            returns.reshape(-1),
            advantages.reshape(-1),
            ppo_cfg,
        )

        mean_reward = float(rew_buf.sum(dim=0).mean().item())
        waves = torch.clamp(env.total_waves_spawned.to(torch.float32), min=1.0)
        mean_catch = float((env.surfer_catches[:, 0].to(torch.float32) / waves).mean().item())

        if (upd + 1) % max(1, updates // 4) == 0:
            snap = copy.deepcopy(model).eval()
            league_snapshots.append(snap)
            if len(league_snapshots) > 8:
                league_snapshots.pop(0)

    return model, TrainRunResult(
        updates=updates,
        total_transitions=total_transitions,
        mean_episode_reward=round(mean_reward, 3),
        mean_catch_rate=round(mean_catch, 4),
        final_losses=last_losses,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multi-agent Crowded Lineup PPO")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--num-surfers", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    model, res = train_crowded_self_play(
        num_envs=args.num_envs,
        num_surfers=args.num_surfers,
        rollout_steps=args.rollout_steps,
        updates=args.updates,
        device=args.device,
    )
    print(f"Training complete: {res}")
    if args.out:
        export_actor_to_serialized_json(
            model,
            args.out,
            training_episodes=args.updates * args.num_envs,
            num_competitors=args.num_surfers - 1,
            avg_catch_rate=res.mean_catch_rate,
            avg_reward=res.mean_episode_reward,
        )
        print(f"Exported ONNX SerializedPolicy to {args.out}")


if __name__ == "__main__":
    main()
