"""Actor-Critic MLP and PPO update implementation matching `docs/gpu-rl-roadmap.md`."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Categorical

from packages.python_sim.contracts import NUM_DISCRETE_ACTIONS, OBSERVATION_DIM


class ActorCriticMLP(nn.Module):
    """Shared-trunk Actor-Critic network (`16 -> 128 -> 128 -> {6, 1}`)."""

    def __init__(
        self,
        obs_dim: int = OBSERVATION_DIM,
        hidden_dim: int = 128,
        num_actions: int = NUM_DISCRETE_ACTIONS,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)
        actions = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs, values


@dataclass
class PPOConfig:
    gamma: float = 0.98
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 256


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float = 0.98,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation over `[T, B]` rollouts."""
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(last_value)

    for t in reversed(range(T)):
        next_val = last_value if t == T - 1 else values[t + 1]
        non_terminal = 1.0 - dones[t].to(torch.float32)
        delta = rewards[t] + gamma * next_val * non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def ppo_update(
    model: ActorCriticMLP,
    optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    cfg: PPOConfig,
) -> dict[str, float]:
    """Run clipped PPO epochs over flattened `[M, obs_dim]` batch."""
    n_samples = obs.shape[0]
    adv_norm = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    steps = 0

    for _ in range(cfg.ppo_epochs):
        perm = torch.randperm(n_samples, device=obs.device)
        for start in range(0, n_samples, cfg.minibatch_size):
            idx = perm[start : start + cfg.minibatch_size]
            logits, values = model(obs[idx])
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions[idx])
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs[idx])
            surr1 = ratio * adv_norm[idx]
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * adv_norm[idx]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * ((values - returns[idx]) ** 2).mean()

            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            total_entropy += float(entropy.item())
            steps += 1

    denom = max(1, steps)
    return {
        "policy_loss": round(total_policy_loss / denom, 5),
        "value_loss": round(total_value_loss / denom, 5),
        "entropy": round(total_entropy / denom, 5),
    }
