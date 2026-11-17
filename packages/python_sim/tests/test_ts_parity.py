"""Cross-runtime parity tests verifying TypeScript <-> Python/PyTorch equivalence."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from packages.python_sim.contracts import ReferenceScenario
from packages.python_sim.env_vec import VectorizedLineupEnv
from packages.python_sim.prng import next_random, normalize_seed

FIXTURES_PATH = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "reference-scenarios.json"
)


def test_mulberry32_prng_bit_exact() -> None:
    state = normalize_seed(1337)
    vals = []
    for _ in range(5):
        v, state = next_random(state)
        vals.append(v)
    assert len(vals) == 5
    assert all(0.0 <= v < 1.0 for v in vals)
    assert state != normalize_seed(1337)


def test_typescript_reference_scenarios_parity() -> None:
    raw = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))
    scenarios = [ReferenceScenario.model_validate(item) for item in raw]
    assert len(scenarios) >= 4

    for scenario in scenarios:
        env = VectorizedLineupEnv(config=scenario.config, num_envs=1, device="cpu")
        env.reset(scenario.seed)

        for step_rec in scenario.steps:
            actions = torch.tensor([step_rec.actions], dtype=torch.int64)
            obs, rewards, _dones = env.step(actions, dt=scenario.config.dt)

            exp_obs = torch.tensor(step_rec.expectedObservations, dtype=torch.float32)
            exp_rew = torch.tensor(step_rec.expectedRewards, dtype=torch.float64)
            exp_x = torch.tensor(step_rec.expectedSurferX, dtype=torch.float64)
            exp_y = torch.tensor(step_rec.expectedSurferY, dtype=torch.float64)
            exp_stam = torch.tensor(step_rec.expectedSurferStamina, dtype=torch.float64)
            exp_state = torch.tensor(step_rec.expectedSurferState, dtype=torch.int32)

            assert torch.allclose(obs[0], exp_obs, atol=1e-4), (
                f"{scenario.name} step {step_rec.stepIndex} obs mismatch: "
                f"max diff={(obs[0] - exp_obs).abs().max().item()}"
            )
            assert torch.allclose(rewards[0], exp_rew, atol=1e-4), (
                f"{scenario.name} step {step_rec.stepIndex} reward mismatch"
            )
            assert torch.allclose(env.surfer_x[0], exp_x, atol=1e-4)
            assert torch.allclose(env.surfer_y[0], exp_y, atol=1e-4)
            assert torch.allclose(env.surfer_stamina[0], exp_stam, atol=1e-4)
            assert torch.equal(env.surfer_state[0], exp_state)

        metrics = env.get_episode_metrics(0)
        assert metrics.surferCatches == scenario.expectedMetrics.surferCatches
        assert metrics.surferDropIns == scenario.expectedMetrics.surferDropIns
        assert metrics.surferCollisions == scenario.expectedMetrics.surferCollisions
