"""Smoke & parity tests for PPO training and ONNX export."""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from packages.python_rl.export_onnx import (
    ActorLogitsModule,
    export_actor_to_serialized_json,
)
from packages.python_rl.self_play import train_crowded_self_play
from packages.python_sim.contracts import CONTRACT_VERSION, NUM_DISCRETE_ACTIONS, OBSERVATION_DIM


def test_ppo_crowded_self_play_and_onnx_parity(tmp_path: Path) -> None:
    model, res = train_crowded_self_play(
        num_envs=4,
        num_surfers=16,
        rollout_steps=12,
        updates=4,
        device="cpu",
        seed=123,
    )
    assert res.updates == 4
    assert res.total_transitions > 0

    out_json = tmp_path / "ppo_policy_v2.json"
    serialized = export_actor_to_serialized_json(
        model,
        out_json,
        training_episodes=16,
        num_competitors=15,
        avg_catch_rate=res.mean_catch_rate,
        avg_reward=res.mean_episode_reward,
    )
    assert out_json.exists()
    assert serialized.version == CONTRACT_VERSION
    assert serialized.metadata.observationDim == OBSERVATION_DIM
    assert serialized.metadata.numActions == NUM_DISCRETE_ACTIONS
    assert serialized.onnxModelBase64 is not None

    onnx_bytes = base64.b64decode(serialized.onnxModelBase64)
    session = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])

    sample_obs = torch.randn((3, OBSERVATION_DIM), dtype=torch.float32)
    with torch.no_grad():
        pt_logits = ActorLogitsModule(model).eval()(sample_obs).numpy()

    ort_logits = session.run(["logits"], {"obs": sample_obs.numpy()})[0]
    assert ort_logits.shape == (3, NUM_DISCRETE_ACTIONS)
    assert np.allclose(pt_logits, ort_logits, atol=1e-5)
