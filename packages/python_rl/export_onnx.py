"""Export trained PPO actor network to ONNX and versioned `SerializedPolicy` JSON."""

from __future__ import annotations

import base64
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn as nn

from packages.python_rl.ppo import ActorCriticMLP
from packages.python_sim.contracts import (
    CONTRACT_VERSION,
    NUM_DISCRETE_ACTIONS,
    OBSERVATION_DIM,
    PolicyMetadata,
    SerializedPolicy,
)


class ActorLogitsModule(nn.Module):
    """Wraps only the actor trunk + policy head for lightweight browser ONNX inference."""

    def __init__(self, actor_critic: ActorCriticMLP) -> None:
        super().__init__()
        self.trunk = actor_critic.trunk
        self.policy_head = actor_critic.policy_head

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.policy_head(self.trunk(obs))
        return out


def export_actor_to_onnx_bytes(model: ActorCriticMLP) -> bytes:
    """Export actor logits network to ONNX (opset 17) in-memory bytes."""
    actor = ActorLogitsModule(model).eval().cpu()
    dummy_obs = torch.zeros((1, OBSERVATION_DIM), dtype=torch.float32)

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=True) as tmp:
        torch.onnx.export(
            actor,
            (dummy_obs,),
            tmp.name,
            export_params=True,
            opset_version=17,
            input_names=["obs"],
            output_names=["logits"],
            dynamic_axes={"obs": {0: "batch"}, "logits": {0: "batch"}},
            dynamo=False,
        )
        return Path(tmp.name).read_bytes()


def export_actor_to_serialized_json(
    model: ActorCriticMLP,
    out_path: str | Path,
    name: str = "PPO-SelfPlay-Crowded",
    training_episodes: int = 0,
    num_competitors: int = 15,
    avg_catch_rate: float = 0.0,
    avg_reward: float = 0.0,
) -> SerializedPolicy:
    onnx_bytes = export_actor_to_onnx_bytes(model)
    payload = SerializedPolicy(
        version=CONTRACT_VERSION,
        metadata=PolicyMetadata(
            version=CONTRACT_VERSION,
            name=name,
            policyType="neural-onnx",
            observationVersion=CONTRACT_VERSION,
            observationDim=OBSERVATION_DIM,
            actionVersion=CONTRACT_VERSION,
            numActions=NUM_DISCRETE_ACTIONS,
            createdAtIso=datetime.now(UTC).isoformat(),
            trainingEpisodes=training_episodes,
            paddleSpeedMultiplier=1.0,
            numCompetitors=num_competitors,
            avgCatchRate=avg_catch_rate,
            avgReward=avg_reward,
        ),
        onnxModelBase64=base64.b64encode(onnx_bytes).decode("ascii"),
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return payload
