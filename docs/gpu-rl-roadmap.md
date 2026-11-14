# GPU Reinforcement Learning Roadmap (Browser Q-Learning → CUDA Batched PPO → Mobile ONNX)

This document defines the architecture and phased migration plan for scaling **Crowded Lineup** from browser-based tabular Q-learning (MVP) to large-scale multi-agent Proximal Policy Optimization (PPO) on NVIDIA GPUs, while running trained neural policies directly in the existing mobile PWA without rewriting the game or simulation contracts.

---

## 1. Architectural Principles Implemented in MVP

The TypeScript MVP was designed from day one around six forward-compatibility invariants:

1. **Data-Oriented State Layout (`WorldState`)**:
   All dynamic entity state in `src/sim/world.ts` is represented as flat numeric arrays (`surferX`, `surferY`, `surferVx`, `surferVy`, `surferStamina`, `surferState`, `waveY`, `wavePeakX`, `waveBreakWidth`, `wwIntensity`, etc.) rather than nested object graphs.
2. **Strict Separation of Simulation and Rendering**:
   `src/sim/world.ts` has zero DOM, Canvas, or browser API dependencies. `OceanCanvasRenderer` (`src/render/canvasRenderer.ts`) is a pure consumer of `WorldState`.
3. **Versioned Serialization Contracts (`CONTRACT_VERSION = 1`)**:
   Every contract type in `src/contracts/types.ts` (`WorldConfig`, `WorldState`, `Observation`, `SurferAction`, `RewardBreakdown`, `StepResult`, `EpisodeMetrics`, `PolicyMetadata`, `SerializedPolicy`, `ReferenceScenario`) carries `version: 1`.
4. **Canonical Continuous Normalized Observations (`OBSERVATION_DIM = 10`)**:
   `observe(world, surferId)` outputs a 10-dimensional continuous feature vector bounded in `[-1, 1]` / `[0, 1]`. Tabular Q-learning (`src/policies/qTable.ts`) discretizes these continuous features *internally* via `discretizeObservation(obs)`, leaving the canonical representation ready for neural network input layers.
5. **Unified `Policy` Interface**:
   Gameplay and evaluation interact exclusively with `Policy.selectAction(observation, world, surferId)` (`src/policies/policy.ts`), making `HumanPolicy`, `RuleBotPolicy`, `QTablePolicy`, and `NeuralOnnxPolicyPlaceholder` drop-in interchangeable.
6. **Deterministic Reference Scenarios for Cross-Runtime Parity**:
   `tests/fixtures/reference-scenarios.json` ships multi-step trajectories generated from the seeded Mulberry32 PRNG (`src/sim/prng.ts`), capturing actions, continuous observations, rewards, surfer coordinates, stamina, and episode metrics at every timestep.

---

## 2. Versioned Contracts Specification (`v1`)

All serialized payloads between TypeScript and future Python/PyTorch tooling validate `version === 1`:

| Contract | Source Type | Key Fields & Invariants |
| :--- | :--- | :--- |
| **WorldConfig** | `WorldConfig` | `version: 1`, `width: 160`, `height: 240`, `dt: 0.1`, `numSurfers`, `maxWaves: 6`, `maxWhitewaters: 24`, `paddleSpeedMultiplier: [0.6, 1.6]` |
| **WorldState** | `WorldState` | `version: 1`, `seed: uint32`, `rngState: uint32`, `stepCount`, flat numeric arrays of length `numSurfers`, `maxWaves`, `maxWhitewaters` |
| **Observation** | `Observation` | `version: 1`, `surferId`, `features: number[10]` (`relPeakX`, `timeToWave`, `relCrossShore`, `stamina`, `competitorPressure`, `waveQuality`, `waveBreakingProgress`, `surferVxNorm`, `surferVyNorm`, `surferStatusNorm`) |
| **Action** | `ActionCode` / `SurferAction` | `version: 1`, discrete `0..5` (`WAIT=0`, `LEFT=1`, `RIGHT=2`, `OUT=3`, `IN=4`, `CATCH=5`) |
| **Reward** | `RewardBreakdown` / `StepResult` | `version: 1`, `catchReward`, `qualityBonus`, `priorityBonus`, `rideReward`, `collisionPenalty`, `dropInPenalty`, `staminaPenalty`, `shapingReward` |
| **EpisodeMetrics** | `EpisodeMetrics` | `version: 1`, `totalWavesSpawned`, `surferCatches[]`, `surferCatchRates[]`, `surferScores[]`, `surferCollisions[]`, `surferDropIns[]` |
| **PolicyMetadata** | `PolicyMetadata` | `version: 1`, `policyType`, `observationVersion: 1`, `observationDim: 10`, `actionVersion: 1`, `numActions: 6` |

---

## 3. Data-Oriented Layout: TypeScript to CUDA Tensor Mapping

Because `WorldState` avoids pointer-heavy object trees, the future Python GPU simulator (`VectorizedLineupEnv`) maps each `WorldState` array directly to a contiguous PyTorch `torch.Tensor` on `cuda:0` with leading dimension `N` (`num_envs = 4096` by default):

| TypeScript `WorldState` Array | PyTorch CUDA Tensor Shape (`N = 4096`, `S = num_surfers`, `W = 6`, `F = 24`) | Dtype |
| :--- | :--- | :--- |
| `rngState` | `[N]` | `torch.int64` (masked `uint32`) |
| `surferX`, `surferY`, `surferVx`, `surferVy`, `surferStamina` | `[N, S]` | `torch.float32` |
| `surferState`, `surferRidingWaveId` | `[N, S]` | `torch.int32` |
| `waveActive`, `waveRiderId` | `[N, W]` | `torch.int32` |
| `waveY`, `waveSpeed`, `wavePeakX`, `wavePeakDriftVx`, `waveQuality`, `waveBreakStartY`, `waveBreakSpreadRate`, `waveBreakWidth` | `[N, W]` | `torch.float32` |
| `wwActive` | `[N, F]` | `torch.int32` |
| `wwX`, `wwY`, `wwWidth`, `wwIntensity`, `wwAge`, `wwMaxAge` | `[N, F]` | `torch.float32` |
| `observe()` output | `[N, S, 10]` | `torch.float32` |
| `step()` actions input | `[N, S]` | `torch.int64` |
| `step()` rewards output | `[N, S]` | `torch.float32` |

---

## 4. Cross-Implementation Parity Strategy (TypeScript ↔ PyTorch/CUDA)

The GPU simulator is a separate vectorized implementation of the simulation rules — TypeScript does not run on CUDA. To guarantee that policies trained in PyTorch behave identically in the mobile browser:

1. **Bit-Exact Mulberry32 PRNG**:
   `src/sim/prng.ts` uses 32-bit unsigned integer arithmetic (`Math.imul`, unsigned shifts `>>>`). The Python test harness implements the exact same 32-bit bit-masked transitions in `torch.int64` (`& 0xFFFFFFFF`).
2. **Fixture Replay Tests (`tests/fixtures/reference-scenarios.json`)**:
   - The TypeScript repo generates and verifies deterministic multi-step scenarios (`solo-wave-catch`, `crowded-lineup-three-surfers`, `stamina-depletion-and-recovery`) via `src/sim/scenarios.ts`.
   - The Python test suite (`pytest tests/test_ts_parity.py`) loads `tests/fixtures/reference-scenarios.json`, initializes `VectorizedLineupEnv(num_envs=len(scenarios))`, feeds `step.actions` at each step, and asserts `torch.allclose(obs, expectedObservations, atol=1e-4)` and `torch.allclose(rewards, expectedRewards, atol=1e-4)`.
3. **CI Parity Gate**:
   Any modification to wave physics, catch windows, stamina rates, or reward weights must regenerate fixtures and pass both `npm run test` (Vitest) and `pytest` before merging.

---

## 5. Future Python GPU Training Module Design (Not Implemented in MVP)

When Phase 2 begins, a standalone `python_rl/` package will be added with the following design:

### 5.1 Batched CUDA Environment (`python_rl/env_cuda.py`)
- **Configurable Parallelism**: Defaults to `num_envs = 4096` on a single NVIDIA GPU (configurable from `64` for CPU debugging up to `16384` on A100/H100).
- **Zero Host-Device Transfers During Rollouts**: Environment steps, observation normalization, reward computation, and auto-resets on `stepCount == maxSteps` execute entirely on the GPU via `torch.compile` / Triton fused kernels.

### 5.2 PPO Architecture (`python_rl/ppo.py`)
- **Actor-Critic MLP**:
  - Input: `Observation.features` (`dim = 10`, `CONTRACT_VERSION = 1`).
  - Trunk: `Linear(10, 128) -> SiLU -> Linear(128, 128) -> SiLU`.
  - Policy Head: `Linear(128, 6)` producing categorical logits over `[WAIT, LEFT, RIGHT, OUT, IN, CATCH]`.
  - Value Head: `Linear(128, 1)` estimating expected discounted return.
- **Hyperparameters**:
  - Rollout horizon `T = 128` steps across `N = 4096` envs (`524,288` transitions per update).
  - GAE `lambda = 0.95`, discount `gamma = 0.98`, clip ratio `0.2`, entropy coefficient `0.01`.

### 5.3 Multi-Agent Curriculum & Self-Play (`python_rl/self_play.py`)
- **Stage 1 (Solo Wave Mastery)**: `num_surfers = 1`, `simplified_waves = True` until catch rate exceeds `40%`.
- **Stage 2 (Vs Rule Bots)**: `num_surfers = 4` where surfers `1..3` run vectorized `RuleBotPolicy` logic.
- **Stage 3 (Historical League Self-Play)**:
  - Surfer `0` is the active PPO learner.
  - Surfers `1..3` sample policies from a league pool (70% recent checkpoints, 20% historical snapshots, 10% deterministic rule bots) to prevent strategy cycling and master priority/drop-in avoidance in crowded lineups.

### 5.4 ONNX Export & Compatibility Envelope (`python_rl/export_onnx.py`)
- Exports the trained actor network via `torch.onnx.export` (opset 17) with input tensor `obs: float32[1, 10]` and output tensor `logits: float32[1, 6]`.
- Packages the `.onnx` binary (or base64 payload) inside a versioned `SerializedPolicy` JSON envelope:
  ```json
  {
    "version": 1,
    "metadata": {
      "version": 1,
      "name": "PPO-SelfPlay-Gen42",
      "policyType": "neural-onnx",
      "observationVersion": 1,
      "observationDim": 10,
      "actionVersion": 1,
      "numActions": 6
    },
    "onnxModelBase64": "<base64-encoded-onnx>"
  }
  ```
- `NeuralOnnxPolicyPlaceholder.validateCompatibility()` (`src/policies/neuralPlaceholder.ts`) verifies `version`, `observationVersion`, `observationDim === 10`, and `numActions === 6` before loading.

---

## 6. Phased Migration Steps

1. **Phase 1 (Complete — Current MVP)**:
   - Data-oriented TypeScript simulation (`src/sim/world.ts`), versioned contracts (`src/contracts/types.ts`), continuous 10D observations, reference scenario fixtures (`tests/fixtures/reference-scenarios.json`), browser Web Worker Q-learning (`src/training/trainingWorker.ts`), and `NeuralOnnxPolicyPlaceholder`.
2. **Phase 2 (Vectorized PyTorch Simulator & Parity Verification)**:
   - Implement `python_rl/env_cuda.py` and verify 100% numerical parity against `tests/fixtures/reference-scenarios.json`.
3. **Phase 3 (GPU PPO + Self-Play Training & ONNX Export)**:
   - Train multi-agent PPO across 4096 parallel CUDA environments with league self-play and export `.onnx` + `SerializedPolicy` v1 metadata.
4. **Phase 4 (Mobile Browser ONNX Inference)**:
   - Add `onnxruntime-web` (WASM/WebGPU backend) inside `src/policies/neuralPlaceholder.ts` to run the exported 128x128 MLP in `< 0.2ms` per step on mobile Chrome.
