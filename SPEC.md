# Crowded Lineup — Build Specification

A lightweight, mobile-first pixel-art surfing strategy game with built-in reinforcement learning.

## Primary goal

A playable game that runs smoothly in Android Chrome, can be installed as a PWA, and lets users train an AI surfer directly in the browser.

## Tech stack

- TypeScript, Vite (vanilla-ts), HTML Canvas 2D, CSS responsive touch controls, `onnxruntime-web`
- Web Worker for tabular Q-learning, Vitest for tests
- Python 3.12 (`uv`, `ruff`, `mypy`, `FastAPI`, `Pydantic v2`), vectorized PyTorch PPO/self-play, Apache Beam, event-time watermark streaming, PostgreSQL 16 + `pgvector` (RLS + append-only audit), MinIO/GCS blob storage, and Terraform GCP IaC (see `README.md` and `docs/adr/`)

## Visual design

- Retro 16-bit-style pixel art, portrait mobile screen, adaptable aspect ratios
- Low-resolution canvas with nearest-neighbor scaling
- Ocean viewed directly from above: offshore at top, shoreline at bottom, incoming swells move toward the bottom
- Irregular whitewater patterns are remnants of breaking waves — NOT mountains or terrain
- Small colorful surfer sprites
- Clear distinction between unbroken swells, breaking crests, and whitewater
- Simple HUD: wave count, catches, stamina, score

## Gameplay

- Player competes with AI-controlled surfers to catch incoming waves
- Touch a target location and hold PADDLE to move toward it; CATCH button attempts takeoff
- Fully usable without a keyboard; keyboard controls may be added for desktop testing
- Adjustable paddling speed, initially 0.6x–1.6x
- Paddling acceleration, stamina depletion and recovery
- Multiple waves with changing peak positions
- Surfers must intercept the correct part of a wave at the correct time
- Catch success depends on position, timing, travel direction, paddling speed, wave quality
- Simple priority and collision rules
- Deterministic rule-based competitor bots

## Simulation architecture

Pure TypeScript simulation module, independent of rendering and browser APIs.

Required API:

- `createWorld(config, seed)`
- `observe(world, surferId)`
- `step(world, actions, dt)`
- `isEpisodeDone(world)`
- `getEpisodeMetrics(world)`

Seeded PRNG. Fixed simulation timestep. No `Date.now()` or `Math.random()` in simulation logic. All gameplay and AI training use exactly the same simulation rules.

## Wave model

Inexpensive procedural model (mathematical curves, simple arrays; no fluid simulation, no WebGL):

- Waves have position, speed, peak location, quality
- Crest moves toward shore; breaking begins per a spatial breaking condition; breaking propagates laterally from the peak
- Whitewater trails persist briefly and fade
- Catch windows are spatial and temporal; takeoff requires satisfying physical catch conditions

## RL architecture

Lightweight tabular Q-learning. Discrete actions: WAIT, LEFT, RIGHT, OUT, IN, CATCH.

Discretize observations (for the Q-table only — see Part 2) using:

- relative peak horizontal position
- estimated time until the wave reaches the surfer
- relative cross-shore position
- stamina
- nearby competitor pressure

Keep the observation table compact. First curriculum: single surfer, simplified waves. Once a viable catching policy is learned, allow training with rule-based competitors.

Rewards:

- valid wave catch: significant positive reward
- wave quality and priority: additional reward
- collision or drop-in: penalty
- wasted stamina: small penalty
- limited potential-based shaping toward viable interception
- terminal rewards must dominate shaping

## Training UI

Dedicated Training tab:

- Episode count: 1K / 5K / 20K
- Paddling speed slider, number of competitors slider
- Train / Pause / Cancel buttons
- Progress and catch-rate chart
- Save and load trained policies locally; export/import policy as JSON
- Play against trained AI

Training runs without rendering in a Web Worker; UI stays responsive. Training only continues while the app is open. Do not assume unlimited mobile CPU or battery.

## Testing

Deterministic Vitest tests covering:

- Seed reproducibility
- Wave propagation
- Whitewater lifetime
- Surfer movement
- Stamina constraints
- Catch success and failure
- Priority and collisions
- Q-table update
- Policy serialization
- Headless simulation without DOM dependencies

## Delivery

- Working mobile-first application; PWA manifest and offline app-shell caching
- `npm run dev`, `npm run build`, `npm run test`
- `npm run train` for optional Node-based headless training
- README explaining controls and training
- Deployable static assets

## Implementation order

1. Pure deterministic simulation and tests
2. Playable mobile Canvas game
3. Rule-based competitor surfers
4. Training UI and Q-learning in Web Worker
5. Policy persistence and evaluation

Avoid overengineering. No unnecessary dependencies. Prioritize a genuinely playable mobile MVP. If a complex feature threatens completion, implement the simplest correct version and document the limitation.

---

# Part 2 — Future GPU RL compatibility (added 2026-11-14)

Design the mobile game so it can later support large-scale RL on NVIDIA GPUs without rewriting the mobile application.

1. **Data-oriented simulation.** Represent world state with numeric arrays, not deeply nested object graphs.
2. **Rendering entirely separate from simulation.**
3. **Versioned contracts** for: WorldState, Observation, Action, Reward, EpisodeMetrics, Policy metadata. Every serialized form carries a version field.
4. **Continuous normalized observations** are the canonical observation representation. Tabular Q-learning may discretize internally, but discrete bins must NOT be the only available observation representation.
5. **Deterministic reference scenarios**, including expected state transitions and rewards, for future TypeScript/PyTorch parity tests. Ship them as data (e.g. JSON fixtures generated from the TS sim) plus tests that replay them.
6. **Game independent of the RL algorithm.** Interchangeable policies behind one Policy interface: Human input, Rule-based bot, Q-table policy, Neural network policy.
7. **Reserve ONNX policy loading** via onnxruntime-web behind the neural-policy interface, but do NOT add this dependency until neural policy inference is actually needed.
8. **Document a future Python training module** (do not implement it): PyTorch PPO, CUDA batched environments, 4096 configurable parallel environments, self-play support, ONNX policy export, model version and observation compatibility checks.
9. **A GPU simulator will be a separate, vectorized implementation** of the simulation rules. Do not assume TypeScript code runs directly on CUDA. Ensure cross-implementation parity with automated tests (the reference scenarios in item 5).
10. **Do not implement the Python GPU trainer now.** Keep the first mobile release lightweight.

## Part 2 deliverable

- `docs/gpu-rl-roadmap.md` — the migration plan from browser Q-learning to GPU PPO and mobile ONNX inference, covering: the versioned contracts, the data-oriented state layout, the reference-scenario parity test strategy, the future Python module design (PPO, CUDA batched envs, 4096 parallel envs, self-play, ONNX export, version/observation compatibility checks), and the phased migration steps.
