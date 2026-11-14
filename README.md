# Crowded Lineup

Mobile-first pixel-art surfing strategy game with built-in reinforcement learning.

Compete with AI surfers to catch incoming waves. Train a tabular Q-learning surfer
directly in your browser (Web Worker), save/export its policy, then play against it.

- TypeScript + Vite + Canvas 2D, installable PWA, no backend
- Pure deterministic simulation (seeded, fixed timestep) shared by gameplay and training
- Data-oriented world state, versioned contracts, continuous normalized observations —
  designed so a future GPU (PyTorch PPO / CUDA batched envs) trainer can reach parity
  without rewriting the mobile app. See [docs/gpu-rl-roadmap.md](docs/gpu-rl-roadmap.md)

Full build specification: [SPEC.md](SPEC.md)

Status: initial scaffold — implementation in progress.
