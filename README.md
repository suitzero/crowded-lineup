# Crowded Lineup

Mobile-first pixel-art surfing strategy game with built-in reinforcement learning.

Compete with AI surfers to catch incoming waves. Train a tabular Q-learning surfer directly in your browser (Web Worker), save/export its policy, then play against it.

- **TypeScript + Vite + Canvas 2D**, installable PWA with offline app-shell caching, zero backend or runtime dependencies
- **Pure deterministic simulation** (seeded Mulberry32 PRNG, fixed timestep `dt = 0.1s`) shared by gameplay, Web Worker training, and CLI headless training
- **Data-oriented world state**, versioned contracts (`v1`), and continuous normalized observations (`10D`) — designed so a future GPU (PyTorch PPO / CUDA batched envs) trainer can reach numerical parity without rewriting the mobile app. See [docs/gpu-rl-roadmap.md](docs/gpu-rl-roadmap.md) and [SPEC.md](SPEC.md).

---

## Quick Start

```bash
npm install
npm run dev      # Start local Vite dev server
npm run test     # Run deterministic Vitest suite + fixture replay tests
npm run build    # Type-check and build deployable static assets to dist/
npm run train    # Run headless Node Q-learning training
```

---

## Gameplay & Controls

The ocean is viewed directly from above on a `160x240` pixel-art canvas:
- **Offshore (Top)**: Incoming unbroken swells form here and travel toward the bottom.
- **Lineup / Break Zone (Middle)**: Swells steepen and break starting at the peak (marked on unbroken swells) and peel laterally into green shoulders and foaming whitewater trails.
- **Shoreline (Bottom)**: Shallow water and sandy beach where rides finish.

### Mobile Touch Controls
1. **Set Target**: Tap or drag anywhere on the ocean canvas to place your target reticle.
2. **HOLD PADDLE**: Press and hold the **HOLD PADDLE** button to accelerate toward your target. Paddling depletes stamina; resting recovers stamina.
3. **CATCH WAVE**: When positioned near a breaking peak or green peeling shoulder as the crest reaches you, tap **CATCH WAVE** to take off and ride.
4. **Paddle Speed & Rivals**: Adjust the **Paddle Speed** slider (`0.6x–1.6x`) and toggle between **Rule Bots** and **Trained AI + Bots** at the bottom of the Play tab.

### Desktop Keyboard Controls
- **Move / Paddle**: `Arrow Keys` or `W` / `A` / `S` / `D` (`Up/W` = offshore `OUT`, `Down/S` = shoreward `IN`, `Left/A` = `LEFT`, `Right/D` = `RIGHT`)
- **Catch Wave**: `Space` or `Enter`
- **Audio & Mute**: BGM: '겨울바람' (백순진) 8-bit arrangement, synthesized procedurally via Web Audio API. Toggle the `🔊` / `🔇` mute button in the HUD bar (saved in `localStorage`).

### Priority & Etiquette Rules
- **Inside Priority**: The surfer closest to the wave peak when taking off holds right-of-way.
- **Drop-In Penalty**: Taking off on the shoulder in front of a surfer who already has inside priority triggers a drop-in violation (`-8.0` reward penalty and wipeout).
- **Collisions**: Getting too close to another surfer causes a collision (`-6.0` penalty).

---

## Browser & Headless RL Training

### Browser Training Tab (Web Worker)
Switch to the **TRAIN AI** tab to train a tabular Q-learning agent in a background Web Worker without blocking the UI:
- **Episode Count**: Choose `1K`, `5K`, or `20K` episodes.
- **Sliders**: Configure **Paddling Speed** (`0.6x–1.6x`) and **Rule-Bot Competitors** (`0–5`).
- **Curriculum**: Automatically starts with simplified solo waves so the policy learns wave interception and takeoff timing before introducing rule-bot competitors.
- **Controls**: **TRAIN**, **PAUSE / RESUME**, and **CANCEL**, with a live rolling catch-rate chart.
- **Persistence**:
  - **Save Local / Load Local**: Persist the trained policy in `localStorage`.
  - **Export JSON / Import JSON**: Download or load a versioned (`v1`) policy `.json` file.
  - **Play Against Trained AI**: Immediately jump into the ocean lineup against your trained Q-table surfer.

### Headless CLI Training (`npm run train`)
Train policies from the command line in Node without a browser:

```bash
npm run train -- --episodes 500 --competitors 2 --speed 1.0 --seed 2026 --out policy.json
```
