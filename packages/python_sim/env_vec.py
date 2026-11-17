"""Vectorized PyTorch simulation environment matching `src/sim/world.ts` (v2)."""

from __future__ import annotations

import math

import torch

from packages.python_sim.contracts import (
    OBSERVATION_DIM,
    ActionCode,
    EpisodeMetrics,
    SurferStatus,
    WorldConfig,
)
from packages.python_sim.prng import next_random_batch, normalize_seed


def _clamp_scalar(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


class VectorizedLineupEnv:
    """Batched PyTorch implementation of Crowded Lineup (`[N, ...]`).

    Designed for bit-exact PRNG parity and `atol=1e-4` trajectory parity with
    the TypeScript simulation while scaling to `num_envs = 4096` on CUDA or CPU.
    """

    def __init__(
        self,
        config: WorldConfig | None = None,
        num_envs: int = 1,
        device: str | torch.device = "cpu",
        float_dtype: torch.dtype = torch.float64,
    ) -> None:
        self.config = (config or WorldConfig()).model_copy()
        self.config.paddleSpeedMultiplier = _clamp_scalar(
            self.config.paddleSpeedMultiplier, 0.6, 1.6
        )
        self.config.numSurfers = max(1, int(self.config.numSurfers))

        self.num_envs = num_envs
        self.device = torch.device(device)
        self.float_dtype = float_dtype
        self.S = self.config.numSurfers
        self.W = self.config.maxWaves
        self.F = self.config.maxWhitewaters

        self.rng_state = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
        self.seeds = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
        self.step_count = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
        self.time = torch.zeros(num_envs, dtype=torch.float64, device=self.device)
        self.total_waves_spawned = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
        self.next_wave_spawn_time = torch.full(
            (num_envs,), 4.5, dtype=torch.float64, device=self.device
        )

        # Surfer state tensors [N, S]
        self.surfer_x = torch.zeros((num_envs, self.S), dtype=self.float_dtype, device=self.device)
        self.surfer_y = torch.zeros((num_envs, self.S), dtype=self.float_dtype, device=self.device)
        self.surfer_vx = torch.zeros((num_envs, self.S), dtype=self.float_dtype, device=self.device)
        self.surfer_vy = torch.zeros((num_envs, self.S), dtype=self.float_dtype, device=self.device)
        self.surfer_stamina = torch.ones(
            (num_envs, self.S), dtype=self.float_dtype, device=self.device
        )
        self.surfer_state = torch.zeros((num_envs, self.S), dtype=torch.int32, device=self.device)
        self.surfer_riding_wave_id = torch.full(
            (num_envs, self.S), -1, dtype=torch.int32, device=self.device
        )
        self.surfer_wipeout_timer = torch.zeros(
            (num_envs, self.S), dtype=self.float_dtype, device=self.device
        )
        self.surfer_score = torch.zeros((num_envs, self.S), dtype=torch.float64, device=self.device)
        self.surfer_catches = torch.zeros((num_envs, self.S), dtype=torch.int32, device=self.device)
        self.surfer_collisions = torch.zeros(
            (num_envs, self.S), dtype=torch.int32, device=self.device
        )
        self.surfer_drop_ins = torch.zeros(
            (num_envs, self.S), dtype=torch.int32, device=self.device
        )
        self.surfer_stamina_used = torch.zeros(
            (num_envs, self.S), dtype=torch.float64, device=self.device
        )
        self.surfer_ride_time = torch.zeros(
            (num_envs, self.S), dtype=torch.float64, device=self.device
        )
        self.surfer_last_potential = torch.zeros(
            (num_envs, self.S), dtype=self.float_dtype, device=self.device
        )

        # Wave pool tensors [N, W]
        self.wave_active = torch.zeros((num_envs, self.W), dtype=torch.int32, device=self.device)
        self.wave_id = torch.zeros((num_envs, self.W), dtype=torch.int32, device=self.device)
        self.wave_y = torch.zeros((num_envs, self.W), dtype=self.float_dtype, device=self.device)
        self.wave_speed = torch.zeros(
            (num_envs, self.W), dtype=self.float_dtype, device=self.device
        )
        self.wave_peak_x = torch.zeros(
            (num_envs, self.W), dtype=self.float_dtype, device=self.device
        )
        self.wave_peak_drift_vx = torch.zeros(
            (num_envs, self.W), dtype=self.float_dtype, device=self.device
        )
        self.wave_quality = torch.zeros(
            (num_envs, self.W), dtype=self.float_dtype, device=self.device
        )
        self.wave_break_start_y = torch.zeros(
            (num_envs, self.W), dtype=self.float_dtype, device=self.device
        )
        self.wave_break_spread_rate = torch.zeros(
            (num_envs, self.W), dtype=self.float_dtype, device=self.device
        )
        self.wave_break_width = torch.zeros(
            (num_envs, self.W), dtype=self.float_dtype, device=self.device
        )
        self.wave_rider_id = torch.full(
            (num_envs, self.W), -1, dtype=torch.int32, device=self.device
        )

        # Whitewater pool tensors [N, F]
        self.ww_active = torch.zeros((num_envs, self.F), dtype=torch.int32, device=self.device)
        self.ww_x = torch.zeros((num_envs, self.F), dtype=self.float_dtype, device=self.device)
        self.ww_y = torch.zeros((num_envs, self.F), dtype=self.float_dtype, device=self.device)
        self.ww_width = torch.zeros((num_envs, self.F), dtype=self.float_dtype, device=self.device)
        self.ww_intensity = torch.zeros(
            (num_envs, self.F), dtype=self.float_dtype, device=self.device
        )
        self.ww_age = torch.zeros((num_envs, self.F), dtype=self.float_dtype, device=self.device)
        self.ww_max_age = torch.full(
            (num_envs, self.F), 3.2, dtype=self.float_dtype, device=self.device
        )

    def _rand01(self, env_mask: torch.Tensor | None = None) -> torch.Tensor:
        floats, next_states = next_random_batch(self.rng_state)
        if env_mask is None:
            self.rng_state = next_states
            return floats.to(self.float_dtype)
        self.rng_state = torch.where(env_mask, next_states, self.rng_state)
        return floats.to(self.float_dtype)

    def _rand_range(
        self, lo: float, hi: float, env_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return lo + self._rand01(env_mask) * (hi - lo)

    def _spawn_wave(self, initial_y: float, env_mask: torch.Tensor) -> None:
        """Spawn a wave in the first free slot for environments where `env_mask` is True."""
        if not bool(env_mask.any()):
            return

        cfg = self.config
        free_mask = self.wave_active == 0  # [N, W]
        has_slot = free_mask.any(dim=1) & env_mask
        if not bool(has_slot.any()):
            return

        slot_idx = free_mask.to(torch.int64).argmax(dim=1)  # [N]

        if cfg.simplifiedWaves:
            speed = torch.full((self.num_envs,), 22.0, dtype=self.float_dtype, device=self.device)
            peak_x = self._rand_range(cfg.width * 0.3, cfg.width * 0.7, has_slot)
            drift_vx = torch.zeros((self.num_envs,), dtype=self.float_dtype, device=self.device)
            quality = torch.full((self.num_envs,), 0.85, dtype=self.float_dtype, device=self.device)
            break_start_y = torch.full(
                (self.num_envs,), 75.0, dtype=self.float_dtype, device=self.device
            )
            spread_rate = torch.full(
                (self.num_envs,), 0.75, dtype=self.float_dtype, device=self.device
            )
        else:
            speed = self._rand_range(20.0, 26.0, has_slot)
            peak_x = self._rand_range(26.0, cfg.width - 26.0, has_slot)
            drift_vx = self._rand_range(-3.2, 3.2, has_slot)
            quality = self._rand_range(0.4, 1.0, has_slot)
            break_start_y = 88.0 - quality * 20.0
            spread_rate = self._rand_range(0.65, 0.95, has_slot)

        break_w = torch.where(
            initial_y >= break_start_y,
            6.0 + (initial_y - break_start_y) * spread_rate,
            torch.zeros_like(break_start_y),
        )

        env_ids = torch.nonzero(has_slot, as_tuple=False).squeeze(-1)
        slots = slot_idx[env_ids]
        self.total_waves_spawned[env_ids] += 1

        self.wave_active[env_ids, slots] = 1
        self.wave_id[env_ids, slots] = self.total_waves_spawned[env_ids].to(torch.int32)
        self.wave_y[env_ids, slots] = initial_y
        self.wave_speed[env_ids, slots] = speed[env_ids]
        self.wave_peak_x[env_ids, slots] = peak_x[env_ids]
        self.wave_peak_drift_vx[env_ids, slots] = drift_vx[env_ids]
        self.wave_quality[env_ids, slots] = quality[env_ids]
        self.wave_break_start_y[env_ids, slots] = break_start_y[env_ids]
        self.wave_break_spread_rate[env_ids, slots] = spread_rate[env_ids]
        self.wave_break_width[env_ids, slots] = break_w[env_ids]
        self.wave_rider_id[env_ids, slots] = -1

    def _spawn_whitewater(
        self, env_id: int, x: float, y: float, width: float, intensity: float = 0.95
    ) -> None:
        slot = -1
        oldest_slot = 0
        oldest_ratio = -1.0
        for i in range(self.F):
            if int(self.ww_active[env_id, i].item()) == 0:
                slot = i
                break
            ratio = float(self.ww_age[env_id, i].item()) / max(
                0.1, float(self.ww_max_age[env_id, i].item())
            )
            if ratio > oldest_ratio:
                oldest_ratio = ratio
                oldest_slot = i
        if slot == -1:
            slot = oldest_slot

        self.ww_active[env_id, slot] = 1
        self.ww_x[env_id, slot] = x
        self.ww_y[env_id, slot] = y
        self.ww_width[env_id, slot] = width
        self.ww_intensity[env_id, slot] = _clamp_scalar(intensity, 0.1, 1.0)
        self.ww_age[env_id, slot] = 0.0
        self.ww_max_age[env_id, slot] = 3.2

    def reset(self, seeds: list[int] | int = 42) -> torch.Tensor:
        """Reset all environments with deterministic seeds matching `createWorld`."""
        if isinstance(seeds, int):
            seed_list = [normalize_seed(seeds + i) for i in range(self.num_envs)]
        else:
            seed_list = [normalize_seed(s) for s in seeds]

        self.seeds = torch.tensor(seed_list, dtype=torch.int64, device=self.device)
        self.rng_state = self.seeds.clone()
        self.step_count.zero_()
        self.time.zero_()
        self.total_waves_spawned.zero_()
        self.next_wave_spawn_time.fill_(4.5)

        self.surfer_x.zero_()
        self.surfer_y.zero_()
        self.surfer_vx.zero_()
        self.surfer_vy.zero_()
        self.surfer_stamina.fill_(1.0)
        self.surfer_state.fill_(SurferStatus.FREE)
        self.surfer_riding_wave_id.fill_(-1)
        self.surfer_wipeout_timer.zero_()
        self.surfer_score.zero_()
        self.surfer_catches.zero_()
        self.surfer_collisions.zero_()
        self.surfer_drop_ins.zero_()
        self.surfer_stamina_used.zero_()
        self.surfer_ride_time.zero_()
        self.surfer_last_potential.zero_()

        self.wave_active.zero_()
        self.wave_id.zero_()
        self.wave_y.zero_()
        self.wave_speed.zero_()
        self.wave_peak_x.zero_()
        self.wave_peak_drift_vx.zero_()
        self.wave_quality.zero_()
        self.wave_break_start_y.zero_()
        self.wave_break_spread_rate.zero_()
        self.wave_break_width.zero_()
        self.wave_rider_id.fill_(-1)

        self.ww_active.zero_()
        self.ww_x.zero_()
        self.ww_y.zero_()
        self.ww_width.zero_()
        self.ww_intensity.zero_()
        self.ww_age.zero_()
        self.ww_max_age.fill_(3.2)

        cfg = self.config
        n = self.S
        tiers = 3 if n >= 8 else 1
        tier_y_centers = [74.0, 93.0, 112.0] if tiers == 3 else [92.0]
        all_envs = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        for i in range(n):
            tier_idx = i % tiers
            slot_in_tier = i // tiers
            count_in_tier = max(1, math.ceil((n - tier_idx) / tiers))
            lane_frac = 0.5 if count_in_tier == 1 else (slot_in_tier + 0.5) / count_in_tier
            base_x = 22.0 + lane_frac * (cfg.width - 44.0)
            jx = (
                self._rand_range(-18.0, 18.0, all_envs)
                if n == 1
                else self._rand_range(-6.0, 6.0, all_envs)
            )
            jy = self._rand_range(-6.0, 6.0, all_envs)
            self.surfer_x[:, i] = torch.clamp(base_x + jx, 16.0, cfg.width - 16.0)
            self.surfer_y[:, i] = torch.clamp(tier_y_centers[tier_idx] + jy, 64.0, 130.0)

        # Deterministic pairwise relaxation pass
        min_spawn_dist = 8.5
        if n > 1:
            for _ in range(6):
                for i in range(n):
                    for j in range(i + 1, n):
                        dx = self.surfer_x[:, j] - self.surfer_x[:, i]
                        dy = self.surfer_y[:, j] - self.surfer_y[:, i]
                        dist_sq = dx * dx + dy * dy
                        too_close = dist_sq < (min_spawn_dist * min_spawn_dist)
                        if bool(too_close.any()):
                            dist = torch.clamp(torch.sqrt(dist_sq), min=0.1)
                            push = (min_spawn_dist - dist) * 0.55
                            zero_dist = dist_sq < 0.01
                            fallback_nx = 1.0 if j % 2 == 0 else -1.0
                            nx = torch.where(zero_dist, torch.full_like(dx, fallback_nx), dx / dist)
                            ny = torch.where(zero_dist, torch.zeros_like(dy), dy / dist)
                            self.surfer_x[:, i] = torch.where(
                                too_close,
                                torch.clamp(
                                    self.surfer_x[:, i] - nx * push, 14.0, cfg.width - 14.0
                                ),
                                self.surfer_x[:, i],
                            )
                            self.surfer_y[:, i] = torch.where(
                                too_close,
                                torch.clamp(self.surfer_y[:, i] - ny * push, 62.0, 132.0),
                                self.surfer_y[:, i],
                            )
                            self.surfer_x[:, j] = torch.where(
                                too_close,
                                torch.clamp(
                                    self.surfer_x[:, j] + nx * push, 14.0, cfg.width - 14.0
                                ),
                                self.surfer_x[:, j],
                            )
                            self.surfer_y[:, j] = torch.where(
                                too_close,
                                torch.clamp(self.surfer_y[:, j] + ny * push, 62.0, 132.0),
                                self.surfer_y[:, j],
                            )

        self._spawn_wave(34.0, all_envs)
        self._spawn_wave(8.0, all_envs)
        if cfg.width >= 240.0:
            self._spawn_wave(20.0, all_envs)

        self.surfer_last_potential = self._compute_interception_potential()
        return self.observe()

    def _find_top_two_wave_slots(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Vectorized primary and secondary target wave selection across `[N, S, W]`."""
        cfg = self.config
        active = self.wave_active[:, None, :] == 1  # [N, 1, W]
        wy = self.wave_y[:, None, :]  # [N, 1, W]
        sy = self.surfer_y[:, :, None]  # [N, S, 1]
        sx = self.surfer_x[:, :, None]  # [N, S, 1]
        wspeed = torch.clamp(self.wave_speed[:, None, :], min=10.0)
        wpeak = self.wave_peak_x[:, None, :]
        wqual = self.wave_quality[:, None, :]

        valid = active & (wy <= sy + 8.0) & (wy <= cfg.height - 40.0)
        dy = torch.clamp(sy - wy, min=0.0)
        time_to_reach = dy / wspeed
        dx = torch.abs(sx - wpeak)
        score = time_to_reach * 18.0 + dx * 0.25 - wqual * 4.0
        masked_score = torch.where(valid, score, torch.full_like(score, 1e9))

        top2_vals, top2_idx = torch.topk(masked_score, k=min(2, self.W), dim=2, largest=False)
        best_slot = torch.where(
            top2_vals[:, :, 0] < 1e8,
            top2_idx[:, :, 0],
            torch.full((self.num_envs, self.S), -1, dtype=torch.int64, device=self.device),
        )

        if self.W >= 2:
            second_slot = torch.where(
                top2_vals[:, :, 1] < 1e8,
                top2_idx[:, :, 1],
                torch.full((self.num_envs, self.S), -1, dtype=torch.int64, device=self.device),
            )
        else:
            second_slot = torch.full(
                (self.num_envs, self.S), -1, dtype=torch.int64, device=self.device
            )

        # Fallback: first active wave if no wave satisfied catchability filter
        has_any_active = active.any(dim=2).expand(-1, self.S)
        first_active = active.to(torch.int64).argmax(dim=2).expand(-1, self.S)
        need_fallback = (best_slot == -1) & has_any_active
        best_slot = torch.where(need_fallback, first_active, best_slot)

        return best_slot, second_slot

    def _compute_interception_potential(self) -> torch.Tensor:
        cfg = self.config
        best_slot, _ = self._find_top_two_wave_slots()
        safe_slot = torch.clamp(best_slot, min=0)

        peak_x = torch.gather(self.wave_peak_x, 1, safe_slot)
        break_start = torch.gather(self.wave_break_start_y, 1, safe_slot)
        target_y = torch.clamp(break_start + 10.0, 65.0, 125.0)

        norm_dx = torch.abs(self.surfer_x - peak_x) / (cfg.width * 0.5)
        norm_dy = torch.abs(self.surfer_y - target_y) / (cfg.height * 0.5)
        dist_norm = torch.sqrt(norm_dx * norm_dx + norm_dy * norm_dy)
        pot = torch.clamp(1.2 * (1.0 - torch.clamp(dist_norm, 0.0, 1.0)), min=0.0)

        valid = (self.surfer_state == SurferStatus.FREE) & (best_slot != -1)
        return torch.where(valid, pot, torch.zeros_like(pot))

    def observe(self) -> torch.Tensor:
        """Compute canonical 16D normalized continuous observations `[N, S, 16]`."""
        cfg = self.config
        best_slot, second_slot = self._find_top_two_wave_slots()

        has_primary = best_slot != -1
        safe_primary = torch.clamp(best_slot, min=0)
        peak_x = torch.gather(self.wave_peak_x, 1, safe_primary)
        wy = torch.gather(self.wave_y, 1, safe_primary)
        wspeed = torch.clamp(torch.gather(self.wave_speed, 1, safe_primary), min=10.0)
        break_ref_y = torch.where(
            has_primary,
            torch.gather(self.wave_break_start_y, 1, safe_primary),
            torch.full((self.num_envs, self.S), 85.0, dtype=self.float_dtype, device=self.device),
        )
        wqual = torch.gather(self.wave_quality, 1, safe_primary)
        wbreak_w = torch.clamp(torch.gather(self.wave_break_width, 1, safe_primary), min=1.0)

        rel_peak_x = torch.where(
            has_primary,
            torch.clamp((peak_x - self.surfer_x) / (cfg.width * 0.5), -1.0, 1.0),
            torch.zeros_like(self.surfer_x),
        )
        time_to_wave = torch.where(
            has_primary,
            torch.clamp(((self.surfer_y - wy) / wspeed) / 6.0, -1.0, 1.0),
            torch.ones_like(self.surfer_x),
        )
        wave_quality = torch.where(
            has_primary,
            torch.clamp(wqual, 0.0, 1.0),
            torch.full_like(self.surfer_x, 0.5),
        )
        is_breaking = has_primary & (wy >= break_ref_y)
        wave_breaking_progress = torch.where(
            is_breaking,
            torch.clamp(1.0 - torch.abs(self.surfer_x - peak_x) / (wbreak_w + 18.0), 0.0, 1.0),
            torch.zeros_like(self.surfer_x),
        )

        # Secondary wave peak [14..15]
        has_sec = second_slot != -1
        safe_sec = torch.clamp(second_slot, min=0)
        sec_peak_x = torch.gather(self.wave_peak_x, 1, safe_sec)
        sec_y = torch.gather(self.wave_y, 1, safe_sec)
        sec_speed = torch.clamp(torch.gather(self.wave_speed, 1, safe_sec), min=10.0)
        sec_rel_peak_x = torch.where(
            has_sec,
            torch.clamp((sec_peak_x - self.surfer_x) / (cfg.width * 0.5), -1.0, 1.0),
            torch.zeros_like(self.surfer_x),
        )
        sec_time_to_wave = torch.where(
            has_sec,
            torch.clamp(((self.surfer_y - sec_y) / sec_speed) / 6.0, -1.0, 1.0),
            torch.ones_like(self.surfer_x),
        )

        rel_cross_shore = torch.clamp(
            (self.surfer_y - (break_ref_y + 8.0)) / (cfg.height * 0.4), -1.0, 1.0
        )
        stamina = torch.clamp(self.surfer_stamina, 0.0, 1.0)

        # Pairwise crowd & priority features across [N, S_i, S_j]
        if self.S > 1:
            sx_i = self.surfer_x[:, :, None]
            sy_i = self.surfer_y[:, :, None]
            sx_j = self.surfer_x[:, None, :]
            sy_j = self.surfer_y[:, None, :]
            dx = sx_j - sx_i
            dy = sy_j - sy_i
            dist = torch.sqrt(dx * dx + dy * dy)
            not_self = ~torch.eye(self.S, dtype=torch.bool, device=self.device)[None, :, :]

            p = torch.clamp(1.0 - dist / 45.0, 0.0, 1.0)
            my_peak_dist = torch.abs(self.surfer_x - peak_x)[:, :, None]
            other_peak_dist = torch.abs(sx_j - peak_x[:, :, None])
            closer_to_peak = has_primary[:, :, None] & (other_peak_dist < my_peak_dist)
            p_adj = torch.where(closer_to_peak, torch.clamp(p * 1.35, 0.0, 1.0), p)
            p_adj = torch.where(not_self, p_adj, torch.zeros_like(p_adj))
            competitor_pressure = p_adj.max(dim=2).values

            not_wiped_j = (self.surfer_state != SurferStatus.WIPED_OUT)[:, None, :]
            active_neighbor = not_self & not_wiped_j

            threat_mask = active_neighbor & closer_to_peak & (torch.abs(dy) <= 35.0)
            threat_weight = torch.clamp(1.0 - dist / 60.0, 0.15, 1.0)
            inside_threat = torch.clamp(
                (threat_weight * threat_mask.to(self.float_dtype)).sum(dim=2) / 3.0, 0.0, 1.0
            )

            dy_band = active_neighbor & (torch.abs(dy) <= 40.0)
            left_cnt = (dy_band & (dx >= -45.0) & (dx < -6.0)).sum(dim=2).to(self.float_dtype)
            center_cnt = (dy_band & (dx >= -6.0) & (dx <= 6.0)).sum(dim=2).to(self.float_dtype)
            right_cnt = (dy_band & (dx > 6.0) & (dx <= 45.0)).sum(dim=2).to(self.float_dtype)

            crowd_left = torch.clamp(left_cnt / 6.0, 0.0, 1.0)
            crowd_center = torch.clamp(center_cnt / 4.0, 0.0, 1.0)
            crowd_right = torch.clamp(right_cnt / 6.0, 0.0, 1.0)
        else:
            zeros = torch.zeros_like(self.surfer_x)
            competitor_pressure = zeros
            crowd_left = zeros
            crowd_center = zeros
            crowd_right = zeros
            inside_threat = zeros

        max_speed_ref = 32.0 * cfg.paddleSpeedMultiplier
        vx_norm = torch.clamp(self.surfer_vx / max_speed_ref, -1.0, 1.0)
        vy_norm = torch.clamp(self.surfer_vy / max_speed_ref, -1.0, 1.0)
        status_norm = torch.where(
            self.surfer_state == SurferStatus.FREE,
            torch.zeros_like(self.surfer_x),
            torch.where(
                self.surfer_state == SurferStatus.RIDING,
                torch.full_like(self.surfer_x, 0.5),
                torch.ones_like(self.surfer_x),
            ),
        )

        obs = torch.stack(
            [
                rel_peak_x,
                time_to_wave,
                rel_cross_shore,
                stamina,
                competitor_pressure,
                wave_quality,
                wave_breaking_progress,
                vx_norm,
                vy_norm,
                status_norm,
                crowd_left,
                crowd_center,
                crowd_right,
                inside_threat,
                sec_rel_peak_x,
                sec_time_to_wave,
            ],
            dim=-1,
        )
        assert obs.shape[-1] == OBSERVATION_DIM
        return obs.to(torch.float32)

    def _get_whitewater_foam_at(self, env_id: int, x: float, y: float) -> float:
        max_foam = 0.0
        for i in range(self.F):
            if int(self.ww_active[env_id, i].item()) == 0:
                continue
            dy = abs(y - float(self.ww_y[env_id, i].item()))
            dx = abs(x - float(self.ww_x[env_id, i].item()))
            ww_w = float(self.ww_width[env_id, i].item())
            if dy <= 6.5 and dx <= ww_w * 0.55:
                inten = float(self.ww_intensity[env_id, i].item())
                if inten > max_foam:
                    max_foam = inten
        return max_foam

    def step(
        self, actions: torch.Tensor, dt: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance all `N` environments by `dt`, returning `(obs, rewards, dones)`."""
        cfg = self.config
        effective_dt = dt if (dt is not None and dt > 0) else cfg.dt
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)

        self.step_count += 1
        self.time = torch.round((self.time + effective_dt) * 10000.0) / 10000.0

        catch_reward = torch.zeros(
            (self.num_envs, self.S), dtype=self.float_dtype, device=self.device
        )
        quality_bonus = torch.zeros(
            (self.num_envs, self.S), dtype=self.float_dtype, device=self.device
        )
        priority_bonus = torch.zeros(
            (self.num_envs, self.S), dtype=self.float_dtype, device=self.device
        )
        ride_reward = torch.zeros(
            (self.num_envs, self.S), dtype=self.float_dtype, device=self.device
        )
        collision_penalty = torch.zeros(
            (self.num_envs, self.S), dtype=self.float_dtype, device=self.device
        )
        dropin_penalty = torch.zeros(
            (self.num_envs, self.S), dtype=self.float_dtype, device=self.device
        )
        stamina_penalty = torch.zeros(
            (self.num_envs, self.S), dtype=self.float_dtype, device=self.device
        )

        # 1. Advance existing waves and spawn whitewater trails
        max_break_width = min(cfg.width * 0.65, 96.0)
        for env_id in range(self.num_envs):
            sc = int(self.step_count[env_id].item())
            for w in range(self.W):
                if int(self.wave_active[env_id, w].item()) == 0:
                    continue
                wy = (
                    float(self.wave_y[env_id, w].item())
                    + float(self.wave_speed[env_id, w].item()) * effective_dt
                )
                peak_x = _clamp_scalar(
                    float(self.wave_peak_x[env_id, w].item())
                    + float(self.wave_peak_drift_vx[env_id, w].item()) * effective_dt,
                    20.0,
                    cfg.width - 20.0,
                )
                self.wave_y[env_id, w] = wy
                self.wave_peak_x[env_id, w] = peak_x

                break_start_y = float(self.wave_break_start_y[env_id, w].item())
                if wy >= break_start_y:
                    broken_dist = wy - break_start_y
                    bw = min(
                        max_break_width,
                        6.0 + broken_dist * float(self.wave_break_spread_rate[env_id, w].item()),
                    )
                    self.wave_break_width[env_id, w] = bw
                    if (sc + w) % 4 == 0:
                        self._spawn_whitewater(env_id, peak_x, wy - 2.0, max(14.0, bw * 1.55), 0.95)
                else:
                    self.wave_break_width[env_id, w] = 0.0

                if wy >= cfg.height - 16.0:
                    for i in range(self.S):
                        if (
                            int(self.surfer_state[env_id, i].item()) == SurferStatus.RIDING
                            and int(self.surfer_riding_wave_id[env_id, i].item()) == w
                        ):
                            self.surfer_state[env_id, i] = SurferStatus.FREE
                            self.surfer_riding_wave_id[env_id, i] = -1
                    self.wave_active[env_id, w] = 0
                    self.wave_rider_id[env_id, w] = -1

        # 2. Spawn new waves on schedule
        spawn_due = self.time >= self.next_wave_spawn_time
        if bool(spawn_due.any()):
            self._spawn_wave(8.0, spawn_due)
            if cfg.simplifiedWaves:
                jitter = torch.zeros(self.num_envs, dtype=torch.float64, device=self.device)
            else:
                jitter = self._rand_range(-1.2, 1.5, spawn_due).to(torch.float64)
            next_t = self.time + torch.clamp(cfg.waveSpawnInterval + jitter, min=4.5)
            self.next_wave_spawn_time = torch.where(spawn_due, next_t, self.next_wave_spawn_time)

        # 3. Update whitewater lifetime
        ww_is_active = self.ww_active == 1
        self.ww_age = torch.where(ww_is_active, self.ww_age + effective_dt, self.ww_age)
        self.ww_y = torch.where(ww_is_active, self.ww_y + 4.5 * effective_dt, self.ww_y)
        ww_expired = ww_is_active & (
            (self.ww_age >= self.ww_max_age) | (self.ww_y >= cfg.height - 8.0)
        )
        self.ww_active = torch.where(ww_expired, torch.zeros_like(self.ww_active), self.ww_active)
        self.ww_intensity = torch.where(
            ww_expired,
            torch.zeros_like(self.ww_intensity),
            torch.where(
                ww_is_active,
                torch.clamp(1.0 - self.ww_age / self.ww_max_age, min=0.0),
                self.ww_intensity,
            ),
        )

        # 4 & 5. Process surfers, catch attempts, priority/drop-ins, and collisions per env
        base_max_speed = 26.0 * cfg.paddleSpeedMultiplier
        accel_rate = 85.0 * cfg.paddleSpeedMultiplier
        collision_radius = 6.5

        for env_id in range(self.num_envs):
            catch_attempts: dict[int, list[tuple[int, float, float]]] = {}

            for i in range(self.S):
                act_code = int(actions[env_id, i].item())
                status = int(self.surfer_state[env_id, i].item())

                if status == SurferStatus.WIPED_OUT:
                    timer = float(self.surfer_wipeout_timer[env_id, i].item()) - effective_dt
                    vx = float(self.surfer_vx[env_id, i].item()) * max(
                        0.0, 1.0 - 5.0 * effective_dt
                    )
                    vy = 6.0
                    sx = _clamp_scalar(
                        float(self.surfer_x[env_id, i].item()) + vx * effective_dt,
                        10.0,
                        cfg.width - 10.0,
                    )
                    sy = _clamp_scalar(
                        float(self.surfer_y[env_id, i].item()) + vy * effective_dt,
                        12.0,
                        cfg.height - 18.0,
                    )
                    stam = _clamp_scalar(
                        float(self.surfer_stamina[env_id, i].item()) + 0.08 * effective_dt, 0.0, 1.0
                    )
                    self.surfer_vx[env_id, i] = vx
                    self.surfer_vy[env_id, i] = vy
                    self.surfer_x[env_id, i] = sx
                    self.surfer_y[env_id, i] = sy
                    self.surfer_stamina[env_id, i] = stam
                    if timer <= 0.0:
                        self.surfer_state[env_id, i] = SurferStatus.FREE
                        self.surfer_wipeout_timer[env_id, i] = 0.0
                    else:
                        self.surfer_wipeout_timer[env_id, i] = timer
                    continue

                if status == SurferStatus.RIDING:
                    w = int(self.surfer_riding_wave_id[env_id, i].item())
                    if w < 0 or w >= self.W or int(self.wave_active[env_id, w].item()) == 0:
                        self.surfer_state[env_id, i] = SurferStatus.FREE
                        self.surfer_riding_wave_id[env_id, i] = -1
                    elif act_code == ActionCode.OUT:
                        self.surfer_state[env_id, i] = SurferStatus.FREE
                        self.surfer_riding_wave_id[env_id, i] = -1
                        if int(self.wave_rider_id[env_id, w].item()) == i:
                            self.wave_rider_id[env_id, w] = -1
                        self.surfer_vy[env_id, i] = -10.0
                    else:
                        peak_x = float(self.wave_peak_x[env_id, w].item())
                        sx = float(self.surfer_x[env_id, i].item())
                        shoulder_dir = 1.0 if sx >= peak_x else -1.0
                        lat_speed = shoulder_dir * 14.0
                        if act_code == ActionCode.LEFT:
                            lat_speed = -18.0
                        elif act_code == ActionCode.RIGHT:
                            lat_speed = 18.0

                        wspeed = float(self.wave_speed[env_id, w].item())
                        wy = float(self.wave_y[env_id, w].item())
                        wqual = float(self.wave_quality[env_id, w].item())

                        self.surfer_vx[env_id, i] = lat_speed
                        self.surfer_vy[env_id, i] = wspeed
                        self.surfer_x[env_id, i] = _clamp_scalar(
                            sx + lat_speed * effective_dt, 10.0, cfg.width - 10.0
                        )
                        new_y = _clamp_scalar(wy + 2.0, 12.0, cfg.height - 18.0)
                        self.surfer_y[env_id, i] = new_y
                        self.surfer_stamina[env_id, i] = _clamp_scalar(
                            float(self.surfer_stamina[env_id, i].item()) + 0.12 * effective_dt,
                            0.0,
                            1.0,
                        )
                        self.surfer_ride_time[env_id, i] = round(
                            float(self.surfer_ride_time[env_id, i].item()) + effective_dt, 4
                        )
                        step_ride_score = 1.5 * wqual * effective_dt * 10.0
                        self.surfer_score[env_id, i] = round(
                            float(self.surfer_score[env_id, i].item()) + step_ride_score, 2
                        )
                        ride_reward[env_id, i] += 0.15 * wqual

                        if new_y >= cfg.height - 26.0:
                            self.surfer_state[env_id, i] = SurferStatus.FREE
                            self.surfer_riding_wave_id[env_id, i] = -1
                            if int(self.wave_rider_id[env_id, w].item()) == i:
                                self.wave_rider_id[env_id, w] = -1
                        continue

                # FREE state
                dir_x = 0.0
                dir_y = 0.0
                is_paddling = False
                if act_code == ActionCode.LEFT:
                    dir_x = -1.0
                    is_paddling = True
                elif act_code == ActionCode.RIGHT:
                    dir_x = 1.0
                    is_paddling = True
                elif act_code == ActionCode.OUT:
                    dir_y = -1.0
                    is_paddling = True
                elif act_code == ActionCode.IN or act_code == ActionCode.CATCH:
                    dir_y = 1.0
                    is_paddling = True

                sx = float(self.surfer_x[env_id, i].item())
                sy = float(self.surfer_y[env_id, i].item())
                vx = float(self.surfer_vx[env_id, i].item())
                vy = float(self.surfer_vy[env_id, i].item())
                stam = float(self.surfer_stamina[env_id, i].item())

                stamina_factor = (
                    0.32 if stam <= 0.01 else (0.35 + 0.65 * (stam / 0.25) if stam < 0.25 else 1.0)
                )
                foam_intensity = self._get_whitewater_foam_at(env_id, sx, sy)
                foam_penalty = 1.0 - 0.35 * foam_intensity if foam_intensity > 0 else 1.0
                eff_max_speed = base_max_speed * stamina_factor * foam_penalty

                if is_paddling:
                    target_vx = dir_x * eff_max_speed
                    target_vy = dir_y * eff_max_speed
                    dvx = target_vx - vx
                    dvy = target_vy - vy
                    max_dv = accel_rate * effective_dt
                    dv_mag = math.sqrt(dvx * dvx + dvy * dvy)
                    if dv_mag <= max_dv or dv_mag == 0.0:
                        vx = target_vx
                        vy = target_vy
                    else:
                        vx += (dvx / dv_mag) * max_dv
                        vy += (dvy / dv_mag) * max_dv

                    extra_foam = (
                        0.09 * foam_intensity if (dir_y < 0 and foam_intensity > 0) else 0.0
                    )
                    drain = (0.11 + extra_foam) * effective_dt
                    new_stam = _clamp_scalar(stam - drain, 0.0, 1.0)
                    self.surfer_stamina[env_id, i] = new_stam
                    self.surfer_stamina_used[env_id, i] = round(
                        float(self.surfer_stamina_used[env_id, i].item()) + (stam - new_stam), 4
                    )
                    stamina_penalty[env_id, i] -= 0.015
                else:
                    drag = max(0.0, 1.0 - 4.5 * effective_dt)
                    vx *= drag
                    vy *= drag
                    if abs(vx) < 0.05:
                        vx = 0.0
                    if abs(vy) < 0.05:
                        vy = 0.0
                    self.surfer_stamina[env_id, i] = _clamp_scalar(
                        stam + 0.18 * effective_dt, 0.0, 1.0
                    )

                if foam_intensity > 0.35 and not is_paddling:
                    vy += 4.0 * foam_intensity * effective_dt

                sx = _clamp_scalar(sx + vx * effective_dt, 10.0, cfg.width - 10.0)
                sy = _clamp_scalar(sy + vy * effective_dt, 12.0, cfg.height - 18.0)
                self.surfer_vx[env_id, i] = vx
                self.surfer_vy[env_id, i] = vy
                self.surfer_x[env_id, i] = sx
                self.surfer_y[env_id, i] = sy

                if act_code == ActionCode.CATCH:
                    best_w = -1
                    min_crest = float("inf")
                    for w in range(self.W):
                        if int(self.wave_active[env_id, w].item()) == 0:
                            continue
                        d = abs(sy - float(self.wave_y[env_id, w].item()))
                        if d < min_crest:
                            min_crest = d
                            best_w = w
                    if best_w != -1:
                        wy = float(self.wave_y[env_id, best_w].item())
                        peak_x = float(self.wave_peak_x[env_id, best_w].item())
                        break_start = float(self.wave_break_start_y[env_id, best_w].item())
                        break_w = float(self.wave_break_width[env_id, best_w].item())
                        qual = float(self.wave_quality[env_id, best_w].item())
                        cur_stam = float(self.surfer_stamina[env_id, i].item())

                        dy_c = sy - wy
                        dist_peak = abs(sx - peak_x)
                        in_crest = -7.5 <= dy_c <= 9.5
                        in_cross = (break_start - 8.0) <= wy <= (cfg.height - 32.0)
                        max_horiz = max(18.0, break_w + 15.0)
                        in_horiz = dist_peak <= max_horiz
                        deep_closeout = (wy - break_start > 52.0) and (dist_peak < break_w * 0.45)
                        can_catch = (
                            in_crest
                            and in_cross
                            and in_horiz
                            and (not deep_closeout)
                            and (cur_stam >= 0.08)
                            and (vy >= -3.5)
                        )

                        if can_catch:
                            catch_attempts.setdefault(best_w, []).append((i, dist_peak, qual))
                        else:
                            stamina_penalty[env_id, i] -= 0.05
                            breaking_over = (
                                in_crest
                                and (wy >= break_start + 4.0)
                                and (dist_peak <= break_w + 6.0)
                            )
                            if breaking_over and cur_stam < 0.12:
                                self.surfer_state[env_id, i] = SurferStatus.WIPED_OUT
                                self.surfer_wipeout_timer[env_id, i] = 1.0
                    else:
                        stamina_penalty[env_id, i] -= 0.04

            # Resolve wave priority and drop-ins
            for w_slot, attempts in catch_attempts.items():
                attempts.sort(key=lambda tup: tup[1])
                existing_rider = int(self.wave_rider_id[env_id, w_slot].item())
                has_existing = (
                    existing_rider != -1
                    and int(self.surfer_state[env_id, existing_rider].item()) == SurferStatus.RIDING
                    and int(self.surfer_riding_wave_id[env_id, existing_rider].item()) == w_slot
                )
                for idx, (sid, dist_peak, qual) in enumerate(attempts):
                    dropped_in = idx > 0
                    if has_existing and existing_rider != sid:
                        existing_dist = abs(
                            float(self.surfer_x[env_id, existing_rider].item())
                            - float(self.wave_peak_x[env_id, w_slot].item())
                        )
                        if existing_dist <= dist_peak + 6.0:
                            dropped_in = True

                    if dropped_in:
                        self.surfer_drop_ins[env_id, sid] += 1
                        dropin_penalty[env_id, sid] -= 8.0
                        self.surfer_state[env_id, sid] = SurferStatus.WIPED_OUT
                        self.surfer_wipeout_timer[env_id, sid] = 1.2
                        self.surfer_riding_wave_id[env_id, sid] = -1
                    else:
                        self.surfer_state[env_id, sid] = SurferStatus.RIDING
                        self.surfer_riding_wave_id[env_id, sid] = w_slot
                        self.wave_rider_id[env_id, w_slot] = sid
                        self.surfer_catches[env_id, sid] += 1
                        catch_pts = 25.0 + round(qual * 25.0)
                        self.surfer_score[env_id, sid] = round(
                            float(self.surfer_score[env_id, sid].item()) + catch_pts, 2
                        )
                        catch_reward[env_id, sid] += 12.0
                        quality_bonus[env_id, sid] += round(6.0 * qual, 4)
                        priority_bonus[env_id, sid] += 3.0

            # Pairwise surfer collisions
            for i in range(self.S):
                for j in range(i + 1, self.S):
                    if (
                        int(self.surfer_state[env_id, i].item()) == SurferStatus.WIPED_OUT
                        or int(self.surfer_state[env_id, j].item()) == SurferStatus.WIPED_OUT
                    ):
                        continue
                    xi = float(self.surfer_x[env_id, i].item())
                    yi = float(self.surfer_y[env_id, i].item())
                    xj = float(self.surfer_x[env_id, j].item())
                    yj = float(self.surfer_y[env_id, j].item())
                    dx = xj - xi
                    dy = yj - yi
                    dist_sq = dx * dx + dy * dy
                    if dist_sq < collision_radius * collision_radius:
                        dist = max(0.1, math.sqrt(dist_sq))
                        nx = dx / dist
                        ny = dy / dist
                        overlap = (collision_radius - dist) * 0.5 + 0.5
                        self.surfer_x[env_id, i] = _clamp_scalar(
                            xi - nx * overlap, 10.0, cfg.width - 10.0
                        )
                        self.surfer_y[env_id, i] = _clamp_scalar(
                            yi - ny * overlap, 12.0, cfg.height - 18.0
                        )
                        self.surfer_x[env_id, j] = _clamp_scalar(
                            xj + nx * overlap, 10.0, cfg.width - 10.0
                        )
                        self.surfer_y[env_id, j] = _clamp_scalar(
                            yj + ny * overlap, 12.0, cfg.height - 18.0
                        )
                        self.surfer_vx[env_id, i] *= -0.3
                        self.surfer_vy[env_id, i] *= -0.3
                        self.surfer_vx[env_id, j] *= -0.3
                        self.surfer_vy[env_id, j] *= -0.3
                        self.surfer_collisions[env_id, i] += 1
                        self.surfer_collisions[env_id, j] += 1
                        collision_penalty[env_id, i] -= 6.0
                        collision_penalty[env_id, j] -= 6.0

                        for sid in (i, j):
                            if int(self.surfer_state[env_id, sid].item()) == SurferStatus.RIDING:
                                w_id = int(self.surfer_riding_wave_id[env_id, sid].item())
                                if (
                                    w_id >= 0
                                    and int(self.wave_rider_id[env_id, w_id].item()) == sid
                                ):
                                    self.wave_rider_id[env_id, w_id] = -1
                                self.surfer_state[env_id, sid] = SurferStatus.WIPED_OUT
                                self.surfer_wipeout_timer[env_id, sid] = 1.0
                                self.surfer_riding_wave_id[env_id, sid] = -1

        # 7. Potential-based shaping and total reward
        gamma = 0.98
        new_potential = self._compute_interception_potential()
        raw_shaping = gamma * new_potential - self.surfer_last_potential
        self.surfer_last_potential = new_potential
        shaping = torch.where(
            catch_reward > 0.0,
            torch.zeros_like(raw_shaping),
            torch.clamp(raw_shaping * 0.5, -0.06, 0.06),
        )
        shaping = torch.round(shaping * 10000.0) / 10000.0

        rewards = (
            catch_reward
            + quality_bonus
            + priority_bonus
            + ride_reward
            + collision_penalty
            + dropin_penalty
            + stamina_penalty
            + shaping
        )
        rewards = torch.round(rewards * 10000.0) / 10000.0
        dones = self.step_count >= cfg.maxSteps
        return self.observe(), rewards, dones

    def get_episode_metrics(self, env_id: int = 0) -> EpisodeMetrics:
        waves = max(1, int(self.total_waves_spawned[env_id].item()))
        catches = [int(x) for x in self.surfer_catches[env_id].tolist()]
        return EpisodeMetrics(
            seed=int(self.seeds[env_id].item()),
            totalSteps=int(self.step_count[env_id].item()),
            durationSec=round(float(self.time[env_id].item()), 2),
            totalWavesSpawned=int(self.total_waves_spawned[env_id].item()),
            surferCatches=catches,
            surferCatchRates=[round(c / waves, 4) for c in catches],
            surferScores=[round(float(x), 2) for x in self.surfer_score[env_id].tolist()],
            surferCollisions=[int(x) for x in self.surfer_collisions[env_id].tolist()],
            surferDropIns=[int(x) for x in self.surfer_drop_ins[env_id].tolist()],
            surferStaminaUsed=[
                round(float(x), 4) for x in self.surfer_stamina_used[env_id].tolist()
            ],
            surferRideTime=[round(float(x), 4) for x in self.surfer_ride_time[env_id].tolist()],
        )
