"""Bit-exact Mulberry32 PRNG matching `src/sim/prng.ts`."""

from __future__ import annotations

import torch

UINT32_MASK = 0xFFFFFFFF
MULBERRY_INC = 0x6D2B79F5


def normalize_seed(seed: int) -> int:
    s = (int(seed) & UINT32_MASK) or MULBERRY_INC
    return s & UINT32_MASK


def next_random(rng_state: int) -> tuple[float, int]:
    """Advance scalar 32-bit state and return `(float01, next_state)`."""
    next_state = (rng_state + MULBERRY_INC) & UINT32_MASK
    t = next_state
    t = ((t ^ (t >> 15)) * (t | 1)) & UINT32_MASK
    prod = ((t ^ (t >> 7)) * (t | 61)) & UINT32_MASK
    t = (t ^ ((t + prod) & UINT32_MASK)) & UINT32_MASK
    uint32 = (t ^ (t >> 14)) & UINT32_MASK
    return uint32 / 4294967296.0, next_state


def next_random_batch(rng_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized bit-exact Mulberry32 on `int64` tensor of uint32 states `[N]`."""
    next_states = (rng_states + MULBERRY_INC) & UINT32_MASK
    t = next_states
    t = ((t ^ (t >> 15)) * (t | 1)) & UINT32_MASK
    prod = ((t ^ (t >> 7)) * (t | 61)) & UINT32_MASK
    t = (t ^ ((t + prod) & UINT32_MASK)) & UINT32_MASK
    uint32 = (t ^ (t >> 14)) & UINT32_MASK
    floats = uint32.to(torch.float64) / 4294967296.0
    return floats, next_states
