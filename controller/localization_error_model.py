from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np


LookupTable = Sequence[Tuple[float, float]]


@dataclass(frozen=True)
class RangeMeasurementResult:
    true_ranges: np.ndarray
    measured_ranges: np.ndarray
    bias_values: np.ndarray
    sigma_values: np.ndarray
    drift_values: np.ndarray
    burst_values: np.ndarray
    random_noise_values: np.ndarray


@dataclass
class _TemporalRangeState:
    drift_values: np.ndarray
    burst_values: np.ndarray
    burst_remaining_s: np.ndarray
    last_timestamp: float | None


class LocalizationErrorModel:
    sigma_r_const: float = 0.04

    mu_bias_table: LookupTable = (
        (1.0, 0.40),
        (3.0, 0.44),
        (5.0, 0.50),
        (8.0, 0.60),
        (10.0, 0.67),
        (12.0, 0.70),
        (15.0, 0.72),
        (18.0, 0.76),
        (20.0, 0.80),
    )

    def __init__(self):
        self._temporal_states: dict[str, _TemporalRangeState] = {}
        self._drift_tau_s = 3.0
        self._drift_sigma_scale = 0.75
        self._drift_clip_sigma_scale = 2.5
        self._burst_trigger_rate_hz = 0.08
        self._burst_duration_range_s = (0.6, 2.0)
        self._burst_bias_sigma_scale = 3.5
        self._burst_noise_sigma_scale = 2.5

    def _get_temporal_state(self, entity_key: str, num_anchors: int) -> _TemporalRangeState:
        state = self._temporal_states.get(entity_key)
        if state is None or state.drift_values.shape[0] != num_anchors:
            state = _TemporalRangeState(
                drift_values=np.zeros(num_anchors, dtype=float),
                burst_values=np.zeros(num_anchors, dtype=float),
                burst_remaining_s=np.zeros(num_anchors, dtype=float),
                last_timestamp=None,
            )
            self._temporal_states[entity_key] = state
        return state

    def _sample_dt(self, state: _TemporalRangeState, timestamp: float | None) -> float:
        if timestamp is None:
            return 0.1
        if state.last_timestamp is None:
            state.last_timestamp = float(timestamp)
            return 0.1
        dt = max(1e-3, min(1.0, float(timestamp) - float(state.last_timestamp)))
        state.last_timestamp = float(timestamp)
        return dt

    def _update_drift_values(
        self,
        state: _TemporalRangeState,
        sigma_values: np.ndarray,
        dt: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        phi = float(np.exp(-dt / max(self._drift_tau_s, 1e-3)))
        stationary_std = np.maximum(sigma_values * self._drift_sigma_scale, 1e-3)
        innovation_std = stationary_std * np.sqrt(max(0.0, 1.0 - phi * phi))
        state.drift_values = phi * state.drift_values + rng.normal(loc=0.0, scale=innovation_std)
        drift_clip = np.maximum(sigma_values * self._drift_clip_sigma_scale, 1e-3)
        state.drift_values = np.clip(state.drift_values, -drift_clip, drift_clip)
        return state.drift_values.copy()

    def _update_burst_values(
        self,
        state: _TemporalRangeState,
        sigma_values: np.ndarray,
        dt: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        active_mask = state.burst_remaining_s > 0.0
        if np.any(active_mask):
            state.burst_remaining_s[active_mask] = np.maximum(0.0, state.burst_remaining_s[active_mask] - dt)
            state.burst_values[state.burst_remaining_s <= 0.0] = 0.0

        trigger_probability = np.clip(self._burst_trigger_rate_hz * dt, 0.0, 1.0)
        inactive_indices = np.flatnonzero(state.burst_remaining_s <= 0.0)
        for idx in inactive_indices:
            if float(rng.random()) >= trigger_probability:
                continue
            duration = float(rng.uniform(*self._burst_duration_range_s))
            amplitude = abs(float(rng.normal(loc=0.0, scale=sigma_values[idx] * self._burst_bias_sigma_scale)))
            sign = -1.0 if float(rng.random()) < 0.5 else 1.0
            state.burst_remaining_s[idx] = duration
            state.burst_values[idx] = sign * amplitude
        return state.burst_values.copy()

    @staticmethod
    def _interp_table(distance: float, table: LookupTable) -> float:
        xs = np.asarray([row[0] for row in table], dtype=float)
        ys = np.asarray([row[1] for row in table], dtype=float)
        d = float(np.clip(distance, xs[0], xs[-1]))
        return float(np.interp(d, xs, ys))

    def sigma_r(self, distance: float) -> float:
        _ = distance
        return float(self.sigma_r_const)

    def mu_bias(self, distance: float) -> float:
        return self._interp_table(distance, self.mu_bias_table)

    def perturb_ranges(
        self,
        true_ranges: Iterable[float],
        rng: np.random.Generator,
        *,
        entity_key: str = "default",
        timestamp: float | None = None,
    ) -> RangeMeasurementResult:
        true_ranges = np.asarray(list(true_ranges), dtype=float)
        sigma_values = np.asarray([self.sigma_r(d) for d in true_ranges], dtype=float)
        bias_values = np.asarray([self.mu_bias(d) for d in true_ranges], dtype=float)
        temporal_state = self._get_temporal_state(entity_key, true_ranges.shape[0])
        dt = self._sample_dt(temporal_state, timestamp)
        drift_values = self._update_drift_values(temporal_state, sigma_values, dt, rng)
        burst_values = self._update_burst_values(temporal_state, sigma_values, dt, rng)

        burst_noise_scales = sigma_values * self._burst_noise_sigma_scale * (temporal_state.burst_remaining_s > 0.0)
        gaussian_noise = rng.normal(loc=0.0, scale=sigma_values)
        burst_noise = rng.normal(loc=0.0, scale=burst_noise_scales)
        random_noise_values = gaussian_noise + burst_noise
        measured_ranges = true_ranges + bias_values + drift_values + burst_values + random_noise_values
        return RangeMeasurementResult(
            true_ranges=true_ranges,
            measured_ranges=measured_ranges,
            bias_values=bias_values,
            sigma_values=sigma_values,
            drift_values=drift_values,
            burst_values=burst_values,
            random_noise_values=random_noise_values,
        )
