from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class DelayProfile:
    name: str
    delay_mean_ms: float
    delay_std_ms: float
    delay_min_ms: float
    delay_max_ms: float
    packet_loss_prob: float


class UplinkDelayModel:
    DEFAULT_PROFILE = DelayProfile(
        name="lte_nominal_gcs_uplink",
        delay_mean_ms=97.0,
        delay_std_ms=17.0,
        delay_min_ms=69.0,
        delay_max_ms=130.0,
        packet_loss_prob=0.0,
    )

    def __init__(self, profile: Optional[DelayProfile] = None):
        self.profile = profile or self.DEFAULT_PROFILE
        self._queue: List[Tuple[float, int, object]] = []
        self._last_timestamp: Optional[float] = None
        self._delay_drift_ms: float = 0.0
        self._spike_remaining_s: float = 0.0
        self._spike_extra_ms: float = 0.0
        self._drop_burst_remaining_packets: int = 0
        self._drift_tau_s = 4.5
        self._drift_std_ms = max(1.0, self.profile.delay_std_ms * 0.55)
        self._small_jitter_std_ms = max(1.0, self.profile.delay_std_ms * 0.45)
        self._spike_trigger_rate_hz = 0.06
        self._spike_duration_range_s = (0.4, 1.8)
        self._spike_extra_ms_range = (
            max(self.profile.delay_std_ms * 1.5, 12.0),
            max(self.profile.delay_std_ms * 4.0, 45.0),
        )
        self._drop_burst_trigger_rate_hz = 0.01
        self._drop_burst_packets_range = (1, 2)

    def _sample_dt(self, generation_timestamp: Optional[float]) -> float:
        if generation_timestamp is None:
            return 0.1
        if self._last_timestamp is None:
            self._last_timestamp = float(generation_timestamp)
            return 0.1
        dt = max(1e-3, min(1.0, float(generation_timestamp) - float(self._last_timestamp)))
        self._last_timestamp = float(generation_timestamp)
        return dt

    def _update_delay_drift_ms(self, dt: float, rng: np.random.Generator) -> float:
        phi = float(np.exp(-dt / max(self._drift_tau_s, 1e-3)))
        innovation_std = self._drift_std_ms * np.sqrt(max(0.0, 1.0 - phi * phi))
        self._delay_drift_ms = phi * self._delay_drift_ms + float(rng.normal(loc=0.0, scale=innovation_std))
        drift_clip = max(self.profile.delay_std_ms * 2.0, 10.0)
        self._delay_drift_ms = float(np.clip(self._delay_drift_ms, -drift_clip, drift_clip))
        return self._delay_drift_ms

    def _update_spike_ms(self, dt: float, rng: np.random.Generator) -> float:
        if self._spike_remaining_s > 0.0:
            self._spike_remaining_s = max(0.0, self._spike_remaining_s - dt)
            if self._spike_remaining_s <= 0.0:
                self._spike_extra_ms = 0.0

        trigger_probability = float(np.clip(self._spike_trigger_rate_hz * dt, 0.0, 1.0))
        if self._spike_remaining_s <= 0.0 and float(rng.uniform()) < trigger_probability:
            self._spike_remaining_s = float(rng.uniform(*self._spike_duration_range_s))
            self._spike_extra_ms = float(rng.uniform(*self._spike_extra_ms_range))
        return self._spike_extra_ms

    def _should_drop_packet(self, dt: float, rng: np.random.Generator) -> bool:
        if self._drop_burst_remaining_packets > 0:
            self._drop_burst_remaining_packets -= 1
            return True
        if self.profile.packet_loss_prob > 0.0 and float(rng.uniform()) < self.profile.packet_loss_prob:
            return True
        trigger_probability = float(np.clip(self._drop_burst_trigger_rate_hz * dt, 0.0, 1.0))
        if float(rng.uniform()) < trigger_probability:
            self._drop_burst_remaining_packets = int(rng.integers(self._drop_burst_packets_range[0], self._drop_burst_packets_range[1] + 1))
            self._drop_burst_remaining_packets = max(0, self._drop_burst_remaining_packets - 1)
            return True
        return False

    def sample_delay_seconds(
        self,
        rng: np.random.Generator,
        generation_timestamp: Optional[float] = None,
        *,
        dt: Optional[float] = None,
    ) -> float:
        dt = self._sample_dt(generation_timestamp) if dt is None else float(dt)
        baseline_delay_ms = float(self.profile.delay_mean_ms)
        small_jitter_ms = float(rng.normal(loc=0.0, scale=self._small_jitter_std_ms))
        slow_delay_drift_ms = self._update_delay_drift_ms(dt, rng)
        occasional_spike_ms = self._update_spike_ms(dt, rng)
        delay_ms = baseline_delay_ms + small_jitter_ms + slow_delay_drift_ms + occasional_spike_ms
        delay_cap_ms = max(self.profile.delay_max_ms + self._spike_extra_ms_range[1], self.profile.delay_max_ms * 1.8)
        delay_ms = float(np.clip(delay_ms, self.profile.delay_min_ms, delay_cap_ms))
        return delay_ms / 1000.0

    def enqueue(self, packet: object, generation_timestamp: float, sequence_number: int, rng: np.random.Generator) -> Optional[float]:
        dt = self._sample_dt(generation_timestamp)
        if self._should_drop_packet(dt, rng):
            return None
        arrival_timestamp = float(generation_timestamp + self.sample_delay_seconds(rng, generation_timestamp=generation_timestamp, dt=dt))
        heapq.heappush(self._queue, (arrival_timestamp, int(sequence_number), packet))
        return arrival_timestamp

    def pop_ready(self, now: float) -> List[Tuple[float, int, object]]:
        ready: List[Tuple[float, int, object]] = []
        while self._queue and self._queue[0][0] <= now:
            ready.append(heapq.heappop(self._queue))
        return ready

    def queue_size(self) -> int:
        return len(self._queue)
