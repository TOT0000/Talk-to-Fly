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

    def sample_delay_seconds(self, rng: np.random.Generator) -> float:
        delay_ms = float(rng.normal(self.profile.delay_mean_ms, self.profile.delay_std_ms))
        delay_ms = float(np.clip(delay_ms, self.profile.delay_min_ms, self.profile.delay_max_ms))
        return delay_ms / 1000.0

    def enqueue(self, packet: object, generation_timestamp: float, sequence_number: int, rng: np.random.Generator) -> Optional[float]:
        if self.profile.packet_loss_prob > 0.0 and float(rng.uniform()) < self.profile.packet_loss_prob:
            return None
        arrival_timestamp = float(generation_timestamp + self.sample_delay_seconds(rng))
        heapq.heappush(self._queue, (arrival_timestamp, int(sequence_number), packet))
        return arrival_timestamp

    def pop_ready(self, now: float) -> List[Tuple[float, int, object]]:
        ready: List[Tuple[float, int, object]] = []
        while self._queue and self._queue[0][0] <= now:
            ready.append(heapq.heappop(self._queue))
        return ready

    def queue_size(self) -> int:
        return len(self._queue)
