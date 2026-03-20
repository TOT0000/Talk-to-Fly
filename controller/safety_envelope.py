from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .state_packet import LocalizedStatePacket


CHI2_2_095 = 5.991


@dataclass
class SafetyEnvelope2D:
    entity_type: str
    center_xy: np.ndarray
    major_axis_radius: float
    minor_axis_radius: float
    orientation_rad: float
    orientation_deg: float
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    chi2_val: float
    matrix_xy: np.ndarray
    packet_generation_timestamp: float
    packet_receive_timestamp: Optional[float]
    sequence_number: int

    def directional_radius(self, unit_vec_xy: np.ndarray) -> float:
        direction = np.asarray(unit_vec_xy, dtype=float).reshape(2)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            return 0.0
        u = direction / norm
        value = float(u.T @ self.matrix_xy @ u)
        value = max(value, 0.0)
        return float(np.sqrt(self.chi2_val * value))


def build_safety_envelope(packet: LocalizedStatePacket, chi2_val: float = CHI2_2_095) -> SafetyEnvelope2D:
    matrix_xy = np.asarray(packet.M_xy, dtype=float)
    center_xy = np.asarray(packet.estimated_position_3d[:2], dtype=float)

    eigenvalues, eigenvectors = np.linalg.eigh(matrix_xy)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]

    major_axis_radius = float(np.sqrt(chi2_val * eigenvalues[0]))
    minor_axis_radius = float(np.sqrt(chi2_val * eigenvalues[1]))
    orientation_rad = float(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    orientation_deg = float(np.degrees(orientation_rad))

    return SafetyEnvelope2D(
        entity_type=packet.entity_type,
        center_xy=center_xy,
        major_axis_radius=major_axis_radius,
        minor_axis_radius=minor_axis_radius,
        orientation_rad=orientation_rad,
        orientation_deg=orientation_deg,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        chi2_val=float(chi2_val),
        matrix_xy=matrix_xy,
        packet_generation_timestamp=float(packet.state_generation_timestamp),
        packet_receive_timestamp=packet.received_packet_timestamp,
        sequence_number=int(packet.sequence_number),
    )
