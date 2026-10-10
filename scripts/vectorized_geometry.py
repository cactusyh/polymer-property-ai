from __future__ import annotations
import numpy as np

def dihedrals(points: np.ndarray) -> np.ndarray:
    b0 = -(points[:, 1] - points[:, 0])
    b1 = points[:, 2] - points[:, 1]
    b2 = points[:, 3] - points[:, 2]
    b1 /= np.linalg.norm(b1, axis=1)[:, None]
    v = b0 - np.einsum('ij,ij->i', b0, b1)[:, None] * b1
    w = b2 - np.einsum('ij,ij->i', b2, b1)[:, None] * b1
    return np.degrees(np.arctan2(np.einsum('ij,ij->i', np.cross(b1, v), w), np.einsum('ij,ij->i', v, w)))

def torsion_theta(xyz: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not len(indices):
        return np.empty(0), np.empty(0)
    phi = dihedrals(xyz[indices])
    theta = np.minimum(np.abs(phi) % 360.0, np.abs((phi - 180.0) % 360.0))
    return phi, theta
