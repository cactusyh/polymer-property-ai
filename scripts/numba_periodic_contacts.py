from __future__ import annotations
import numpy as np
from numba import njit

@njit(cache=True)
def _close_mask(xyz, lower, lengths, selected, cutoff):
    count = len(selected)
    mask = np.zeros((count, count), dtype=np.bool_)
    if count < 2:
        return mask
    cells_per_axis = np.empty(3, dtype=np.int64)
    for axis in range(3):
        cells_per_axis[axis] = max(int(lengths[axis] // cutoff), 1)
    total_cells = cells_per_axis[0] * cells_per_axis[1] * cells_per_axis[2]
    heads = np.full(total_cells, -1, dtype=np.int64)
    next_index = np.full(count, -1, dtype=np.int64)
    cell_x = np.empty(count, dtype=np.int64)
    cell_y = np.empty(count, dtype=np.int64)
    cell_z = np.empty(count, dtype=np.int64)
    for position in range(count):
        atom = selected[position]
        x = int(((xyz[atom, 0] - lower[0]) % lengths[0]) / cutoff) % cells_per_axis[0]
        y = int(((xyz[atom, 1] - lower[1]) % lengths[1]) / cutoff) % cells_per_axis[1]
        z = int(((xyz[atom, 2] - lower[2]) % lengths[2]) / cutoff) % cells_per_axis[2]
        cell_x[position], cell_y[position], cell_z[position] = x, y, z
        cell = (x * cells_per_axis[1] + y) * cells_per_axis[2] + z
        next_index[position] = heads[cell]
        heads[cell] = position
    cutoff_squared = cutoff * cutoff
    for left in range(count):
        seen_cells = np.full(27, -1, dtype=np.int64)
        seen_count = 0
        for dx in range(-1, 2):
            x = (cell_x[left] + dx) % cells_per_axis[0]
            for dy in range(-1, 2):
                y = (cell_y[left] + dy) % cells_per_axis[1]
                for dz in range(-1, 2):
                    z = (cell_z[left] + dz) % cells_per_axis[2]
                    cell = (x * cells_per_axis[1] + y) * cells_per_axis[2] + z
                    duplicate = False
                    for index in range(seen_count):
                        if seen_cells[index] == cell:
                            duplicate = True
                            break
                    if duplicate:
                        continue
                    seen_cells[seen_count] = cell
                    seen_count += 1
                    right = heads[cell]
                    while right >= 0:
                        if right > left:
                            delta0 = xyz[selected[right], 0] - xyz[selected[left], 0]
                            delta1 = xyz[selected[right], 1] - xyz[selected[left], 1]
                            delta2 = xyz[selected[right], 2] - xyz[selected[left], 2]
                            delta0 -= lengths[0] * np.round(delta0 / lengths[0])
                            delta1 -= lengths[1] * np.round(delta1 / lengths[1])
                            delta2 -= lengths[2] * np.round(delta2 / lengths[2])
                            if delta0 * delta0 + delta1 * delta1 + delta2 * delta2 < cutoff_squared:
                                mask[left, right] = True
                        right = next_index[right]
    return mask

def contact_indices(xyz, box, ids, bonds, cutoff):
    selected = np.asarray(sorted(ids), dtype=np.int64) - 1
    mask = _close_mask(xyz, box[:, 0], box[:, 1] - box[:, 0], selected, cutoff)
    positions = np.argwhere(mask)
    return {
        (int(selected[left]) + 1, int(selected[right]) + 1)
        for left, right in positions
        if abs(int(selected[left]) - int(selected[right])) > 3
        and (int(selected[left]) + 1, int(selected[right]) + 1) not in bonds
        and (int(selected[right]) + 1, int(selected[left]) + 1) not in bonds
    }
