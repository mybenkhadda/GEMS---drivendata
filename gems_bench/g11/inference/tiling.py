"""Tile layout over the master grid: every pixel covered, last tile flush with the edge, fixed overlap."""
from __future__ import annotations


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if tile >= length:
        return [0]
    step = max(tile - overlap, 1)
    s = list(range(0, length - tile + 1, step))
    if s[-1] != length - tile:
        s.append(length - tile)
    return s


def tiles(shape, tile: int, overlap: int, mask=None) -> list[tuple[int, int, int, int]]:
    """(r0, r1, c0, c1) windows; with ``mask`` only windows containing a True pixel are returned."""
    h, w = shape
    th, tw = min(tile, h), min(tile, w)
    out = []
    for r in tile_starts(h, th, overlap):
        for c in tile_starts(w, tw, overlap):
            if mask is None or mask[r:r + th, c:c + tw].any():
                out.append((r, r + th, c, c + tw))
    return out
