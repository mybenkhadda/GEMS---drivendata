"""Exact band resolution (Stage A).

The feature GeoTIFF names its bands ``<identifier> - <description>``; ``gems_eda.io._band_names`` keeps the identifier.
Every lookup here is an EXACT identifier match: a substring / prefix lookup silently returns ``iso_grav_anom_hg`` when
``iso_grav_anom`` was asked for (the G05 bug), so it is refused by construction and a missing or ambiguous name fails
closed with the list of available identifiers.
"""
from __future__ import annotations


class BandResolutionError(KeyError):
    pass


def identifier(name_or_description: str) -> str:
    """``'rtp - reduced to pole'`` -> ``'rtp'``; an identifier is returned unchanged (stripped)."""
    return str(name_or_description).split(" - ")[0].strip()


def resolve(requested, available) -> dict:
    """{requested identifier: 1-based band index} by EXACT identifier equality.

    Fails closed on: a requested identifier absent from the file, an identifier present more than once, or an empty
    request. Near misses (substring matches) are listed in the error so the caller sees what a fuzzy lookup would
    have picked, but they are never used."""
    ids = [identifier(a) for a in available]
    req = [identifier(r) for r in requested]
    if not req:
        raise BandResolutionError("no band requested")
    dup = sorted({i for i in ids if ids.count(i) > 1})
    out, errs = {}, []
    for r in req:
        hits = [k for k, i in enumerate(ids) if i == r]
        if len(hits) == 1:
            out[r] = hits[0] + 1
            continue
        if r in dup:
            errs.append(f"{r!r} appears {len(hits)} times in the file")
            continue
        near = [i for i in ids if r in i or i in r]
        errs.append(f"{r!r} not found (exact match required; substring near misses NOT used: {near})")
    if errs:
        raise BandResolutionError("; ".join(errs) + f" | available identifiers: {ids}")
    return out


def resolution_table(requested, available) -> list[dict]:
    """One row per requested identifier (status RESOLVED / MISSING / DUPLICATE, index, near misses) - never raises;
    the preflight writes it, then fails closed through ``resolve``."""
    ids = [identifier(a) for a in available]
    rows = []
    for r in (identifier(x) for x in requested):
        hits = [k for k, i in enumerate(ids) if i == r]
        near = [i for i in ids if i != r and (r in i or i in r)]
        st = "RESOLVED" if len(hits) == 1 else ("DUPLICATE" if hits else "MISSING")
        rows.append({"requested": r, "status": st, "band_index": hits[0] + 1 if len(hits) == 1 else None,
                     "substring_near_misses": ";".join(near), "rule": "exact identifier equality"})
    return rows
