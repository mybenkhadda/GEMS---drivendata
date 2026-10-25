"""Optional postprocessing research layer (M6.6): topology-aware operations on binary fault masks."""
from .topology import DEFAULT_GRID, OPERATIONS, ablation, apply_ops, apply_per_zone, topology_metrics

__all__ = ["DEFAULT_GRID", "OPERATIONS", "ablation", "apply_ops", "apply_per_zone", "topology_metrics"]
