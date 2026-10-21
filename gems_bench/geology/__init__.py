"""Geological geometry utilities: axial strike algebra, label-free orientation fields, trace topology."""
from .orientation import (along_normal_offsets, axial_angle, axial_difference, axial_vector, mean_axial,
                          orientation_entropy, strike_field)
from .skeleton import (branch_points, component_length_km, endpoints, orientation_targets, skeletonize,
                       trace_strata)

__all__ = ["along_normal_offsets", "axial_angle", "axial_difference", "axial_vector", "mean_axial",
           "orientation_entropy", "strike_field", "branch_points", "component_length_km", "endpoints",
           "orientation_targets", "skeletonize", "trace_strata"]
