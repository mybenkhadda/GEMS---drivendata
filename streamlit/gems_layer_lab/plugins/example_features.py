"""Example TRUSTED feature plugins (loaded from this local directory; full Python, not sandboxed).

Call them from the formula editor, e.g.::

    feature_B = custom_disruption(det_elev, det_elev_slope, sigma=4)
    mag_edge_persist = edge_persistence(rtp, sigma_min=1, sigma_max=10)
"""
import numpy as np

from lab.ops import gradmag, local_std, multiscale_persistence, orientation, axial_agreement, safe_divide
from lab.plugins import register_feature


@register_feature(name="custom_disruption", inputs=["elevation", "slope"], parameters={"sigma": 3.0, "eps": 0.01},
                  halo_px=lambda p: int(np.ceil(4 * p["sigma"])) + 4, units="ratio (elevation gradient / slope std)",
                  description="terrain edge strength relative to local slope texture")
def custom_disruption(layers, sigma=3.0, eps=0.01):
    edge = gradmag(layers["elevation"], sigma)
    texture = local_std(layers["slope"], sigma)
    return safe_divide(edge, texture, eps)


@register_feature(name="edge_persistence", inputs=["field"], parameters={"sigma_min": 1.0, "sigma_max": 8.0, "n": 4},
                  halo_px=lambda p: int(np.ceil(4 * p["sigma_max"])) + 3, units="[0, 1]",
                  description="multiscale persistence of a gradient edge")
def edge_persistence(layers, sigma_min=1.0, sigma_max=8.0, n=4):
    return multiscale_persistence(layers["field"], sigma_min, sigma_max, int(n))


@register_feature(name="fabric_agreement", inputs=["a", "b"], parameters={"sigma": 5.0},
                  halo_px=lambda p: int(np.ceil(4 * p["sigma"])) + 8, units="[0, 1] (cos^2 of axial difference)",
                  description="agreement of the structural fabric of two fields")
def fabric_agreement(layers, sigma=5.0):
    return axial_agreement(orientation(layers["a"], sigma), orientation(layers["b"], sigma))
