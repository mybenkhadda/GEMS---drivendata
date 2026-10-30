"""GEMS geological feature discovery & fault association (notebooks/colab/GEMS_GEOLOGICAL_FEATURE_DISCOVERY.ipynb).

    from gems_eda.feature_discovery import FeatureDiscoveryExperiment
    E = FeatureDiscoveryExperiment(CONFIG); E.load(); E.compute_features(); E.global_association() ...

Reuses the Layer Lab engine (``streamlit/gems_layer_lab/lab``: restricted formula language, transformation library,
association / complementarity metrics) so formulas behave identically in the Streamlit studio, this notebook and the
exporters. Outputs: ``<GEMS_2>/outputs/geological_feature_discovery/``.
"""
FD_VERSION = "gfd.1"
FD_CODE_REV = "gfd.1-r1"


def __getattr__(name):
    if name == "FeatureDiscoveryExperiment":
        from .experiment import FeatureDiscoveryExperiment

        return FeatureDiscoveryExperiment
    raise AttributeError(name)
