# core/constants.py

# Tolerances
TOLERANCE_QUATERNION = 2
POLYGON_REORDER_TOL = 1e-12
CONVEX_DECOMP_TOL = 1e-12
MIN_OVERLAP_AREA = 1e-12

# Defaults
RDF_DEFAULT_BINS = 200
RDF_DEFAULT_R_MAX = 4.0
DEFAULT_NEIGHBOR_R_MAX = 4.0
COMPONENTS = ["C_min", "C_avg", "C_max", "C_eff"]

# Metric registry (loaded at runtime)
AVAILABLE_METRICS = {}