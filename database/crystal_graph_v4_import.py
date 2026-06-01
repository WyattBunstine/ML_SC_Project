import os
import sys

def _resolve_rp_path():
    env = os.environ.get("RP_TOLERANCE_FACTOR_PATH")
    if env:
        return os.path.abspath(env)
    # Default: sibling directory of this project
    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.normpath(os.path.join(project_root, "..", "RPToleranceFactor"))

_rp_path = _resolve_rp_path()
if _rp_path not in sys.path:
    sys.path.insert(0, _rp_path)

try:
    from crystal_graph_v4 import build_crystal_graph_from_cif
except ImportError as exc:
    raise ImportError(
        f"Could not import crystal_graph_v4 from '{_rp_path}'. "
        "Set the RP_TOLERANCE_FACTOR_PATH environment variable to the RPToleranceFactor "
        "project directory and ensure its dependencies (pymatgen, scipy, numpy) are installed."
    ) from exc
