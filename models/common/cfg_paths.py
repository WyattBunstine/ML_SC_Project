"""Config path expansion: keep site-specific storage roots out of the repo.

Configs reference bulk data through ``${ML_SC_DATA}`` (e.g.
``"index_path": "${ML_SC_DATA}/MPtrj/packed_v45"``). Every config loader passes
the parsed JSON through :func:`expand_config_paths`, which expands environment
variables inside every string. ``ML_SC_DATA`` defaults to ``database/datafiles/``
under the repository root, so a local run needs nothing set; a cluster job
exports it to wherever the packs are mirrored.
"""
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DATA_ROOT = os.path.join(REPO_ROOT, "database", "datafiles")


def data_root():
    """The ``${ML_SC_DATA}`` root in effect (env var, else database/datafiles)."""
    return os.environ.get("ML_SC_DATA") or DEFAULT_DATA_ROOT


def expand_path(value):
    """Expand ``${VAR}`` / ``$VAR`` / ``~`` in one string (non-strings pass through)."""
    if not isinstance(value, str) or not ("$" in value or value.startswith("~")):
        return value
    os.environ.setdefault("ML_SC_DATA", DEFAULT_DATA_ROOT)
    return os.path.expanduser(os.path.expandvars(value))


def expand_config_paths(obj):
    """Recursively expand environment variables in every string of a parsed config."""
    if isinstance(obj, dict):
        return {k: expand_config_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_config_paths(v) for v in obj]
    return expand_path(obj)
