import argparse
import math
import os

# Use the official Materials Project client. NOTE: do NOT import MPRester from
# pymatgen.ext.matproj -- that is a legacy web-API shim whose materials.summary.search
# follows a different (incompatible) kwarg spec and silently mishandles this query.
try:
    from mp_api.client import MPRester
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "mp-api is required to download non-SC structures. Install it with "
        "'pip install mp-api'."
    ) from exc
from pymatgen.io.cif import CifWriter


def _get_api_key():
    """Read the Materials Project API key from the MP_API_KEY environment variable."""
    key = os.environ.get("MP_API_KEY")
    if not key:
        raise RuntimeError(
            "Materials Project API key not found. Set the MP_API_KEY environment "
            "variable (e.g. $env:MP_API_KEY = '...' on Windows, or "
            "export MP_API_KEY=... on Linux/macOS)."
        )
    return key


def gen_dataset(min_band_gap=1.0, prop_file=None, cif_loc=None, limit=None, chunk_size=1000):
    """Download large-band-gap (non-superconducting) materials from the Materials
    Project to use as negative examples for the SC/non-SC classifier.

    Writes one CIF per material into ``cif_loc`` and a headerless
    ``<material_id>.cif,0.0`` row per material into ``prop_file``. The 0.0 here is
    only a placeholder T_c -- these rows are marked non-SC at database-build time
    via the explicit ``label`` column (``main.py build-db --nonsc-source ...``),
    NOT via this value.

    Parameters
    ----------
    min_band_gap : float    minimum band gap in eV (default 1.0)
    prop_file : str or None output id->property CSV
                            (default: database/Non_SC_DB_MP/Non_SC.csv)
    cif_loc : str or None   output CIF directory
                            (default: database/Non_SC_DB_MP/cifs/)
    limit : int or None     cap the number of materials written (for class balance
                            against the ~5.8k superconductors)
    chunk_size : int        MP API page size
    """
    here = os.path.dirname(os.path.abspath(__file__))
    if prop_file is None:
        prop_file = os.path.join(here, "Non_SC.csv")
    if cif_loc is None:
        cif_loc = os.path.join(here, "cifs")
    os.makedirs(cif_loc, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(prop_file)), exist_ok=True)

    # When a limit is set, only pull as many API pages as needed.
    num_chunks = math.ceil(limit / chunk_size) if limit else None

    with MPRester(_get_api_key()) as mpr:
        data = mpr.materials.summary.search(
            band_gap=(min_band_gap, 1000),
            all_fields=False,
            fields=["material_id", "structure"],
            num_chunks=num_chunks,
            chunk_size=chunk_size,
        )
        if limit:
            data = data[:limit]

        written = 0
        with open(prop_file, "w") as file:
            for mat in data:
                mid = str(mat.material_id)
                CifWriter(mat.structure).write_file(os.path.join(cif_loc, mid + ".cif"))
                file.write(mid + ".cif,0.0\n")
                written += 1

    print(f"Wrote {written} non-SC materials -> {prop_file} (cifs in {cif_loc}/)")
    return prop_file, cif_loc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download non-superconductor (large band gap) materials from "
                    "the Materials Project as negative examples."
    )
    parser.add_argument("--min-band-gap", type=float, default=1.0,
                        help="minimum band gap in eV (default: 1.0)")
    parser.add_argument("--prop-file", default=None,
                        help="output id->property CSV (default: Non_SC.csv beside this script)")
    parser.add_argument("--cif-loc", default=None,
                        help="output CIF directory (default: cifs/ beside this script)")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the number of materials written (for class balance)")
    parser.add_argument("--chunk-size", type=int, default=1000,
                        help="MP API page size (default: 1000)")
    args = parser.parse_args()
    gen_dataset(args.min_band_gap, args.prop_file, args.cif_loc, args.limit, args.chunk_size)
