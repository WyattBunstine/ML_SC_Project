import argparse
import csv
import math
import os

# Use the official Materials Project client. NOTE: do NOT import MPRester from
# pymatgen.ext.matproj -- that is a legacy web-API shim whose materials.summary.search
# follows a different (incompatible) kwarg spec and silently mishandles this query.
try:
    from mp_api.client import MPRester
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "mp-api is required to download structures. Install it with "
        "'pip install mp-api'."
    ) from exc
from pymatgen.io.cif import CifWriter


def _get_api_key():
    """Read the Materials Project API key from the MP_API_KEY environment variable."""
    key = os.environ.get("MP_API_KEY")
    if not key:
        raise RuntimeError(
            "Materials Project API key not found. Set the MP_API_KEY environment "
            "variable (e.g. export MP_API_KEY=... on Linux/macOS, or "
            "$env:MP_API_KEY = '...' on Windows)."
        )
    return key


def gen_dataset(prop_file=None, cif_loc=None, limit=None, chunk_size=1000,
                theoretical=False):
    """Download experimentally-observed materials from the Materials Project and
    record their thermodynamic energies as regression targets.

    Writes one CIF per material into ``cif_loc`` and one row per material into
    ``prop_file`` with header:

        cif,material_id,e_above_hull,formation_energy_per_atom

    where ``cif`` is the CIF filename (``<material_id>.cif``), ``e_above_hull`` is
    the energy above the convex hull (eV/atom, >=0; 0 == on-hull/stable) and
    ``formation_energy_per_atom`` is the formation enthalpy (eV/atom, typically
    negative). Both targets are emitted as columns; the consumer chooses which to
    train on. The cgv4 graph builder (``database_main.py``) preserves both columns
    in its index, and the MPNN selects one via the config's ``target_column`` key
    -- so no target is baked in here.

    Materials with no structure (cannot be written to CIF), no energy data (both
    targets absent), or that fail to serialize to CIF are skipped and counted.
    The CSV is rewritten in full on every run; a CIF already on disk is not
    re-written (cheap CIF reuse), but the API query and CSV are not incrementally
    resumed -- a crashed run leaves a truncated CSV, and a completed re-run
    overwrites it cleanly.

    Parameters
    ----------
    prop_file : str or None  output id->property CSV
                             (default: database/datafiles/MP_Energy/mp_energy.csv)
    cif_loc : str or None    output CIF directory
                             (default: database/datafiles/MP_Energy/cifs/)
    limit : int or None      cap the number of materials written (for testing)
    chunk_size : int         MP API page size
    theoretical : bool or None  theoretical flag to filter on. False (default)
                             keeps only experimentally-observed (ICSD-backed)
                             materials; None applies no filter (includes all).
    """
    # This script lives in database/; its data lives under database/datafiles/.
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "datafiles", "MP_Energy")
    if prop_file is None:
        prop_file = os.path.join(data_dir, "mp_energy.csv")
    if cif_loc is None:
        cif_loc = os.path.join(data_dir, "cifs")
    os.makedirs(cif_loc, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(prop_file)), exist_ok=True)

    # When a limit is set, only pull as many API pages as needed.
    num_chunks = math.ceil(limit / chunk_size) if limit else None

    with MPRester(_get_api_key()) as mpr:
        data = mpr.materials.summary.search(
            theoretical=theoretical,
            all_fields=False,
            fields=["material_id", "structure",
                    "energy_above_hull", "formation_energy_per_atom"],
            num_chunks=num_chunks,
            chunk_size=chunk_size,
        )
        if limit:
            data = data[:limit]

        written = 0
        skipped_no_structure = 0
        skipped_no_energy = 0
        skipped_cif_error = 0
        with open(prop_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["cif", "material_id", "e_above_hull", "formation_energy_per_atom"])
            for mat in data:
                if mat.structure is None:
                    skipped_no_structure += 1
                    continue
                # Need at least one usable target. Skip rows where both are absent.
                e_hull = mat.energy_above_hull
                e_form = mat.formation_energy_per_atom
                if e_hull is None and e_form is None:
                    skipped_no_energy += 1
                    continue

                mid = str(mat.material_id)
                cif_name = mid + ".cif"
                cif_path = os.path.join(cif_loc, cif_name)
                # A single unserializable structure must not abort the whole pull:
                # skip it (and any stale partial file) and keep going.
                if not os.path.exists(cif_path):
                    try:
                        CifWriter(mat.structure).write_file(cif_path)
                    except Exception as exc:
                        skipped_cif_error += 1
                        print(f"  skipping {mid}: CIF write failed "
                              f"({type(exc).__name__}: {exc})")
                        if os.path.exists(cif_path):
                            os.remove(cif_path)
                        continue
                writer.writerow([
                    cif_name,
                    mid,
                    "" if e_hull is None else e_hull,
                    "" if e_form is None else e_form,
                ])
                written += 1

    print(f"Wrote {written} materials -> {prop_file} (cifs in {cif_loc}/)")
    if skipped_no_structure:
        print(f"  skipped {skipped_no_structure} with no structure")
    if skipped_no_energy:
        print(f"  skipped {skipped_no_energy} with no energy data")
    if skipped_cif_error:
        print(f"  skipped {skipped_cif_error} that failed to serialize to CIF")
    return prop_file, cif_loc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download experimentally-observed Materials Project structures "
                    "with their energy-above-hull and formation-energy targets."
    )
    parser.add_argument("--prop-file", default=None,
                        help="output id->property CSV (default: mp_energy.csv beside this script)")
    parser.add_argument("--cif-loc", default=None,
                        help="output CIF directory (default: cifs/ beside this script)")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the number of materials written (for testing)")
    parser.add_argument("--chunk-size", type=int, default=1000,
                        help="MP API page size (default: 1000)")
    parser.add_argument("--include-theoretical", action="store_true",
                        help="include theoretical (non-experimental) materials too")
    args = parser.parse_args()
    gen_dataset(
        prop_file=args.prop_file,
        cif_loc=args.cif_loc,
        limit=args.limit,
        chunk_size=args.chunk_size,
        theoretical=None if args.include_theoretical else False,
    )
