"""Extract_MPtrj.py — stream the bulk MPtrj trajectory JSON into build records.

MPtrj ships as one giant JSON object, ``{mp_id: {frame_id: {structure, labels}}}``,
~12 GB / ~1.5M frames. Loading it whole would need tens of GB of RAM, so we stream
it material-by-material with ijson and yield one lightweight record per frame.

Each frame carries DFT labels; for the energy-only warm-up we keep the two
energy targets and map MPtrj's names onto the project's KNOWN_TARGET_COLUMNS:

    ef_per_atom      -> formation_energy_per_atom   (per-frame formation energy)
    energy_per_atom  -> energy_per_atom             (corrected total energy / atom)

Forces / stress / magmom are present in the source but ignored here (the static
graph + scalar MPNN can't consume them yet — that's the forces project).

``use_float=True`` makes ijson emit plain floats instead of Decimal, so the
structure dict feeds straight into ``pymatgen Structure.from_dict`` and the label
values are JSON/pickle-friendly without a per-frame conversion pass.
"""
import ijson

DEFAULT_MPTRJ_JSON = "database/datafiles/MPtrj/MPtrj_2022.9_full.json"


def iter_mptrj_frames(json_path=DEFAULT_MPTRJ_JSON):
    """Yield one record per MPtrj frame, streaming (bounded memory).

    record = {
        "id": <frame_id>,                       # unique per frame; used as graph filename stem
        "structure": <pymatgen Structure as_dict>,
        "formation_energy_per_atom": float | None,
        "energy_per_atom": float | None,
        "mp_id": <parent material id>,
    }

    A frame missing both energy labels or its structure is skipped (can't train
    on it / can't build a graph). Frames with a usable structure but only one
    energy label are kept — the missing target is left out of the record and the
    index simply carries NaN there.
    """
    with open(json_path, "rb") as f:
        # kvitems at the root yields (mp_id, {frame_id: frame}) one material at a
        # time; each material's frames subtree is small, so RAM stays bounded.
        for mp_id, frames in ijson.kvitems(f, "", use_float=True):
            if not isinstance(frames, dict):
                continue
            for frame_id, frame in frames.items():
                structure = frame.get("structure")
                if not structure:
                    continue
                ef = frame.get("ef_per_atom")
                epa = frame.get("energy_per_atom")
                if ef is None and epa is None:
                    continue
                rec = {"id": frame_id, "structure": structure, "mp_id": mp_id}
                if ef is not None:
                    rec["formation_energy_per_atom"] = float(ef)
                if epa is not None:
                    rec["energy_per_atom"] = float(epa)
                yield rec
