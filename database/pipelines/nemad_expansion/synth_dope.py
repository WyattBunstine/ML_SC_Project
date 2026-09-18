"""Self-contained wrapper around 3DSC's synthetic-doping internals: takes ONE
pymatgen parent Structure + a target doped formula, returns the doped
(disordered) Structure or (None, reason). Mirrors synthetic_doping() lines 361-461."""

import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _HERE)                                     # sibling modules (e.g. synth_dope)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_HERE)))) # repo root (database.*, models.*)
# 3DSC repo (for synthetic_doping internals); override with THREEDSC_REPO env var.
THREEDSC_REPO = _os.environ.get("THREEDSC_REPO", _os.path.expanduser("~/Downloads/old_files/3DSC-main"))

import sys, warnings, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0,THREEDSC_REPO)
from copy import deepcopy
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from superconductors_3D.dataset_preparation import _4_synthetic_doping as SD
from superconductors_3D.dataset_preparation.utils.check_dataset import (
    get_chem_dict, standardise_chem_formula, normalise_chemdict, chemdict_to_formula, find_doping_pairs)
from superconductors_3D.machine_learning.own_libraries.own_functions import only_unique_elements

def synth_dope_one(structure, target_formula, symprec=0.1):
    structure = deepcopy(structure); structure.remove_oxidation_states()
    fstruct = structure.formula
    norm_struct = standardise_chem_formula(fstruct, normalise=True)
    norm_sc     = standardise_chem_formula(target_formula, normalise=True)
    cd_sc = get_chem_dict(target_formula); cd_struct = get_chem_dict(fstruct)
    els_sc, els_struct = list(cd_sc), list(cd_struct)
    if not all(e in els_sc for e in els_struct):
        return None, "struct element not in target"
    if norm_sc == norm_struct:
        return structure, "identical"          # parent already matches -> use as-is
    big, small, *_ = find_doping_pairs(cd_sc, verbose=False)
    sa = SpacegroupAnalyzer(structure, symprec=symprec)
    try: symm_struct = sa.get_symmetrized_structure()
    except TypeError: return None, "spg not recognized"
    symm_sites, equiv = SD.get_symm_equiv_sites(symm_struct)
    fi, fri, fsi, free_si, free_map = SD.free_and_fixed_sites(symm_sites, equiv, structure)
    if set(free_map.values()) != set(free_si): return None, "non-unique free mapping"
    if not only_unique_elements(list(free_map)): return None, "free elements not unique site"
    els_all_fixed = [e for e in els_struct if e not in list(free_map)]
    els_all_ordered = SD.elements_all_ordered(structure, els_struct)
    norm, excl = SD.formula_scaling(els_all_fixed, cd_sc, cd_struct, els_all_ordered)
    if excl: return None, "different scaling of fixed sites"
    cd_sc = normalise_chemdict(cd_sc, norm); fsc = chemdict_to_formula(cd_sc)
    st, reason = SD.insert_additional_elements(big, small, els_all_fixed, cd_sc, cd_struct,
                                               free_map, symm_sites, equiv, structure, els_sc, els_struct)
    if pd.notna(reason): return None, f"insert: {reason}"
    cd_struct = get_chem_dict(st.formula)
    st, reason = SD.modify_occupancies(free_map, cd_sc, cd_struct, equiv, st, symm_sites)
    if pd.notna(reason): return None, f"occ: {reason}"
    if standardise_chem_formula(fsc) != standardise_chem_formula(st.formula):
        return None, "formulas still not equal"
    # Relaxed gate: 3DSC rejects if a symprec=0.1 CIF roundtrip changes the structure.
    # Our builder reads occupancies directly, so validate a P1 (symmetry-free) roundtrip
    # preserves the composition instead — far higher yield, same info to the graph.
    from pymatgen.io.cif import CifWriter, CifParser
    import io
    try:
        cifstr = CifWriter(st, symprec=None, write_site_properties=False).__str__()
        st2 = CifParser.from_str(cifstr).parse_structures(primitive=False)[0]
        if abs(st2.composition.num_atoms - st.composition.num_atoms) > 0.05*st.composition.num_atoms:
            return None, "P1 roundtrip composition drift"
    except Exception as e:
        return None, f"cif write/read: {type(e).__name__}"
    return st, "doped"

if __name__ == "__main__":
    import os
    key = os.environ.get("MP_API_KEY") or sys.exit("error: set MP_API_KEY (env-only)")
    from mp_api.client import MPRester
    m=pd.read_csv("database/datafiles/NE_SCDB/nemad_matches.csv")
    # diverse test cases across families/tiers
    tests=[]
    for fam_query in ["La1.9Sr0.1CuO4","YBa2Cu3O6.5","Bi2Sr2CaCu2O8.2","LaFeAsO0.9F0.1","Nd1.85Ce0.15CuO4","Ba0.6K0.4BiO3"]:
        r=m[m.formula==fam_query]
        if len(r): tests.append(r.iloc[0])
    # fallback: some tier2/3 cuprates
    if len(tests)<6:
        for r in m[(m.tier>=2)&(m.formula.str.contains("Cu"))].head(8).itertuples():
            tests.append(m.loc[r.Index])
    mids=list({t.material_id for t in tests})
    with MPRester(key) as mpr:
        structs={mid:mpr.get_structure_by_material_id(mid) for mid in mids}
    print(f"{'target':26s}{'parent':12s}{'tier':>4}  result")
    for t in tests[:8]:
        st,reason=synth_dope_one(structs[t.material_id], t.formula)
        if st is not None:
            comp=st.composition.formula.replace(" ","")
            dis="disordered" if not st.is_ordered else "ordered"
            print(f"  {t.formula[:24]:24s}{t.material_id:12s}{int(t.tier):>4}  OK[{reason},{dis}] -> {comp[:28]}")
        else:
            print(f"  {t.formula[:24]:24s}{t.material_id:12s}{int(t.tier):>4}  FAIL: {reason}")
