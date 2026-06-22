"""Entry point for the ML superconductor (CGCNN) project.

A single CLI for the whole workflow; run all commands from the project root:

  * build-db : generate / regenerate the database files consumed by the CNN
  * train    : train + evaluate the CGCNN from a JSON config
  * plot     : scatter-plot test predictions vs. targets

Examples:
  python main.py build-db --kind basic
  python main.py train configs/orig_basic.json
  python main.py plot
"""
import argparse
import os
import subprocess
import sys

import database.database_main as database

# Default (id->property csv, cif directory) pair. id_prop.csv is headerless:
# each row is "<cif_filename>,<tc>" and the filenames live in cifs/.
DEFAULT_SOURCE = ["database/datafiles/MP/id_prop.csv", "database/datafiles/MP/cifs/"]

DEFAULT_OUTPUTS = {
    "atom-init": "database/datafiles/atom_init.json",
    "basic": "database/datafiles/MP/SC_MP_basic",
    "cgv4": "database/datafiles/MP/SC_MP_V4",
}

DEFAULT_CGV4_GRAPH_DIR = "database/datafiles/MP/graphs_v4"


def _action_word(path):
    """'Regenerating' if the target already exists, otherwise 'Generating'."""
    return "Regenerating" if os.path.exists(path) else "Generating"


def cmd_build_db(args):
    if args.kind == "atom-init":
        out = args.output or DEFAULT_OUTPUTS["atom-init"]
        print(f"{_action_word(out)} element feature file -> {out}")
        database.generate_atom_init(output_file=out, max_z=args.max_z)
        print("Done.")
        return

    # basic / cgv4 consume (csv, cif_dir, label) source specs. Superconductor
    # sources (--source, or the default) are labeled 1; non-superconductor sources
    # (--nonsc-source) are labeled 0. The label is stored in the DB's `label`
    # column and is what the SC/non-SC classifier trains on (it is NOT derived from
    # T_c, since ~31% of the SC dataset has T_c=0.0).
    sc_sources = args.source if args.source else [DEFAULT_SOURCE]
    sources = [[csv_path, cif_dir, 1] for csv_path, cif_dir in sc_sources]
    if args.nonsc_source:
        sources += [[csv_path, cif_dir, 0] for csv_path, cif_dir in args.nonsc_source]

    for csv_path, cif_dir, _label in sources:
        if not os.path.exists(csv_path):
            sys.exit(f"error: source csv not found: {csv_path}")
        if not os.path.isdir(cif_dir):
            sys.exit(f"error: cif directory not found: {cif_dir}")

    out = args.output or DEFAULT_OUTPUTS[args.kind]
    print(f"{_action_word(out + '.pickle')} {args.kind} dataset -> {out}.pickle / {out}.csv")
    if args.limit:
        print(f"  (limited to first {args.limit} rows per source)")

    if args.kind == "basic":
        database.generate_Basic_DB(
            sources,
            output_file=out,
            parallel=args.parallel,
            timing=args.timing,
            batch_size=args.batch_size,
            has_header=args.has_header,
            limit=args.limit,
        )
    else:  # cgv4
        graph_dir = args.graph_dir or DEFAULT_CGV4_GRAPH_DIR
        print(f"  Graph files -> {graph_dir}/")
        database.generate_CGv4_DB(
            sources,
            output_dir=graph_dir,
            output_index=out,
            has_header=args.has_header,
            limit=args.limit,
        )
    print("Done.")


def cmd_build_mptrj(args):
    # Build cgv4 graphs for every MPtrj trajectory frame, streaming the bulk JSON
    # so the 12 GB file is never fully loaded. Energy-only warm-up: each frame's
    # ef_per_atom -> formation_energy_per_atom and energy_per_atom are indexed as
    # selectable targets. Needs the RPToleranceFactor graph builder on the path
    # (same as build-db --kind cgv4), so run this locally, then sync the resulting
    # graphs_v4/ + index pickle to the cluster.
    if not os.path.exists(args.input):
        sys.exit(f"error: MPtrj JSON not found: {args.input}")
    from database.Extract_MPtrj import iter_mptrj_frames

    graph_dir = args.graph_dir or "database/datafiles/MPtrj/graphs_v4"
    out = args.output or "database/datafiles/MPtrj/MPtrj_V4"
    print(f"{_action_word(out + '.pickle')} MPtrj cgv4 dataset -> {out}.pickle / {out}.csv")
    print(f"  Graph files -> {graph_dir}/   (source: {args.input})")
    if args.limit:
        print(f"  (limited to first {args.limit} frames)")
    database.generate_CGv4_DB_from_structures(
        iter_mptrj_frames(args.input),
        output_dir=graph_dir,
        output_index=out,
        limit=args.limit,
        n_workers=args.workers,
    )
    print("Done.")


def cmd_augment_positions(args):
    # Backfill frac_coords + lattice onto existing MPtrj graphs from the source
    # structures (NO Voronoi rebuild) so the packed store can carry geometry for the
    # long-range distance bias. Resumable + parallel; run where the graphs live
    # (cluster scratch), then re-pack with pack-dataset.
    if not os.path.exists(args.input):
        sys.exit(f"error: MPtrj JSON not found: {args.input}")
    graph_dir = args.graph_dir or "database/datafiles/MPtrj/graphs_v4"
    if not os.path.isdir(graph_dir):
        sys.exit(f"error: graph dir not found: {graph_dir}")
    from database.Extract_MPtrj import iter_mptrj_frames
    print(f"Augmenting graphs in {graph_dir} with positions from {args.input}")
    database.augment_graphs_with_positions(
        iter_mptrj_frames(args.input), graph_dir=graph_dir,
        n_workers=args.workers, limit=args.limit)
    print("Done. Re-pack (pack-dataset) to carry positions into the packed store.")


def cmd_augment_physics(args):
    # Backfill the multitask TARGETS (forces/magmom/stress + positions) onto existing MPtrj
    # graphs and add the bandgap index column -> the graph state for packed_v4 (conservative-
    # autograd pretraining). to_jimage is NOT recomputed here (degenerate from bond_length);
    # it comes from a REBUILD (build-mptrj) whose compactor keeps the builder's exact offset.
    # Run AFTER build-mptrj. Resumable + parallel; run where the graphs live (cluster scratch).
    if not os.path.exists(args.input):
        sys.exit(f"error: MPtrj JSON not found: {args.input}")
    graph_dir = args.graph_dir or "database/datafiles/MPtrj/graphs_v4"
    if not os.path.isdir(graph_dir):
        sys.exit(f"error: graph dir not found: {graph_dir}")
    if args.index and not os.path.exists(args.index):
        sys.exit(f"error: index not found: {args.index}")
    from database.Extract_MPtrj import iter_mptrj_frames
    print(f"Augmenting graphs in {graph_dir} with multitask targets "
          f"(forces/magmom/stress) from {args.input}; bandgap -> {args.index}")
    database.augment_graphs_with_physics(
        iter_mptrj_frames(args.input), graph_dir=graph_dir, index_path=args.index,
        n_workers=args.workers, limit=args.limit)
    print("Done. Re-pack (pack-dataset) into packed_v4 to carry the new fields.")


def cmd_fetch_dos(args):
    # Fetch MP total DOS and attach it (resampled to the fixed E_F-aligned grid) to the
    # relaxed MP graphs the index points at -> the DOS-bearing population (a separate
    # masked-union member) for multitask pretraining. Needs MP_API_KEY + network.
    # Resumable; then pack-dataset on this index -> the DOS pack (has_dos=true).
    if not os.path.exists(args.index):
        sys.exit(f"error: index not found: {args.index}")
    if not os.environ.get("MP_API_KEY"):
        sys.exit("error: set the MP_API_KEY environment variable before fetching DOS.")
    from database.Download_MP_dos import fetch_and_attach_dos, N_ENERGY
    print(f"Fetching MP DOS for {args.index} (E_F-aligned, {N_ENERGY} bins, "
          f"broaden={args.broaden} eV, {args.workers} workers) -> graph['dos']. "
          f"DOS covers a SUBSET of MP.")
    fetch_and_attach_dos(args.index, broaden_ev=args.broaden, limit=args.limit,
                         workers=args.workers)
    print("Done. pack-dataset on this index -> the DOS pack; verify has_dos=true.")


def cmd_pack_dataset(args):
    # Pack a cgv4 index (any dataset: MP_Energy, SC, MPtrj) into the columnar
    # binary format that PackedCIFDataV4 trains from: graph JSONs are parsed and
    # neighbor lists extracted ONCE here; training then memory-maps tensors
    # (~20-40x faster sample reads, no per-epoch JSON cost). Point a config's
    # index_path at the output directory to train from it.
    if not os.path.exists(args.index):
        sys.exit(f"error: index not found: {args.index}")
    sys.path.insert(0, os.path.join("models", "common"))
    from pack import pack_dataset
    print(f"Packing {args.index} -> {args.out}")
    if args.limit:
        print(f"  (limited to first {args.limit} samples)")
    pack_dataset(args.index, args.out, n_workers=args.workers,
                 limit=args.limit)
    print("Done.")


def cmd_download_nonsc(args):
    # Lazy import: only needs pymatgen/mp_api when actually downloading, and keeps
    # the (heavy) import off the path of other commands.
    from database.Download_MP_data import gen_dataset
    if not os.environ.get("MP_API_KEY"):
        sys.exit("error: set the MP_API_KEY environment variable before downloading "
                 "(e.g. $env:MP_API_KEY = '...').")
    gen_dataset(
        min_band_gap=args.min_band_gap,
        prop_file=args.prop_file,
        cif_loc=args.cif_loc,
        limit=args.limit,
        chunk_size=args.chunk_size,
    )


def cmd_download_energy(args):
    # Lazy import: only needs pymatgen/mp_api when actually downloading, and keeps
    # the (heavy) import off the path of other commands.
    from database.Download_MP_energy import gen_dataset
    if not os.environ.get("MP_API_KEY"):
        sys.exit("error: set the MP_API_KEY environment variable before downloading "
                 "(e.g. $env:MP_API_KEY = '...').")
    gen_dataset(
        prop_file=args.prop_file,
        cif_loc=args.cif_loc,
        limit=args.limit,
        chunk_size=args.chunk_size,
        theoretical=None if args.include_theoretical else False,
    )


def cmd_train(args):
    if not os.path.exists(args.config):
        sys.exit(f"error: config not found: {args.config}")
    script = os.path.join("models", "CGCNNMain.py")
    if not os.path.exists(script):
        sys.exit(f"error: {script} not found (run main.py from the project root)")

    # Run CGCNNMain.py as its own process: it expects its directory (models/) on
    # sys.path for its imports, while data paths in the config are resolved
    # relative to the project root, which we inherit as the cwd.
    print(f"Training CGCNN with config {args.config} ...")
    result = subprocess.run([sys.executable, script, args.config])
    if result.returncode != 0:
        sys.exit(result.returncode)


def _run_trainer(script, script_dir, config, banner):
    """Run a model trainer entry as its own process. The entry self-inserts
    models/common on sys.path; we add its own package dir to PYTHONPATH so it can
    import its sibling modules, and inherit cwd so config data paths resolve."""
    if not os.path.exists(config):
        sys.exit(f"error: config not found: {config}")
    if not os.path.exists(script):
        sys.exit(f"error: {script} not found (run main.py from the project root)")
    env = os.environ.copy()
    env["PYTHONPATH"] = script_dir + os.pathsep + env.get("PYTHONPATH", "")
    print(banner)
    result = subprocess.run([sys.executable, script, os.path.abspath(config)],
                            cwd=os.getcwd(), env=env)
    if result.returncode != 0:
        sys.exit(result.returncode)


def cmd_train_mpnn(args):
    _run_trainer(os.path.join("models", "MPNN", "MPNNMain.py"),
                 os.path.join("models", "MPNN"), args.config,
                 f"Training MPNN with config {args.config} ...")


def cmd_train_gps(args):
    _run_trainer(os.path.join("models", "GPSTransformer", "gps_main.py"),
                 os.path.join("models", "GPSTransformer"), args.config,
                 f"Training GPSTransformer with config {args.config} ...")


def cmd_embed_mace(args):
    from models.head.embed_mace import embed_index
    embed_index(args.index, args.cif_dir, args.out, model=args.model,
                device=args.device, only_label=args.only_label, limit=args.limit)


def cmd_embed_gps(args):
    # Export per-structure GPS encoder embeddings from a pretrained multitask checkpoint
    # for the frozen T_c probe (the twin of embed-mace). Writes <id>.npy of per-atom h.
    if not os.path.exists(args.checkpoint):
        sys.exit(f"error: checkpoint not found: {args.checkpoint}")
    if not os.path.exists(args.index):
        sys.exit(f"error: index not found: {args.index}")
    sys.path.insert(0, os.path.join("models", "common"))
    sys.path.insert(0, os.path.join("models", "GPSTransformer"))
    from models.head.embed_gps import embed_index
    embed_index(args.checkpoint, args.index, args.out,
                device=args.device, batch_size=args.batch_size)


def cmd_train_head(args):
    if not os.path.exists(args.config):
        sys.exit(f"error: config not found: {args.config}")
    from models.head.HeadMain import run
    run(args.config)


def cmd_plot(args):
    import plot  # imported lazily so matplotlib isn't loaded for other commands
    if args.epoch_log:
        missing = [f for f in args.epoch_log if not os.path.exists(f)]
        if missing:
            sys.exit("error: epoch-log file(s) not found: " + ", ".join(missing))
        plot.plot_epoch_logs(args.epoch_log)
        return
    if not os.path.exists(args.results):
        sys.exit(f"error: results file not found: {args.results} (train a model first)")
    plot.plot_results(args.results)


TOP_DESCRIPTION = """\
ML superconductor screening project entry point.

A single CLI for the whole workflow: build the database files, download non-SC
negatives, train and evaluate the models (baseline CGCNN or the crystal_graph_v4
MPNN, for T_c regression or SC/non-SC classification), and plot predictions. Run
all commands from the project root, since paths (database/datafiles/MP/cifs/,
configs/orig_basic.json, ...) are resolved relative to it.
"""

TOP_EPILOG = """\
commands:
  build-db        generate / regenerate the database files used by the models
  download-nonsc  download non-superconductor negatives from the Materials Project
  download-energy download experimental MP structures with energy targets
  train           train + evaluate the baseline CGCNN from a JSON config
  train-mpnn      train + evaluate the crystal_graph_v4 MPNN from a JSON config
  plot            scatter-plot test predictions vs. targets

typical workflow (from the project root):
  python main.py build-db --kind atom-init          # element feature file
  python main.py build-db --kind basic              # superconductor dataset pickle
  python main.py train configs/orig_basic.json           # regression: train + evaluate on test

  # SC/non-SC classifier (needs MP_API_KEY for the download):
  python main.py download-nonsc --limit 5000        # non-SC negatives
  python main.py build-db --kind basic \\
      --source database/datafiles/MP/id_prop.csv database/datafiles/MP/cifs/ \\
      --nonsc-source database/datafiles/Non_SC_DB_MP/Non_SC.csv database/datafiles/Non_SC_DB_MP/cifs/ \\
      --output database/datafiles/MP/SC_MP_basic_combined   # combined labeled dataset
  python main.py train configs/orig_classify_basic.json  # classification: train + evaluate

  python main.py plot                               # visualize models/test_result.csv
  python main.py plot --epoch-log models/MPNN/mpnn_result_epoch_log.csv   # one run's per-epoch stats
  python main.py plot --epoch-log run_a/..._epoch_log.csv run_b/..._epoch_log.csv  # compare runs

  # MP energy-target benchmark (needs MP_API_KEY for the download):
  python main.py download-energy                    # experimental MP structures + energies
  python main.py build-db --kind cgv4 --has-header \\
      --source database/datafiles/MP_Energy/mp_energy.csv database/datafiles/MP_Energy/cifs/ \\
      --output database/datafiles/MP_Energy/MP_Energy_V4 \\
      --graph-dir database/datafiles/MP_Energy/graphs_v4   # energy cgv4 index + its own graphs
  python main.py train-mpnn configs/mpnn_eform.json # regress formation energy

See 'python main.py <command> -h' for command-specific options.
"""

BUILD_DB_EPILOG = """\
database file kinds (build-db --kind):
  atom-init  per-element feature vectors written to atom_init.json
  basic      parsed dataset pickle/csv of (id, value, struc_dict, label)
  cgv4       crystal_graph_v4 pre-computed graphs (consumed by the MPNN model)

examples:
  python main.py build-db --kind atom-init
  python main.py build-db --kind basic --parallel
  python main.py build-db --kind cgv4 --limit 50
  python main.py build-db --kind basic --source database/datafiles/MP/id_prop.csv database/datafiles/MP/cifs/
  python main.py build-db --kind basic \\
      --source database/datafiles/MP/id_prop.csv database/datafiles/MP/cifs/ \\
      --nonsc-source database/datafiles/Non_SC_DB_MP/Non_SC.csv database/datafiles/Non_SC_DB_MP/cifs/

notes:
  * Run from the project root; all paths are resolved relative to it.
  * The default source is database/datafiles/MP/id_prop.csv + database/datafiles/MP/cifs/, a headerless
    "<cif_filename>,<tc>" file. Pass --has-header for a CSV with 'cif'/'tc' columns
    (e.g. 3DSC_MP.csv).
  * basic writes both <output>.pickle and <output>.csv; the CNN loads the pickle.
  * --limit N processes only the first N rows of each source -- handy for a quick
    smoke test before a full (slow) run.
  * Regenerating atom-init will not byte-match the committed atom_init.json: only
    the logic is preserved, and pymatgen's electron-affinity reference data has
    changed since the committed file was created.
"""


def build_parser():
    parser = argparse.ArgumentParser(
        prog="main.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=TOP_DESCRIPTION,
        epilog=TOP_EPILOG,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    db = sub.add_parser(
        "build-db",
        help="generate or regenerate the database files used by the CNN",
        description="Generate or regenerate a database file used by the CNN.",
        epilog=BUILD_DB_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    db.add_argument(
        "--kind",
        required=True,
        choices=["atom-init", "basic", "cgv4"],
        help="which file to (re)generate: element features, the basic dataset, "
             "or crystal_graph_v4 pre-computed graphs",
    )
    db.add_argument(
        "--source",
        action="append",
        nargs=2,
        metavar=("CSV", "CIF_DIR"),
        help="a superconductor id->property CSV and its cif directory; repeatable. "
             "Labeled SC (label=1) in the database. Ignored for --kind atom-init. "
             f"Default: {DEFAULT_SOURCE[0]} {DEFAULT_SOURCE[1]}",
    )
    db.add_argument(
        "--nonsc-source",
        action="append",
        nargs=2,
        metavar=("CSV", "CIF_DIR"),
        help="[basic/cgv4] a non-superconductor id->property CSV and its cif "
             "directory; repeatable. Labeled non-SC (label=0) in the database for "
             "the SC/non-SC classifier. Generate one with 'main.py download-nonsc'.",
    )
    db.add_argument(
        "--output",
        help="output path/prefix (kind-specific default; basic appends .pickle/.csv)",
    )
    db.add_argument(
        "--has-header",
        action="store_true",
        help="[basic/cgv4] source CSV has a header row with 'cif' and 'tc' columns "
             "(default: headerless, columns are filename,tc)",
    )
    db.add_argument(
        "--limit",
        type=int,
        default=None,
        help="[basic/cgv4] only process the first N rows of each source (useful for testing)",
    )
    db.add_argument(
        "--parallel",
        action="store_true",
        help="[basic] parse cifs across threads",
    )
    db.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="[basic] rows per thread when --parallel (default: 1000)",
    )
    db.add_argument(
        "--timing",
        action="store_true",
        help="[basic] print database construction timing",
    )
    db.add_argument(
        "--max-z",
        type=int,
        default=95,
        help="[atom-init] generate features for Z = 1 .. max_z - 1 (default: 95, "
             "covers actinides through Pu present in the MP energy data)",
    )
    db.add_argument(
        "--graph-dir",
        default=None,
        help=f"[cgv4] directory to write per-material JSON graph files "
             f"(default: {DEFAULT_CGV4_GRAPH_DIR})",
    )
    db.set_defaults(func=cmd_build_db)

    tr = sub.add_parser(
        "train",
        help="train and evaluate the baseline CGCNN from a JSON config",
        description="Train the baseline CGCNN and evaluate it on the held-out test "
                    "split, driven by a JSON config. The config's \"task\" key selects "
                    "T_c regression (configs/orig_basic.json) or SC/non-SC classification "
                    "(configs/orig_classify_basic.json). Runs models/CGCNNMain.py; predictions "
                    "and a per-epoch telemetry log are written under the config's out_file.",
    )
    tr.add_argument(
        "config",
        help="path to the JSON training config (e.g. configs/orig_basic.json or configs/orig_classify_basic.json)",
    )
    tr.set_defaults(func=cmd_train)

    tm = sub.add_parser(
        "train-mpnn",
        help="train and evaluate the MPNN (v4 graph) model from a JSON config",
        description="Train CrystalMPNN using pre-computed crystal_graph_v4 graphs. "
                    "Build the graph database first with: python main.py build-db --kind cgv4. "
                    "Runs models/MPNN/MPNNMain.py; results are written to config's out_file + '.csv'.",
    )
    tm.add_argument(
        "config",
        help="path to the JSON training config (e.g. configs/mpnn_basic.json)",
    )
    tm.set_defaults(func=cmd_train_mpnn)

    tg = sub.add_parser(
        "train-gps",
        help="train and evaluate the GPSTransformer (v4 graph) model from a JSON config",
        description="Train GPSCrystalNet (local angle-biased shell attention + "
                    "within-crystal global attention) on crystal_graph_v4 graphs. "
                    "Runs models/GPSTransformer/gps_main.py; shares the data layer "
                    "and training loop with the MPNN via models/common.",
    )
    tg.add_argument("config", help="path to the JSON training config (e.g. configs/gps/gps_eform.json)")
    tg.set_defaults(func=cmd_train_gps)

    dn = sub.add_parser(
        "download-nonsc",
        help="download non-superconductor (large band gap) structures from the Materials Project",
        description="Download large-band-gap (non-superconducting) materials from the "
                    "Materials Project as negative examples for the SC/non-SC classifier. "
                    "Requires the MP_API_KEY environment variable. Writes CIFs + a prop CSV "
                    "that you then pass to 'build-db --nonsc-source CSV CIF_DIR'.",
    )
    dn.add_argument("--min-band-gap", type=float, default=4.0,
                    help="minimum band gap in eV (default: 4.0; high cutoff keeps only "
                         "clear insulators, avoiding metallic/SC contamination of negatives)")
    dn.add_argument("--prop-file", default=None,
                    help="output id->property CSV (default: database/datafiles/Non_SC_DB_MP/Non_SC.csv)")
    dn.add_argument("--cif-loc", default=None,
                    help="output CIF directory (default: database/datafiles/Non_SC_DB_MP/cifs/)")
    dn.add_argument("--limit", type=int, default=None,
                    help="cap the number of materials downloaded (for class balance vs. ~5.8k SCs)")
    dn.add_argument("--chunk-size", type=int, default=1000,
                    help="MP API page size (default: 1000)")
    dn.set_defaults(func=cmd_download_nonsc)

    de = sub.add_parser(
        "download-energy",
        help="download experimental Materials Project structures with energy targets",
        description="Download experimentally-observed materials from the Materials "
                    "Project (theoretical=False by default) with their energy-above-hull "
                    "and formation-energy targets. Requires the MP_API_KEY environment "
                    "variable. Writes CIFs + mp_energy.csv (columns: cif, material_id, "
                    "e_above_hull, formation_energy_per_atom); both targets are kept as "
                    "columns and the training target is chosen via the config's "
                    "target_column. Feed it to 'build-db --kind cgv4 --has-header "
                    "--source mp_energy.csv CIF_DIR'.",
    )
    de.add_argument("--prop-file", default=None,
                    help="output id->property CSV (default: database/datafiles/MP_Energy/mp_energy.csv)")
    de.add_argument("--cif-loc", default=None,
                    help="output CIF directory (default: database/datafiles/MP_Energy/cifs/)")
    de.add_argument("--limit", type=int, default=None,
                    help="cap the number of materials downloaded (for testing)")
    de.add_argument("--chunk-size", type=int, default=1000,
                    help="MP API page size (default: 1000)")
    de.add_argument("--include-theoretical", action="store_true",
                    help="include theoretical (non-experimental) materials too")
    de.set_defaults(func=cmd_download_energy)

    mt = sub.add_parser(
        "build-mptrj",
        help="build cgv4 graphs for every MPtrj trajectory frame (energy-only warm-up)",
        description="Stream the bulk MPtrj JSON (~12 GB, ~1.5M frames) and build one "
                    "compact crystal_graph_v4 per frame, with each frame's ef_per_atom "
                    "(-> formation_energy_per_atom) and energy_per_atom indexed as "
                    "selectable targets. Resumable: re-running skips frames whose graph "
                    "already exists. Needs the RPToleranceFactor builder on the path "
                    "(run locally), then sync graphs_v4/ + the index pickle to the cluster. "
                    "Train with: python main.py train-mpnn <config> (index_path -> the pickle).",
    )
    mt.add_argument("--input", default="database/datafiles/MPtrj/MPtrj_2022.9_full.json",
                    help="bulk MPtrj JSON (default: database/datafiles/MPtrj/MPtrj_2022.9_full.json)")
    mt.add_argument("--graph-dir", default=None,
                    help="output dir for per-frame graph JSONs (default: database/datafiles/MPtrj/graphs_v4)")
    mt.add_argument("--output", default=None,
                    help="index pickle/csv path prefix (default: database/datafiles/MPtrj/MPtrj_V4)")
    mt.add_argument("--limit", type=int, default=None,
                    help="only build the first N frames (for a quick end-to-end test)")
    mt.add_argument("--workers", type=int, default=None,
                    help="worker processes (default: os.cpu_count())")
    mt.set_defaults(func=cmd_build_mptrj)

    ap = sub.add_parser(
        "augment-positions",
        help="backfill frac_coords + lattice onto existing MPtrj graphs (no rebuild)",
        description="Stream the MPtrj JSON and attach each frame's fractional coords "
                    "+ lattice to its already-built compact graph (matched by <id>.json), "
                    "verifying atom order against the source structure. The cheap "
                    "alternative to a full Voronoi rebuild for the long-range distance "
                    "bias. Resumable (graphs already carrying positions are skipped). "
                    "After this, re-run pack-dataset to carry positions into the pack.",
    )
    ap.add_argument("--input", default="database/datafiles/MPtrj/MPtrj_2022.9_full.json",
                    help="bulk MPtrj JSON (the same source build-mptrj used)")
    ap.add_argument("--graph-dir", default=None,
                    help="dir of per-frame graph JSONs to augment (default: database/datafiles/MPtrj/graphs_v4)")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N frames")
    ap.add_argument("--workers", type=int, default=None,
                    help="worker processes (default: os.cpu_count())")
    ap.set_defaults(func=cmd_augment_positions)

    aph = sub.add_parser(
        "augment-physics",
        help="backfill multitask targets (forces/magmom/stress + positions) onto MPtrj graphs",
        description="Stream the MPtrj JSON and attach each frame's per-atom forces/magmom, "
                    "per-structure stress, and fractional coords + lattice to its compact "
                    "graph (Z-verified atom order); also add the bandgap index column. Run "
                    "AFTER build-mptrj — to_jimage (needed for exact PBC forces) is carried by "
                    "the rebuild's compactor, NOT recomputed here (it's degenerate from "
                    "bond_length for multi-image bonds). Resumable. pack-dataset -> packed_v4.",
    )
    aph.add_argument("--input", default="database/datafiles/MPtrj/MPtrj_2022.9_full.json",
                     help="bulk MPtrj JSON (the same source build-mptrj used)")
    aph.add_argument("--graph-dir", default=None,
                     help="dir of per-frame graph JSONs to augment (default: database/datafiles/MPtrj/graphs_v4)")
    aph.add_argument("--index", default=None,
                     help="index pickle to add the bandgap column to (matched by id)")
    aph.add_argument("--limit", type=int, default=None, help="only process the first N frames")
    aph.add_argument("--workers", type=int, default=None,
                     help="worker processes (default: os.cpu_count())")
    aph.set_defaults(func=cmd_augment_physics)

    fd = sub.add_parser(
        "fetch-dos",
        help="fetch MP total DOS + attach to relaxed MP graphs (the multitask DOS target)",
        description="Stream an MP index, fetch each material's total DOS via the MP-API, "
                    "resample it onto a fixed E_F-aligned grid (DOS_N_ENERGY bins over "
                    "[-10,+5] eV, optional Gaussian broadening), and attach it as "
                    "graph['dos'] (atomic, resumable). DOS exists for a SUBSET of MP (only "
                    "materials with an electronic-structure calc), so expect many no_dos "
                    "skips. Needs MP_API_KEY. pack-dataset after -> the DOS pack.",
    )
    fd.add_argument("--index", required=True,
                    help="MP index pickle (id + graph_path) whose graphs get graph['dos']")
    fd.add_argument("--broaden", type=float, default=0.1,
                    help="Gaussian broadening sigma in eV (default 0.1; 0 = none)")
    fd.add_argument("--limit", type=int, default=None, help="only process the first N materials")
    fd.add_argument("--workers", type=int, default=8,
                    help="parallel DOS-fetch threads (default 8; the fetch is network-bound). "
                         "A bulk has-DOS pre-filter runs first so the no-DOS majority is "
                         "settled without a per-material download.")
    fd.set_defaults(func=cmd_fetch_dos)

    pk = sub.add_parser(
        "pack-dataset",
        help="pack a cgv4 index + graphs into the fast columnar training format",
        description="One-time conversion: parse every graph JSON referenced by an "
                    "index pickle and store the extracted neighbor data as flat "
                    "binary arrays + offsets (see models/MPNN/MPNNPack.py). Training "
                    "configs then point index_path at the output DIRECTORY; sample "
                    "tensors are bitwise-identical to the lazy loader but ~20-40x "
                    "faster to read (no JSON parse / neighbor build per epoch). "
                    "max_num_nbr / poly / angle flags stay read-time parameters — "
                    "one pack serves every config variant.",
    )
    pk.add_argument("--index", required=True,
                    help="source index pickle (e.g. database/datafiles/MPtrj/MPtrj_V4.pickle)")
    pk.add_argument("--out", required=True,
                    help="output pack directory (e.g. .../MPtrj/packed_v1)")
    pk.add_argument("--workers", type=int, default=None,
                    help="extraction worker processes (default: os.cpu_count())")
    pk.add_argument("--limit", type=int, default=None,
                    help="only pack the first N samples (for testing)")
    pk.set_defaults(func=cmd_pack_dataset)

    em = sub.add_parser(
        "embed-mace",
        help="one-time frozen-MACE embedding pass over a cgv4 index",
        description="Compute per-atom MACE-MP-0 descriptor matrices (invariant "
                    "l=0 channels) for every index row and write one <id>.npy "
                    "per structure (+ manifest.json / failed.txt). Atom order is "
                    "asserted against each stored graph JSON. Resumable: "
                    "existing .npy files are skipped. See models/head/embed_mace.py.",
    )
    em.add_argument("--index", required=True, help="cgv4 index pickle/csv")
    em.add_argument("--cif-dir", required=True, nargs="+",
                    help="CIF directory (repeatable; tried in order per id)")
    em.add_argument("--out", required=True, help="output embedding directory")
    em.add_argument("--model", default="medium", help="MACE-MP-0 size (default medium)")
    em.add_argument("--device", default="cuda")
    em.add_argument("--only-label", type=int, default=None,
                    help="restrict to index rows with this label (1=SC, 0=non-SC)")
    em.add_argument("--limit", type=int, default=None)
    em.set_defaults(func=cmd_embed_mace)

    eg = sub.add_parser(
        "embed-gps",
        help="export per-structure GPS encoder embeddings from a pretrained checkpoint",
        description="Run a PRETRAINED multitask GPSCrystalNet encoder over the transfer "
                    "structures and save each structure's per-atom h as <id>.npy — the same "
                    "per-structure layout embed-mace writes, so the frozen TcHead pipeline "
                    "(train-head) consumes them UNCHANGED. The transfer dataset uses the "
                    "encoder's training feature flags (read from the checkpoint). Resumable. "
                    "See models/head/embed_gps.py.",
    )
    eg.add_argument("--checkpoint", required=True, help="pretrained GPS checkpoint (.pth.tar)")
    eg.add_argument("--index", required=True, help="transfer-structure index pickle or pack dir")
    eg.add_argument("--out", required=True, help="output dir for <id>.npy embeddings")
    eg.add_argument("--device", default="cpu", help="cpu or cuda (default cpu)")
    eg.add_argument("--batch-size", type=int, default=64)
    eg.set_defaults(func=cmd_embed_gps)

    th = sub.add_parser(
        "train-head",
        help="train the small pluggable-encoder T_c head (probe + trunk + ensemble)",
        description="Linear probe + (optional) SC/non-SC trunk pretraining + "
                    "seed-ensembled T_c regression on frozen encoder embeddings "
                    "with the physical-descriptor bypass. See models/head/HeadMain.py.",
    )
    th.add_argument("config", help="head config JSON (see configs/head/)")
    th.set_defaults(func=cmd_train_head)

    pl = sub.add_parser(
        "plot",
        help="scatter-plot CNN test predictions vs. targets",
        description="Plot predicted vs. target T_c from a CGCNN results CSV.",
    )
    pl.add_argument(
        "--results",
        default="models/test_result.csv",
        help="results CSV written by training (default: models/test_result.csv)",
    )
    pl.add_argument(
        "--epoch-log",
        nargs="+",
        metavar="EPOCH_LOG",
        help="instead of the scatter, overlay per-epoch stats from one or more "
             "*_epoch_log.csv files (interactive: a row=metric / col=run table "
             "toggles individual lines; legend grouped by run)",
    )
    pl.set_defaults(func=cmd_plot)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
