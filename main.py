"""Entry point for the ML superconductor (CGCNN) project.

A single CLI for the whole workflow; run all commands from the project root:

  * build-db : generate / regenerate the database files consumed by the CNN
  * train    : train + evaluate the CGCNN from a JSON config
  * plot     : scatter-plot test predictions vs. targets

Examples:
  python main.py build-db --kind basic
  python main.py train configs/basic.json
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
    "basic": "database/datafiles/MP/id_prop_basic",
    "cgv4": "database/datafiles/MP/id_prop_v4",
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
    script = os.path.join("CNN", "CGCNNMain.py")
    if not os.path.exists(script):
        sys.exit(f"error: {script} not found (run main.py from the project root)")

    # Run CGCNNMain.py as its own process: it expects its directory (CNN/) on
    # sys.path for its imports, while data paths in the config are resolved
    # relative to the project root, which we inherit as the cwd.
    print(f"Training CGCNN with config {args.config} ...")
    result = subprocess.run([sys.executable, script, args.config])
    if result.returncode != 0:
        sys.exit(result.returncode)


def cmd_train_mpnn(args):
    if not os.path.exists(args.config):
        sys.exit(f"error: config not found: {args.config}")
    script = os.path.join("CNN", "MPNN", "MPNNMain.py")
    if not os.path.exists(script):
        sys.exit(f"error: {script} not found (run main.py from the project root)")

    # MPNNMain.py lives in CNN/MPNN/ and imports from that directory, so we
    # run it from there with the project root injected into PYTHONPATH so that
    # relative data paths in the config resolve correctly.
    env = os.environ.copy()
    mpnn_dir = os.path.join("CNN", "MPNN")
    env["PYTHONPATH"] = mpnn_dir + os.pathsep + env.get("PYTHONPATH", "")
    print(f"Training MPNN with config {args.config} ...")
    result = subprocess.run(
        [sys.executable, script, os.path.abspath(args.config)],
        cwd=os.getcwd(),
        env=env,
    )
    if result.returncode != 0:
        sys.exit(result.returncode)


def cmd_plot(args):
    import plot  # imported lazily so matplotlib isn't loaded for other commands
    if args.epoch_log:
        if not os.path.exists(args.epoch_log):
            sys.exit(f"error: epoch-log file not found: {args.epoch_log}")
        plot.plot_epoch_log(args.epoch_log)
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
configs/basic.json, ...) are resolved relative to it.
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
  python main.py train configs/basic.json           # regression: train + evaluate on test

  # SC/non-SC classifier (needs MP_API_KEY for the download):
  python main.py download-nonsc --limit 5000        # non-SC negatives
  python main.py build-db --kind basic \\
      --source database/datafiles/MP/id_prop.csv database/datafiles/MP/cifs/ \\
      --nonsc-source database/datafiles/Non_SC_DB_MP/Non_SC.csv database/datafiles/Non_SC_DB_MP/cifs/ \\
      --output database/datafiles/MP/id_prop_basic_combined   # combined labeled dataset
  python main.py train configs/classify_basic.json  # classification: train + evaluate

  python main.py plot                               # visualize CNN/test_result.csv

  # MP energy-target benchmark (needs MP_API_KEY for the download):
  python main.py download-energy                    # experimental MP structures + energies
  python main.py build-db --kind cgv4 --has-header \\
      --source database/datafiles/MP_Energy/mp_energy.csv database/datafiles/MP_Energy/cifs/ \\
      --output database/datafiles/MP_Energy/id_prop_v4_energy # multi-target cgv4 index
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
        default=85,
        help="[atom-init] generate features for Z = 1 .. max_z - 1 (default: 85)",
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
                    "T_c regression (configs/basic.json) or SC/non-SC classification "
                    "(configs/classify_basic.json). Runs CNN/CGCNNMain.py; predictions "
                    "and a per-epoch telemetry log are written under the config's out_file.",
    )
    tr.add_argument(
        "config",
        help="path to the JSON training config (e.g. configs/basic.json or configs/classify_basic.json)",
    )
    tr.set_defaults(func=cmd_train)

    tm = sub.add_parser(
        "train-mpnn",
        help="train and evaluate the MPNN (v4 graph) model from a JSON config",
        description="Train CrystalMPNN using pre-computed crystal_graph_v4 graphs. "
                    "Build the graph database first with: python main.py build-db --kind cgv4. "
                    "Runs CNN/MPNN/MPNNMain.py; results are written to config's out_file + '.csv'.",
    )
    tm.add_argument(
        "config",
        help="path to the JSON training config (e.g. configs/mpnn_basic.json)",
    )
    tm.set_defaults(func=cmd_train_mpnn)

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

    pl = sub.add_parser(
        "plot",
        help="scatter-plot CNN test predictions vs. targets",
        description="Plot predicted vs. target T_c from a CGCNN results CSV.",
    )
    pl.add_argument(
        "--results",
        default="CNN/test_result.csv",
        help="results CSV written by training (default: CNN/test_result.csv)",
    )
    pl.add_argument(
        "--epoch-log",
        help="instead of the scatter, plot per-epoch stats from an *_epoch_log.csv "
             "(interactive: checkboxes toggle which columns are shown)",
    )
    pl.set_defaults(func=cmd_plot)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
