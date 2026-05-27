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
DEFAULT_SOURCE = ["database/MP/id_prop.csv", "database/MP/cifs/"]

DEFAULT_OUTPUTS = {
    "atom-init": "database/atom_init.json",
    "basic": "database/MP/id_prop_basic",
    "ce": "database/MP/id_prop_ce",
}


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

    # basic / ce both consume (csv, cif_dir) source pairs.
    sources = args.source if args.source else [DEFAULT_SOURCE]
    for csv_path, cif_dir in sources:
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
    else:  # ce
        database.generate_CE_DB(
            sources,
            output_file=out,
            has_header=args.has_header,
            limit=args.limit,
        )
    print("Done.")


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


def cmd_plot(args):
    if not os.path.exists(args.results):
        sys.exit(f"error: results file not found: {args.results} (train a model first)")
    import plot  # imported lazily so matplotlib isn't loaded for other commands
    plot.plot_results(args.results)


TOP_DESCRIPTION = """\
ML superconductor (CGCNN) project entry point.

A single CLI for the whole workflow: build the database files, train and evaluate
the CGCNN, and plot its predictions. Run all commands from the project root, since
paths (database/MP/cifs/, configs/basic.json, ...) are resolved relative to it.
"""

TOP_EPILOG = """\
commands:
  build-db   generate / regenerate the database files used by the CNN
  train      train + evaluate the CGCNN from a JSON config
  plot       scatter-plot test predictions vs. targets

typical workflow (from the project root):
  python main.py build-db --kind atom-init       # element feature file
  python main.py build-db --kind basic           # dataset pickle
  python main.py train configs/basic.json        # train, then evaluate on the test split
  python main.py plot                            # visualize CNN/test_result.csv

See 'python main.py <command> -h' for command-specific options.
"""

BUILD_DB_EPILOG = """\
database file kinds (build-db --kind):
  atom-init  per-element feature vectors written to atom_init.json
  basic      parsed dataset pickle/csv of (id, value, struc_dict)
  ce         the basic dataset plus a per-site coordination-environment column
             ('ce'); consumed by the CGCNNCoordEnv model

examples:
  python main.py build-db --kind atom-init
  python main.py build-db --kind basic --parallel
  python main.py build-db --kind ce --limit 50
  python main.py build-db --kind basic --source database/MP/id_prop.csv database/MP/cifs/

notes:
  * Run from the project root; all paths are resolved relative to it.
  * The default source is database/MP/id_prop.csv + database/MP/cifs/, a headerless
    "<cif_filename>,<tc>" file. Pass --has-header for a CSV with 'cif'/'tc' columns
    (e.g. 3DSC_MP.csv).
  * basic/ce write both <output>.pickle and <output>.csv; the CNN loads the pickle.
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
        choices=["atom-init", "basic", "ce"],
        help="which file to (re)generate: element features, the basic dataset, "
             "or the dataset with coordination environments",
    )
    db.add_argument(
        "--source",
        action="append",
        nargs=2,
        metavar=("CSV", "CIF_DIR"),
        help="an id->property CSV and its cif directory; repeatable. Ignored for "
             f"--kind atom-init. Default: {DEFAULT_SOURCE[0]} {DEFAULT_SOURCE[1]}",
    )
    db.add_argument(
        "--output",
        help="output path/prefix (kind-specific default; basic/ce append .pickle/.csv)",
    )
    db.add_argument(
        "--has-header",
        action="store_true",
        help="[basic/ce] source CSV has a header row with 'cif' and 'tc' columns "
             "(default: headerless, columns are filename,tc)",
    )
    db.add_argument(
        "--limit",
        type=int,
        default=None,
        help="[basic/ce] only process the first N rows of each source (useful for testing)",
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
    db.set_defaults(func=cmd_build_db)

    tr = sub.add_parser(
        "train",
        help="train and evaluate the CGCNN from a JSON config",
        description="Train the CGCNN and evaluate it on the held-out test split, "
                    "driven by a JSON config such as configs/basic.json. "
                    "Runs CNN/CGCNNMain.py; test predictions are written to the "
                    "config's out_file + '.csv'.",
    )
    tr.add_argument(
        "config",
        help="path to the JSON training config (e.g. configs/basic.json)",
    )
    tr.set_defaults(func=cmd_train)

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
    pl.set_defaults(func=cmd_plot)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
