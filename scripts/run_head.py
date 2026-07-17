"""Run one or more transfer-head configs (train-head) in sequence.

A thin wrapper around models.head.HeadMain.run — the same entrypoint
`main.py train-head` uses — so head training runs on the cluster without
shipping main.py or the database/ package (HeadMain imports only models.head +
models.common). Each config writes its own timestamped model_data/ run dir.

    python scripts/run_head.py configs/head/a.json configs/head/b.json ...
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "models", "common"),
           os.path.join(_ROOT, "models", "GPSTransformer"),
           os.path.join(_ROOT, "models", "head")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.head.HeadMain import run  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: run_head.py <config.json> [config.json ...]")
    for cfg in sys.argv[1:]:
        print(f"\n===== run_head: {cfg} =====", flush=True)
        run(cfg)
