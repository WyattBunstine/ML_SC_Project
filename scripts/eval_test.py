#!/usr/bin/env python3
"""Evaluate a saved model's checkpoint on a test set and write per-sample
predictions for plotting — without re-training.

Two uses:
  1. Recover the test output of an interrupted run: the trainer only writes the
     test CSV after training finishes, so a cancelled run leaves a usable
     `*_model_best.pth.tar` but no predictions. This regenerates them.
  2. Apples-to-apples model comparison: with `--test-ids FILE`, evaluate any
     model on the SAME set of material IDs, so different model families
     (MPNN vs. the original CGCNN) are scored on identical materials. Use
     `--export-ids FILE` on one run to capture its test split, then feed that
     file to the others.

It is "pluggable": each model family is a small adapter (see ModelAdapter
subclasses). The right one is auto-detected from the run's config.json. Adding a
new model = adding one adapter.

Output is a headerless `cif_id,target,pred` CSV (regression) or
`cif_id,true_label,p_sc` (classification) — the same format the trainers write,
so plot.py consumes it directly.

Usage:
    eval_test.py --run model_data/<run_dir>/ [--checkpoint best|last]
                 [--split test|val|train|all] [--test-ids FILE]
                 [--export-ids FILE] [--out FILE]

⚠ Comparison caveat: a fair comparison requires the shared test IDs to have been
held OUT of every compared model's TRAINING. Because the MPNN and original CGCNN
use different pickles with independent shuffles, their own test splits do NOT
contain the same materials even at the same seed. Designate a common holdout (or
train both with a shared split) and pass it via --test-ids; this tool evaluates
whatever IDs you give it but cannot verify they were held out.
"""
import argparse
import csv
import glob
import json
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Shared data layer lives in models/common; model code in models/MPNN (baseline
# under models/OriginalCGCNN, reached via models/).
for p in (os.path.join(REPO_ROOT, "models"),
          os.path.join(REPO_ROOT, "models", "common"),
          os.path.join(REPO_ROOT, "models", "MPNN")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _find_checkpoint(run_dir, which):
    """Return the path to the best/last/swa checkpoint in run_dir."""
    suffix = {"best": "_model_best.pth.tar",
              "last": "_checkpoint.pth.tar",
              "swa": "_swa.pth.tar"}[which]
    hits = sorted(glob.glob(os.path.join(run_dir, "*" + suffix)))
    if not hits:
        raise FileNotFoundError(
            f"no '*{suffix}' checkpoint in {run_dir} "
            f"(have: {[os.path.basename(x) for x in glob.glob(os.path.join(run_dir, '*.pth.tar'))]})")
    return hits[0]


class ModelAdapter:
    """A model family that can rebuild itself from a run dir and run inference.

    Subclasses implement matches/__init__/split_indices/cif_ids/predict. All run
    on a fixed device and in eval mode.
    """
    name = "base"

    @staticmethod
    def matches(config):
        raise NotImplementedError

    def __init__(self, run_dir, config, ckpt_path, device):
        self.run_dir, self.config, self.device = run_dir, config, device
        self.is_classification = config.get("task", "regression") == "classification"
        # Load on CPU; the model is moved to `device` after load_state_dict, while
        # the normalizer's tensors stay on CPU (we denorm CPU model outputs).
        self.ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    def split_indices(self, split):
        """Dataset positions for split in {'test','val','train','all'}."""
        raise NotImplementedError

    def id_to_index(self):
        """Map cif_id -> dataset position."""
        raise NotImplementedError

    def predict(self, indices):
        """Return list of (cif_id, target_or_label, prediction)."""
        raise NotImplementedError


class MPNNAdapter(ModelAdapter):
    name = "MPNN"

    @staticmethod
    def matches(config):
        return "index_path" in config

    def __init__(self, run_dir, config, ckpt_path, device):
        super().__init__(run_dir, config, ckpt_path, device)
        from data import load_cif_dataset
        from MPNNModel import CrystalMPNN
        from MPNNMain import Normalizer

        c = config
        edge_agg = c.get("edge_aggregation", c.get("aggregation", "ecn_weighted"))
        self.dataset = load_cif_dataset(
            c["index_path"],
            max_num_nbr=c.get("max_num_nbr", 14),
            max_num_poly_nbr=c.get("max_num_poly_nbr", 16),
            graph_cache_size=c.get("graph_cache_size", 4096),
            target_column=c.get("target_column"),
            use_bond_angles=c.get("use_bond_angles", False),
            use_poly_edges=c.get("use_poly_edges", True),
            # set_transformer checkpoints were trained WITH the angle-bias matrix;
            # evaluating without it would silently change the model's inputs.
            build_angle_bias=(edge_agg == "set_transformer"),
        )
        (a, n, _, poly, _, _), _, _, _ = self.dataset[0]
        self.model = CrystalMPNN(
            orig_atom_fea_len=a.shape[-1], nbr_fea_len=n.shape[-1], poly_fea_len=poly.shape[-1],
            atom_fea_len=c.get("atom_feat_len", 64), edge_hidden_dim=c.get("edge_hidden_dim", 128),
            n_conv=c.get("n_conv", 3), h_fea_len=c.get("h_feat_len", 128), n_h=c.get("n_hidden", 1),
            edge_aggregation=edge_agg,
            classification=self.is_classification, use_poly_edges=c.get("use_poly_edges", True),
            atom_pooling=c.get("atom_pooling", "mean"), set2set_steps=c.get("set2set_steps", 3),
            use_coord_magnitude=c.get("use_coord_magnitude", False),
            set_transformer_heads=c.get("set_transformer_heads", 4),
            poly_fusion=c.get("poly_fusion", "sum"),
        )
        # Feature-norm buffers are registered in __init__, so load_state_dict
        # restores both weights and those stats. No need to recompute stats.
        self.model.load_state_dict(self.ckpt["state_dict"])
        self.model.to(device).eval()

        self.normalizer = Normalizer(torch.zeros(2), transform="none")
        if not self.is_classification:
            self.normalizer.load_state_dict(self.ckpt["normalizer"])

    def id_to_index(self):
        return {rec[0]: i for i, rec in enumerate(self.dataset.data)}

    def split_indices(self, split):
        if split == "all":
            return list(range(len(self.dataset)))
        from data import get_sc_nonsc_loaders
        from MPNNMain import _parse_ratio
        c = self.config
        # Resolve split_by the way the RUN did, not the way today's data would:
        # 1. metadata.json records the value training actually resolved (the
        #    run-dir config.json is copied before resolution, so it lacks it);
        # 2. else the config's explicit value;
        # 3. else the shared auto-resolution (MPNNData.resolve_split_by).
        # Without (1), a checkpoint trained under auto->frame, evaluated after the
        # index gained mp_id (auto->material), would silently reproduce a
        # DIFFERENT split and score trained-on samples as 'test'.
        split_by = None
        meta_path = os.path.join(self.run_dir, "metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    split_by = json.load(f).get("dataset", {}).get("split_by")
            except (OSError, ValueError):
                pass
        if split_by is None:
            from data import resolve_split_by
            split_by = resolve_split_by(c.get("split_by"), self.dataset)
        L = get_sc_nonsc_loaders(
            self.dataset, batch_size=c.get("batch_size", 64),
            val_ratio=c["val_ratio"], test_ratio=c["test_ratio"],
            sc_to_nonsc_ratio=_parse_ratio(c.get("SC_to_non_SC_ratio")),
            num_workers=0, seed=c.get("split_seed", 123), split_by=split_by)
        key = {"test": "test_realistic", "val": "val_realistic", "train": "train"}[split]
        return list(L[key].sampler)

    def predict(self, indices):
        from torch.utils.data import DataLoader
        from data import collate_pool
        from MPNNMain import _to_input_var
        loader = DataLoader(self.dataset, batch_size=self.config.get("batch_size", 64),
                            sampler=indices, collate_fn=collate_pool, num_workers=0)
        cuda = self.device.type == "cuda"
        rows = []
        for input_batch, target, label, cif_ids in loader:
            with torch.no_grad():
                out = self.model(*_to_input_var(input_batch, cuda)).detach().cpu()
            if self.is_classification:
                p_sc = torch.exp(out)[:, 1].tolist()
                rows += list(zip(cif_ids, label.view(-1).tolist(), p_sc))
            else:
                pred = self.normalizer.denorm(out).view(-1).tolist()
                rows += list(zip(cif_ids, target.view(-1).tolist(), pred))
        return rows


class OrigCGCNNAdapter(ModelAdapter):
    name = "Original CGCNN"

    @staticmethod
    def matches(config):
        return "dataset" in config and "index_path" not in config

    def __init__(self, run_dir, config, ckpt_path, device):
        super().__init__(run_dir, config, ckpt_path, device)
        from OriginalCGCNN.data import CIFData
        from OriginalCGCNN.CGCNNOrig import CrystalGraphConvNet
        from CGCNNMain import Normalizer

        c = config
        self.dataset = CIFData(c["dataset_rd"], c["atom_init"], c["dataset"],
                               target_column=c.get("target_column"))
        structures, _, _, _ = self.dataset[0]
        self.model = CrystalGraphConvNet(
            structures[0].shape[-1], structures[1].shape[-1],
            atom_fea_len=c.get("atom_feat_len", 64), n_conv=c.get("n_conv", 3),
            h_fea_len=c.get("h_feat_len", 128), n_h=c.get("n_hidden", 1),
            classification=self.is_classification)
        self.model.load_state_dict(self.ckpt["state_dict"])
        self.model.to(device).eval()

        self.normalizer = Normalizer(torch.zeros(2))
        if not self.is_classification:
            self.normalizer.load_state_dict(self.ckpt["normalizer"])

    def id_to_index(self):
        return {rec[0]: i for i, rec in enumerate(self.dataset.id_prop_data)}

    def split_indices(self, split):
        if split == "all":
            return list(range(len(self.dataset)))
        from OriginalCGCNN.data import collate_pool, get_train_val_test_loader
        c = self.config
        tr, va, te = get_train_val_test_loader(
            dataset=self.dataset, collate_fn=collate_pool, batch_size=c.get("batch_size", 64),
            train_ratio=None, val_ratio=c["val_ratio"], test_ratio=c["test_ratio"],
            num_workers=0, return_test=True)
        return list({"test": te, "val": va, "train": tr}[split].sampler)

    def predict(self, indices):
        from torch.utils.data import DataLoader
        from OriginalCGCNN.data import collate_pool
        loader = DataLoader(self.dataset, batch_size=self.config.get("batch_size", 64),
                            sampler=indices, collate_fn=collate_pool, num_workers=0)
        cuda = self.device.type == "cuda"

        def to_dev(inp):
            if cuda:
                return (inp[0].cuda(non_blocking=True), inp[1].cuda(non_blocking=True),
                        inp[2].cuda(non_blocking=True),
                        [ci.cuda(non_blocking=True) for ci in inp[3]])
            return inp

        rows = []
        for inp, target, label, cif_ids in loader:
            with torch.no_grad():
                out = self.model(*to_dev(inp)).detach().cpu()
            if self.is_classification:
                p_sc = torch.exp(out)[:, 1].tolist()
                rows += list(zip(cif_ids, label.view(-1).tolist(), p_sc))
            else:
                pred = self.normalizer.denorm(out).view(-1).tolist()
                rows += list(zip(cif_ids, target.view(-1).tolist(), pred))
        return rows


ADAPTERS = [MPNNAdapter, OrigCGCNNAdapter]


def _detect(config):
    for cls in ADAPTERS:
        if cls.matches(config):
            return cls
    raise SystemExit("eval_test: cannot detect model type from config "
                     "(expected 'index_path' for MPNN or 'dataset' for original CGCNN).")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory (contains config.json + *.pth.tar)")
    ap.add_argument("--checkpoint", choices=["best", "last", "swa"], default="best")
    ap.add_argument("--split", choices=["test", "val", "train", "all"], default="test")
    ap.add_argument("--test-ids", help="file of cif_ids (one per line) to evaluate instead of --split")
    ap.add_argument("--export-ids", help="write the evaluated cif_ids to this file (for reuse on other models)")
    ap.add_argument("--out", help="output CSV path (default <run>/<ckpt-base>_eval.csv)")
    a = ap.parse_args()

    run_dir = a.run.rstrip("/")
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise SystemExit(f"eval_test: no config.json in {run_dir}")
    with open(cfg_path) as f:
        config = json.load(f)

    ckpt_path = _find_checkpoint(run_dir, a.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter_cls = _detect(config)
    print(f"eval_test: {adapter_cls.name} | {os.path.basename(ckpt_path)} | device={device.type}")
    adapter = adapter_cls(run_dir, config, ckpt_path, device)

    # Choose what to evaluate: an explicit id list (cross-model) or a split.
    if a.test_ids:
        with open(a.test_ids) as f:
            wanted = [ln.strip() for ln in f if ln.strip()]
        idx_of = adapter.id_to_index()
        indices = [idx_of[i] for i in wanted if i in idx_of]
        missing = [i for i in wanted if i not in idx_of]
        print(f"eval_test: {len(indices)}/{len(wanted)} requested ids found in this dataset")
        if missing:
            print(f"  WARNING: {len(missing)} ids not in dataset (e.g. {missing[:3]})")
        if not indices:
            raise SystemExit("eval_test: none of the requested ids are in this dataset.")
    else:
        indices = adapter.split_indices(a.split)
        print(f"eval_test: evaluating {len(indices)} samples from the '{a.split}' split")

    rows = adapter.predict(indices)

    out = a.out or os.path.join(run_dir,
                                os.path.basename(ckpt_path).replace(".pth.tar", "") + "_eval.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow(r)
    print(f"eval_test: wrote {len(rows)} predictions -> {out}")

    if not adapter.is_classification and rows:
        mae = sum(abs(t - p) for _, t, p in rows) / len(rows)
        print(f"eval_test: MAE = {mae:.4f} (target units)")

    if a.export_ids:
        with open(a.export_ids, "w") as f:
            f.write("\n".join(str(r[0]) for r in rows) + "\n")
        print(f"eval_test: exported {len(rows)} cif_ids -> {a.export_ids}")


if __name__ == "__main__":
    main()
