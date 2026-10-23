"""
DEFERRED (future hardware phase). Exports the trained intent model to ONNX so it can
later be converted to TensorRT (FP16/INT8) on a Jetson. Included now only so the
architecture is export-ready; do NOT treat this as part of the simulation deliverable.

    python runtime/export_onnx.py --config config/config.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    import torch
    from intent.model import build_model

    wd = cfg["paths"]["weights"]
    meta = json.load(open(f"{wd}/intent_meta.json"))
    in_dim = meta["in_dim"]; seq_len = meta["seq_length"]
    model = build_model(in_dim, cfg)
    model.load_state_dict(torch.load(f"{wd}/intent_model.pth", map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, seq_len, in_dim)
    out_path = f"{wd}/intent_model.onnx"
    torch.onnx.export(model, dummy, out_path,
                      input_names=["seq"], output_names=["logit"],
                      dynamic_axes={"seq": {0: "batch"}}, opset_version=17)
    print(f"Exported -> {out_path}")
    print("Next (on Jetson): trtexec --onnx=intent_model.onnx --fp16 "
          "--saveEngine=intent_model.engine")


if __name__ == "__main__":
    main()
