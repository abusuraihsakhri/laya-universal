"""Export a Laya checkpoint to ONNX for cross-platform inference.

Requires PyTorch and the upstream `laya` package once, on any machine (a CPU
build of torch is fine). The exported directory then runs everywhere with only
onnxruntime — no PyTorch at inference time.

    pip install "laya-universal[export]"
    laya-universal convert --model convaiinnovations/laya --output ./laya-onnx
    laya-universal predict --model ./laya-onnx --state "..." --questions q.json

Writes:
    model.onnx       full decision model (inputs: input_ids, attention_mask,
                     marker_pos, marker_mask, qtype; outputs: logits, act_logits)
    encoder.onnx     encoder-only graph (output: hidden_states) enabling
                     embed_fn_from_agent / predict_shortlist on ONNX backends
    rl_agent_config.json, encoder/config.json, tokenizer/
"""

import argparse
import shutil
import sys
from pathlib import Path


def _load_upstream_agent(model_id_or_path):
    try:
        import torch
    except ImportError:
        sys.exit(
            "PyTorch is required for the export itself (the exported model does not "
            "need it).\n  pip install torch"
        )
    try:
        from laya.agent import Agent
    except ImportError:
        sys.exit("The upstream `laya` package is required for export.\n  pip install laya")
    print(f"Loading PyTorch agent from {model_id_or_path} ...")
    return Agent(model_id_or_path, device="cpu"), torch


def _source_dir(agent, model_id_or_path):
    """Directory holding config/tokenizer, without trusting upstream Agent internals."""
    src = getattr(agent, "model_dir", None)
    if src and (Path(src) / "rl_agent_config.json").is_file():
        return Path(src)
    value = str(model_id_or_path)
    if Path(value).exists():
        return Path(value)
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            value,
            allow_patterns=["rl_agent_config.json", "encoder/config.json", "tokenizer/*"],
        )
    )


def export_to_onnx(model_id_or_path: str, output_path: str, dtype: str = "float32"):
    """Export a Laya checkpoint to an ONNX directory.

    dtype='float32' is the safe default: the CPU execution provider has limited
    fp16 support. fp16 targets GPU providers (DirectML/CUDA) and halves the file.
    """
    agent, torch = _load_upstream_agent(model_id_or_path)
    model = agent.model
    model.eval()
    if dtype == "float16":
        model = model.half()

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = (
        torch.randint(0, 100, (1, 16), dtype=torch.long),
        torch.ones((1, 16), dtype=torch.long),
        torch.tensor([[1, 5]], dtype=torch.long),
        torch.tensor([[True, True]], dtype=torch.bool),
        torch.tensor([0], dtype=torch.long),
    )
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "marker_pos": {0: "batch_size", 1: "num_markers"},
        "marker_mask": {0: "batch_size", 1: "num_markers"},
        "qtype": {0: "batch_size"},
        "logits": {0: "batch_size", 1: "num_markers"},
        "act_logits": {0: "batch_size"},
    }

    print(f"Exporting model.onnx ({dtype}) ...")
    torch.onnx.export(
        model,
        inputs,
        str(out_dir / "model.onnx"),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
        output_names=["logits", "act_logits"],
        dynamic_axes=dynamic_axes,
    )

    encoder = getattr(model, "encoder", None)
    if encoder is not None:

        class _EncoderWrapper(torch.nn.Module):
            def __init__(self, enc):
                super().__init__()
                self.encoder = enc

            def forward(self, input_ids, attention_mask):
                return self.encoder(input_ids, attention_mask)

        print("Exporting encoder.onnx ...")
        torch.onnx.export(
            _EncoderWrapper(encoder),
            (inputs[0], inputs[1]),
            str(out_dir / "encoder.onnx"),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["hidden_states"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "seq_len"},
                "attention_mask": {0: "batch_size", 1: "seq_len"},
                "hidden_states": {0: "batch_size", 1: "seq_len"},
            },
        )
    else:
        print("model has no .encoder attribute; skipping encoder.onnx")

    src = _source_dir(agent, model_id_or_path)
    shutil.copy2(src / "rl_agent_config.json", out_dir / "rl_agent_config.json")
    (out_dir / "encoder").mkdir(exist_ok=True)
    shutil.copy2(src / "encoder" / "config.json", out_dir / "encoder" / "config.json")
    tok_out = out_dir / "tokenizer"
    if tok_out.exists():
        shutil.rmtree(tok_out)
    shutil.copytree(src / "tokenizer", tok_out)

    print(f"Exported to {out_dir}:")
    for p in sorted(out_dir.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(out_dir)}")
    print(f"\nLoad it with: laya.load({str(out_dir)!r})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a Laya model to ONNX format")
    parser.add_argument("--model", type=str, default="convaiinnovations/laya",
                        help="HuggingFace Hub ID or local path")
    parser.add_argument("--output", type=str, default="./laya-onnx",
                        help="Output directory for the ONNX model")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16"],
                        help="Export precision (fp32 for CPU providers, fp16 for GPU)")
    args = parser.parse_args()
    export_to_onnx(args.model, args.output, args.dtype)
