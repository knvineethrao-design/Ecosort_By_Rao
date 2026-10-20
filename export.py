from pathlib import Path
import argparse
import torch

from model import WasteClassifier


def main():
    parser = argparse.ArgumentParser(description="Export the best waste classifier checkpoint to ONNX.")
    parser.add_argument("--checkpoint", default="outputs/best_model.pth")
    parser.add_argument("--output", default="outputs/waste_classifier.onnx")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\nTrain first with: python train.py"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = WasteClassifier(num_classes=6)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    if params >= 350_000:
        raise RuntimeError(f"Parameter limit exceeded: {params:,}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, 96, 96)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=12,
        do_constant_folding=True,
    )

    size_bytes = output_path.stat().st_size
    size_mb = size_bytes / 1_000_000
    print("Exported:", output_path.resolve())
    print(f"Parameters: {params:,}")
    print(f"ONNX size: {size_mb:.3f} MB ({size_bytes:,} bytes)")
    print("Meets parameter limit:", params < 350_000)
    print("Meets 1.50 MB file limit:", size_mb <= 1.50)
    if size_mb > 1.50:
        print("WARNING: ONNX file exceeds the 1.50 MB limit.")


if __name__ == "__main__":
    main()
