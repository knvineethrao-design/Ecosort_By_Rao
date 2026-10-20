from pathlib import Path
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Verify ONNX Runtime can load and run the exported model.")
    parser.add_argument("--model", default="outputs/waste_classifier.onnx")
    parser.add_argument("--data", default=None, help="Optional dataset folder to measure ONNX accuracy")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}. Run python export.py first.")

    import onnx
    import onnxruntime as ort

    graph = onnx.load(str(model_path))
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    sample = np.zeros((1, 3, 96, 96), dtype=np.float32)
    output = session.run([output_meta.name], {input_meta.name: sample})[0]

    print("ONNX checker: passed")
    print("Input name:", input_meta.name)
    print("Input shape:", input_meta.shape)
    print("Output name:", output_meta.name)
    print("Output shape:", output.shape)
    print("Output finite:", bool(np.isfinite(output).all()))
    print("File size (MB):", round(model_path.stat().st_size / 1_000_000, 4))
    if output.shape != (1, 6):
        raise RuntimeError(f"Unexpected output shape: {output.shape}; expected (1, 6)")
    if not np.isfinite(output).all():
        raise RuntimeError("ONNX output contains NaN or infinite values.")
    # Scale-invariance check: evaluator may send 0-1, 0-255 or normalized pixels.
    rng = np.random.default_rng(0)
    x = rng.random((4, 3, 96, 96), dtype=np.float32)
    a = session.run(None, {input_meta.name: x})[0]
    b = session.run(None, {input_meta.name: x * 255})[0]
    print("Same predictions for 0-1 and 0-255 input:", bool((a.argmax(1) == b.argmax(1)).all()))

    # Optional: accuracy of the ONNX model on a dataset folder (class subfolders)
    if args.data:
        from PIL import Image
        names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        ok = n = 0
        for i, c in enumerate(names):
            files = sorted(
                f for f in (Path(args.data) / c).rglob("*")
                if f.is_file() and f.suffix.lower() in extensions
            )
            for f in files:
                im = Image.open(f).convert("RGB").resize((96, 96), Image.Resampling.BILINEAR)
                im = np.asarray(im, dtype=np.float32) / 255.0
                im = (im - mean) / std
                p = session.run(None, {input_meta.name: im.transpose(2, 0, 1)[None]})[0].argmax(1)[0]
                ok += int(p == i); n += 1
        if n == 0:
            raise RuntimeError(f"No supported images found under class folders in {args.data}.")
        print(f"ONNX accuracy on {args.data}: {ok}/{n} = {ok/n:.2%}")
    print("ONNX Runtime inference: passed")


if __name__ == "__main__":
    main()
