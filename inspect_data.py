from pathlib import Path
import builtins

root = Path("data")

if not root.exists():
    print("ERROR: The data folder does not exist.")
    raise SystemExit

for path in sorted(root.rglob("*")):
    if path.is_dir():
        image_count = sum(
            1 for file in path.iterdir()
            if file.is_file() and file.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".bmp", ".webp"
            }
        )

        if image_count > 0:
            print(f"{path}: {image_count} images")

    elif path.is_file() and path.suffix.lower() == ".zip":
        print(f"ZIP archive found: {path}")