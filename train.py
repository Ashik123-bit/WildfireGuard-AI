
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.data.prepare_dataset import prepare_dataset


def main() -> None:
    project_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Train the WildfireGuard fire/smoke detector."
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=416)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--device", default="")

    args = parser.parse_args()

    if args.epochs < 1 or args.image_size < 32 or args.batch < 1:
        parser.error("Training parameters must be positive.")

    dataset_yaml, train_count, val_count = prepare_dataset(
        project_root, project_root / "data" / "processed"
    )

    print(f"Training images: {train_count}")
    print(f"Validation images: {val_count}")
    print(f"Starting weights: {args.weights}")

    from ultralytics import YOLO

    model = YOLO(args.weights)

    model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch,
        device=args.device or None,
        workers=0,
        project=str(project_root / "outputs"),
        name="wildfire_v2",
        exist_ok=True,
        patience=15,
        save=True,
        plots=True,
    )

    trained_weights = (
        project_root / "outputs" / "wildfire_v2" / "weights" / "best.pt"
    )

    if not trained_weights.is_file():
        raise FileNotFoundError(
            f"Expected model weights not found: {trained_weights}"
        )

    destination = project_root / "models" / "wildfire_yolo_v2.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(trained_weights, destination)

    print(f"New model saved to: {destination}")


if __name__ == "__main__":
    main()
