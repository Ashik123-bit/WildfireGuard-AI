
from pathlib import Path
from ultralytics import YOLO


def main():
    project_root = Path(__file__).resolve().parent

    dataset_yaml = project_root / "data" / "processed" / "data.yaml"

    print("Starting WildfireGuard 512 experiment")
    print("Image size: 512")
    print("Batch size: 1")
    print("Epochs: 30")

    model = YOLO("yolo11n.pt")

    model.train(
        data=str(dataset_yaml),
        epochs=30,
        imgsz=512,
        batch=1,
        device="cpu",
        workers=0,
        project=str(project_root / "outputs"),
        name="wildfire_512",
        exist_ok=False,
        patience=10,
        save=True,
        plots=True,
    )

    best_weights = (
        project_root / "outputs" / "wildfire_512" / "weights" / "best.pt"
    )

    if best_weights.is_file():
        print(f"\nTraining completed!")
        print(f"Best model: {best_weights}")
    else:
        print("\nTraining finished, but best.pt was not found.")


if __name__ == "__main__":
    main()

