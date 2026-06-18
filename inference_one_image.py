from pathlib import Path
import argparse


DEFAULT_DATASET_ROOT = Path("/content/drive/MyDrive/OPIXray_full")
DEFAULT_RESULTS_DIR = Path("/content/drive/MyDrive/opixray_full_yolov8n_20epochs_final")


def mount_drive_if_colab() -> None:
    try:
        from google.colab import drive  # type: ignore
    except Exception:
        return
    drive.mount("/content/drive", force_remount=False)


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def resolve_weights(weights_arg: str | None, results_dir: Path) -> Path:
    candidates = []
    if weights_arg:
        candidates.append(Path(weights_arg))
    candidates.extend(
        [
            results_dir / "weights" / "best.pt",
            DEFAULT_RESULTS_DIR / "weights" / "best.pt",
            Path("weights/best.pt"),
            Path("best.pt"),
        ]
    )

    weights = first_existing(candidates)
    if weights is None:
        raise FileNotFoundError(
            "Could not find best.pt. In Colab, mount Drive and make sure the final results folder is at "
            "/content/drive/MyDrive/opixray_full_yolov8n_20epochs_final, or pass --weights with the full path."
        )
    return weights


def find_default_image(dataset_root: Path) -> Path:
    candidate_dirs = [
        dataset_root / "test" / "test_image",
        dataset_root / "val" / "images",
        Path("/content/OPIXray_full_YOLO/images/val"),
        Path("/content/drive/MyDrive/OPIXray_full_YOLO/images/val"),
    ]
    for folder in candidate_dirs:
        if folder.exists():
            for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                image = next(folder.glob(pattern), None)
                if image is not None:
                    return image
    raise FileNotFoundError(
        "No default sample image found. Pass one explicitly with --image, for example "
        "--image /content/drive/MyDrive/OPIXray_full/test/test_image/000001.jpg"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the trained YOLOv8n model on one OPIXray image.")
    parser.add_argument("--weights", default=None, help="Optional full path to trained best.pt.")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="Drive folder containing weights/best.pt.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT), help="Original OPIXray dataset folder in Drive.")
    parser.add_argument("--image", default=None, help="Optional image path. If omitted, one test image is selected automatically.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--project", default="/content/runs/detect", help="Output directory root.")
    parser.add_argument("--name", default="single_image_prediction", help="Output run name.")
    parser.add_argument("--no-mount", action="store_true", help="Do not mount Google Drive automatically in Colab.")
    args = parser.parse_args()

    if not args.no_mount:
        mount_drive_if_colab()

    results_dir = Path(args.results_dir)
    dataset_root = Path(args.dataset_root)
    weights = resolve_weights(args.weights, results_dir)
    image_path = Path(args.image) if args.image else find_default_image(dataset_root)

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    from ultralytics import YOLO

    model = YOLO(str(weights))
    model.predict(
        source=str(image_path),
        conf=args.conf,
        save=True,
        project=args.project,
        name=args.name,
    )

    print("Single-image inference completed.")
    print(f"Input image: {image_path}")
    print(f"Model weights: {weights}")
    print(f"Prediction output: {Path(args.project) / args.name}")


if __name__ == "__main__":
    main()
