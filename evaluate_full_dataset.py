from pathlib import Path
import argparse


DEFAULT_RESULTS_DIR = Path("/content/drive/MyDrive/opixray_full_yolov8n_20epochs_final")
DEFAULT_DATA_YAML = Path("/content/OPIXray_full_YOLO/data.yaml")


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
            "Could not find best.pt. Make sure the final results folder from Drive is available, "
            "or pass --weights with the full path."
        )
    return weights


def resolve_data_yaml(data_arg: str | None) -> Path:
    candidates = []
    if data_arg:
        candidates.append(Path(data_arg))
    candidates.extend(
        [
            DEFAULT_DATA_YAML,
            Path("/content/drive/MyDrive/OPIXray_full_YOLO/data.yaml"),
            Path("OPIXray_full_YOLO/data.yaml"),
        ]
    )
    data_yaml = first_existing(candidates)
    if data_yaml is None:
        raise FileNotFoundError(
            "Could not find data.yaml. Run prepare_opixray_yolo.py first, or pass --data with the full path."
        )
    return data_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained YOLOv8n model on the OPIXray validation split.")
    parser.add_argument("--weights", default=None, help="Optional full path to trained best.pt.")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="Drive folder containing weights/best.pt.")
    parser.add_argument("--data", default=None, help="Optional path to YOLO data.yaml.")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
    parser.add_argument("--batch", type=int, default=8, help="Validation batch size.")
    parser.add_argument("--no-mount", action="store_true", help="Do not mount Google Drive automatically in Colab.")
    args = parser.parse_args()

    if not args.no_mount:
        mount_drive_if_colab()

    weights = resolve_weights(args.weights, Path(args.results_dir))
    data_yaml = resolve_data_yaml(args.data)

    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(data=str(data_yaml), imgsz=args.imgsz, batch=args.batch)

    print("Full-dataset validation completed.")
    print(f"Model weights: {weights}")
    print(f"Dataset yaml: {data_yaml}")
    print(metrics)


if __name__ == "__main__":
    main()
