from pathlib import Path
import argparse
import shutil


DEFAULT_DATA_YAML = Path("/content/OPIXray_full_YOLO/data.yaml")
DEFAULT_DRIVE_OUTPUT = Path("/content/drive/MyDrive/opixray_full_yolov8n_20epochs_final")


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
    parser = argparse.ArgumentParser(description="Train YOLOv8n on the full OPIXray YOLO-format dataset.")
    parser.add_argument("--data", default=None, help="Optional path to YOLO data.yaml.")
    parser.add_argument("--model", default="yolov8n.pt", help="Base YOLO model.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--name", default="opixray_full_yolov8n_20epochs", help="Ultralytics run name.")
    parser.add_argument("--save-to-drive", action="store_true", help="Copy the run folder to Google Drive after training.")
    parser.add_argument("--drive-output", default=str(DEFAULT_DRIVE_OUTPUT), help="Drive destination for final results.")
    parser.add_argument("--no-mount", action="store_true", help="Do not mount Google Drive automatically in Colab.")
    args = parser.parse_args()

    if not args.no_mount:
        mount_drive_if_colab()

    data_yaml = resolve_data_yaml(args.data)
    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
    )

    print("Full-dataset training completed.")
    print(f"Dataset yaml: {data_yaml}")

    save_dir = Path(getattr(results, "save_dir", "")) if results is not None else Path()
    if args.save_to_drive:
        if not save_dir.exists():
            raise FileNotFoundError(f"Training output folder not found: {save_dir}")
        destination = Path(args.drive_output)
        shutil.copytree(save_dir, destination, dirs_exist_ok=True)
        print(f"Copied final run to Drive: {destination}")
    elif save_dir.exists():
        print(f"Training output folder: {save_dir}")


if __name__ == "__main__":
    main()
