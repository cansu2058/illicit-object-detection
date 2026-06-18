from pathlib import Path
import argparse
import shutil


CLASS_NAMES = [
    "Folding_Knife",
    "Straight_Knife",
    "Scissor",
    "Utility_Knife",
    "Multi-tool_Knife",
]


def mount_drive_if_colab() -> None:
    try:
        from google.colab import drive  # type: ignore
    except Exception:
        return
    drive.mount("/content/drive", force_remount=False)


def find_image(image_dir: Path, stem_or_name: str) -> Path | None:
    candidate = image_dir / stem_or_name
    if candidate.exists():
        return candidate

    stem = Path(stem_or_name).stem
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def convert_split(split_name: str, image_dir: Path, annotation_dir: Path, output_dir: Path, class_to_id: dict[str, int]) -> tuple[int, int]:
    out_images = output_dir / "images" / split_name
    out_labels = output_dir / "labels" / split_name
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0

    for index, annotation_path in enumerate(sorted(annotation_dir.glob("*.txt")), 1):
        lines = [line.strip() for line in annotation_path.read_text(errors="ignore").splitlines() if line.strip()]
        if not lines:
            skipped += 1
            continue

        first_parts = lines[0].split()
        image_name = first_parts[0] if len(first_parts) >= 6 else annotation_path.stem
        image_path = find_image(image_dir, image_name)
        if image_path is None:
            image_path = find_image(image_dir, annotation_path.stem)
        if image_path is None:
            print(f"[WARN] Missing image for annotation: {annotation_path.name}")
            skipped += 1
            continue

        from PIL import Image

        with Image.open(image_path) as image:
            width, height = image.size

        yolo_lines = []
        for line in lines:
            parts = line.split()
            if len(parts) < 6:
                continue

            class_name = parts[1]
            if class_name not in class_to_id:
                print(f"[WARN] Unknown class '{class_name}' in {annotation_path.name}")
                continue

            xmin, ymin, xmax, ymax = map(float, parts[2:6])
            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height
            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height

            yolo_lines.append(
                f"{class_to_id[class_name]} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
            )

        if not yolo_lines:
            skipped += 1
            continue

        target_image = out_images / image_path.name
        target_label = out_labels / f"{image_path.stem}.txt"
        shutil.copy2(image_path, target_image)
        target_label.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
        copied += 1

        if index % 500 == 0:
            print(f"{split_name}: processed {index}")

    return copied, skipped


def write_data_yaml(output_dir: Path) -> None:
    lines = [
        f"path: {output_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    (output_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the original OPIXray annotations to YOLO format.")
    parser.add_argument("--root", default="/content/drive/MyDrive/OPIXray_full", help="Original OPIXray folder.")
    parser.add_argument("--out", default="/content/OPIXray_full_YOLO", help="Output YOLO-format dataset folder.")
    parser.add_argument("--overwrite", action="store_true", help="Delete the output folder before conversion.")
    parser.add_argument("--no-mount", action="store_true", help="Do not mount Google Drive automatically in Colab.")
    args = parser.parse_args()

    if not args.no_mount:
        mount_drive_if_colab()

    root = Path(args.root)
    output_dir = Path(args.out)

    train_img_dir = root / "train" / "train_image"
    train_ann_dir = root / "train" / "train_annotation"
    test_img_dir = root / "test" / "test_image"
    test_ann_dir = root / "test" / "test_annotation"

    for required in [train_img_dir, train_ann_dir, test_img_dir, test_ann_dir]:
        if not required.exists():
            raise FileNotFoundError(f"Missing required OPIXray folder: {required}")

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)

    class_to_id = {name: index for index, name in enumerate(CLASS_NAMES)}
    train_copied, train_skipped = convert_split("train", train_img_dir, train_ann_dir, output_dir, class_to_id)
    val_copied, val_skipped = convert_split("val", test_img_dir, test_ann_dir, output_dir, class_to_id)
    write_data_yaml(output_dir)

    print("Conversion completed.")
    print(f"Train copied: {train_copied}, skipped: {train_skipped}")
    print(f"Val copied: {val_copied}, skipped: {val_skipped}")
    print(f"data.yaml: {output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
