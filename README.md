# Illicit Object Detection in X-ray Baggage Images using YOLOv8n and OPIXray

**Course Project Practical Part**
**Author:** Cansu Karagoz

## 1. Project Description

This project implements and evaluates an object detection system for detecting prohibited knife/cutter-like objects in X-ray baggage images.

### Selected Method

* YOLOv8n using the Ultralytics framework

### Selected Dataset

* OPIXray full train/test split

### Detected Classes

| Class ID | Class Name       |
| -------- | ---------------- |
| 0        | Folding_Knife    |
| 1        | Straight_Knife   |
| 2        | Scissor          |
| 3        | Utility_Knife    |
| 4        | Multi-tool_Knife |

The model was trained and evaluated on the full OPIXray dataset converted to YOLO format.

### Final Experiment

| Item                        | Value        |
| --------------------------- | ------------ |
| Training images             | 7,109        |
| Validation/Test images      | 1,776        |
| Model                       | YOLOv8n      |
| Epochs                      | 20           |
| Image size                  | 640 x 640    |
| Batch size                  | 8            |
| Platform                    | Google Colab |
| GPU used for final training | Tesla T4     |

## 2. Files Included

The submission package contains:

| File                              | Description                                                                              |
| --------------------------------- | ---------------------------------------------------------------------------------------- |
| `Karagoz_Cansu_Presentation.pptx` | PowerPoint presentation of the practical project.                                        |
| `Object_Detection.ipynb`          | Main notebook containing the complete practical workflow, final results, and figures.    |
| `requirements.txt`                | Python dependencies required to run the scripts.                                         |
| `dataset_link.txt`                | Links to the original OPIXray dataset, the Drive dataset copy, and final results folder. |
| `best.pt`                         | Best trained YOLOv8n model weights.                                                      |
| `prepare_opixray_yolo.py`         | Script to convert original OPIXray annotations into YOLO format.                         |
| `train_full_dataset.py`           | Script to train YOLOv8n on the full OPIXray YOLO-format dataset.                         |
| `evaluate_full_dataset.py`        | Script to evaluate the trained model on the validation/test split.                       |
| `inference_one_image.py`          | Script to run the trained model on one image.                                            |

## 3. Installation Instructions

### Recommended Environment

* Google Colab
* Python 3.10 or later
* GPU runtime recommended for training and evaluation
* CPU runtime works for single-image inference and can run evaluation slowly
* Ultralytics YOLOv8

In Google Colab, enable GPU when available:

```text
Runtime -> Change runtime type -> Hardware accelerator -> GPU
```

To check GPU availability:

```bash
!nvidia-smi
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or in Google Colab:

```bash
!pip install -r requirements.txt
```

The `requirements.txt` file contains:

```text
ultralytics
opencv-python
pillow
matplotlib
pandas
pyyaml
```

If needed, Ultralytics can also be installed directly:

```bash
pip install ultralytics
```

To mount Google Drive in Colab:

```python
from google.colab import drive
drive.mount("/content/drive")
```

### Important Colab Note

The Python scripts include an optional Drive mount step, but Colab subprocesses cannot always mount Drive correctly.

For the most reliable run, mount Drive manually in a notebook cell first, change to the project folder, and add `--no-mount` to script commands:

```bash
%cd /content/drive/MyDrive/Project
!pip install -r requirements.txt
!python inference_one_image.py --no-mount
```

## 4. Dataset Information

### Original Dataset

OPIXray

### Original Dataset Link

[OPIXray GitHub Repository](https://github.com/OPIXray-author/OPIXray)

### Dataset Copy and Final Results Folder

See `dataset_link.txt`.

The original OPIXray annotation format is:

```text
image_name class_name xmin ymin xmax ymax
```

The dataset was converted to YOLO format:

```text
class_id x_center y_center width height
```

All bounding box coordinates are normalized between 0 and 1.

### Expected YOLO Dataset Structure

```text
OPIXray_full_YOLO/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

The `data.yaml` file should contain:

```yaml
path: /content/OPIXray_full_YOLO
train: images/train
val: images/val

names:
  0: Folding_Knife
  1: Straight_Knife
  2: Scissor
  3: Utility_Knife
  4: Multi-tool_Knife
```

If the dataset is stored in Google Drive, update the path, for example:

```yaml
path: /content/drive/MyDrive/OPIXray_full_YOLO
```

## 5. Prepare Dataset in YOLO Format

If starting from the original OPIXray dataset folders, run:

```bash
python prepare_opixray_yolo.py --root /content/drive/MyDrive/OPIXray_full --out /content/OPIXray_full_YOLO --overwrite
```

In Google Colab, after mounting Drive and changing to the project folder, use:

```bash
!python prepare_opixray_yolo.py --no-mount --overwrite
```

### Expected Original OPIXray Structure

```text
OPIXray_full/
├── train/
│   ├── train_image/
│   └── train_annotation/
└── test/
    ├── test_image/
    └── test_annotation/
```

The script creates:

```text
OPIXray_full_YOLO/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

## 6. Training on the Full Dataset

To train YOLOv8n on the full OPIXray dataset:

```bash
python train_full_dataset.py --data /content/OPIXray_full_YOLO/data.yaml --epochs 20 --imgsz 640 --batch 8
```

In Google Colab, if the dataset is stored in Drive, use:

```bash
!python train_full_dataset.py --no-mount --data /content/drive/MyDrive/OPIXray_full_YOLO/data.yaml --epochs 20 --imgsz 640 --batch 8
```

The training script uses:

| Parameter  | Value                      |
| ---------- | -------------------------- |
| Model      | YOLOv8n pretrained weights |
| Epochs     | 20                         |
| Image size | 640                        |
| Batch size | 8                          |

The trained model is saved by Ultralytics in a `runs/detect/` folder.

The best model is saved as:

```text
weights/best.pt
```

> Note: Training on CPU is possible but not recommended because it can take a very long time. The submitted final model is already included as `best.pt`.

## 7. Evaluation on the Full Dataset

To evaluate the trained model on the validation/test set:

```bash
python evaluate_full_dataset.py --weights best.pt --data /content/OPIXray_full_YOLO/data.yaml
```

If using files stored in Google Drive from Colab:

```bash
!python evaluate_full_dataset.py --no-mount --weights /content/drive/MyDrive/opixray_full_yolov8n_20epochs_final/weights/best.pt --data /content/drive/MyDrive/OPIXray_full_YOLO/data.yaml
```

If the default Drive folders are already used, this shorter command is enough:

```bash
!python evaluate_full_dataset.py --no-mount
```

The evaluation reports:

* Precision
* Recall
* mAP@50
* mAP@50:95
* Inference speed

## 8. Single-Image Inference

To run the trained model on one image:

```bash
python inference_one_image.py --weights best.pt --image sample_image.jpg
```

Example using Google Drive paths from Colab:

```bash
!python inference_one_image.py --no-mount --weights /content/drive/MyDrive/opixray_full_yolov8n_20epochs_final/weights/best.pt --image /content/drive/MyDrive/OPIXray_full/test/test_image/example.jpg
```

If the default Drive folders are already used, this shorter command is enough and automatically selects one test image:

```bash
!python inference_one_image.py --no-mount
```

If no image path is provided, the script can automatically select one image from the available OPIXray test/validation folder, if the dataset folder exists.

The output prediction image is saved by Ultralytics in:

```text
runs/detect/single_image_prediction/
```

The output image contains the detected object bounding boxes, class labels, and confidence scores.

## 9. Final Results

The final YOLOv8n model achieved the following results on the full OPIXray validation/test split:

### Overall Results

| Metric          |        Value |
| --------------- | -----------: |
| Precision       |        0.859 |
| Recall          |        0.791 |
| mAP@50          |        0.862 |
| mAP@50:95       |        0.405 |
| Inference time  | 2.5 ms/image |
| End-to-end time | 5.2 ms/image |
| Inference FPS   |      400 FPS |
| End-to-end FPS  |   192.31 FPS |

### Class-Level Results

| Class            | Precision | Recall | mAP@50 | mAP@50:95 |
| ---------------- | --------: | -----: | -----: | --------: |
| Folding_Knife    |     0.859 |  0.822 |  0.895 |     0.422 |
| Straight_Knife   |     0.798 |  0.498 |  0.681 |     0.268 |
| Scissor          |     0.962 |  0.970 |  0.979 |     0.517 |
| Utility_Knife    |     0.846 |  0.783 |  0.833 |     0.401 |
| Multi-tool_Knife |     0.833 |  0.883 |  0.921 |     0.419 |

## 10. Saved Results

The final results folder contains:

```text
results.csv
class_level_results.csv
results.png
confusion_matrix.png
confusion_matrix_normalized.png
BoxPR_curve.png
BoxF1_curve.png
BoxP_curve.png
BoxR_curve.png
val_batch0_pred.jpg
val_batch0_labels.jpg
weights/best.pt
```

These files are used in the PowerPoint presentation to show training curves, confusion matrix, qualitative predictions, and final metrics.

## 11. Notes

A small subset was first used to verify that the dataset conversion, training, and evaluation pipeline worked correctly. After confirming the pipeline, the final experiment was performed using the full OPIXray train/test split.

The final model and results were saved to Google Drive to avoid losing files after the Colab runtime disconnected.

Files stored only in `/content/` are temporary in Google Colab. For reproducibility, important files such as `best.pt`, result figures, and CSV metric files should be saved in Google Drive or included in the final submission ZIP.
