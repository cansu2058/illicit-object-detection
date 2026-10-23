# X-ray Baggage Object Detection with YOLOv8n

**Five-class prohibited-object detection in X-ray baggage images using Python, Ultralytics YOLOv8n, and the OPIXray dataset.**

This project covers the core stages of an object detection workflow: preparing annotations, fine-tuning a pretrained model, evaluating detection performance, and running inference on individual images. It was developed as an individual computer vision course project.

## At a glance

| Dataset | Model | Training images | Evaluation images | mAP@50 | mAP@50:95 |
|---|---|---:|---:|---:|---:|
| OPIXray | YOLOv8n | 7,109 | 1,776 | **0.862** | **0.405** |

- **Precision:** 0.859  
- **Recall:** 0.791  
- **Training environment:** Google Colab, NVIDIA Tesla T4 GPU  
- **Implementation:** Python and Ultralytics (PyTorch-based). Model training was managed through the Ultralytics API; no custom PyTorch training loop was implemented.

> **Evaluation note:** The original OPIXray test directory was converted into the YOLO `val` directory and used for the reported evaluation. Because this directory was also configured as validation data during model training, these results should **not** be described as performance on a separately held-out, untouched final test set.

## Problem and approach

X-ray baggage screening can involve overlapping objects, unusual appearance, and differences from ordinary RGB photographs. The goal of this project was to train a detector to localize and classify five categories of prohibited items in X-ray images.

The workflow was:

1. Read OPIXray images and original bounding-box annotations.
2. Convert bounding boxes to normalized YOLO labels.
3. Fine-tune pretrained `yolov8n.pt` weights on the converted dataset.
4. Evaluate overall and per-class detection metrics.
5. Run single-image inference with bounding boxes, class names, and confidence scores.

### Object classes

| ID | Class |
|---:|---|
| 0 | Folding Knife |
| 1 | Straight Knife |
| 2 | Scissor |
| 3 | Utility Knife |
| 4 | Multi-tool Knife |

## Results

### Overall detection metrics

| Metric | Reported value |
|---|---:|
| Precision | 0.859 |
| Recall | 0.791 |
| mAP@50 | 0.862 |
| mAP@50:95 | 0.405 |

### Per-class performance

| Class | Precision | Recall | mAP@50 | mAP@50:95 |
|---|---:|---:|---:|---:|
| Folding Knife | 0.859 | 0.822 | 0.895 | 0.422 |
| Straight Knife | 0.798 | 0.498 | 0.681 | 0.268 |
| Scissor | 0.962 | 0.970 | 0.979 | 0.517 |
| Utility Knife | 0.846 | 0.783 | 0.833 | 0.401 |
| Multi-tool Knife | 0.833 | 0.883 | 0.921 | 0.419 |

**Interpretation:** Scissor was the strongest class by mAP@50 (0.979). Straight Knife had the lowest recall (0.498), suggesting a substantial number of missed detections for this class. The difference between mAP@50 and mAP@50:95 also suggests room for improvement at stricter bounding-box overlap thresholds.

### Runtime measurements

The original experiment reported the following timing figures on its evaluation environment:

| Measurement | Reported value |
|---|---:|
| Inference time | 2.5 ms/image |
| End-to-end time | 5.2 ms/image |

These are experimental timing measurements, **not** a verified real-time application benchmark. They should not be interpreted as guaranteed throughput for a camera stream or another deployment environment.

### Qualitative outputs

The project notebook, [`Object_Detection.ipynb`](Object_Detection.ipynb), contains the experiment workflow and figures. Additional artifacts saved during the original run include `val_batch0_pred.jpg`, `val_batch0_labels.jpg`, `results.png`, and confusion-matrix images. These are not embedded here because they are not currently included as standalone images in this repository.

Once those actual output files have been added to the repository, a small gallery of correctly labeled predictions and failure cases can be included here.

## Dataset preparation

The original dataset and its usage terms are available from the [OPIXray project](https://github.com/OPIXray-author/OPIXray). Dataset download and result-folder references from the original coursework are listed in [`dataset_link.txt`](dataset_link.txt); access to external Drive files may vary.

Original annotation format:

```text
image_name class_name xmin ymin xmax ymax
```

Converted YOLO annotation format:

```text
class_id x_center y_center width height
```

The center coordinates, width, and height are normalized relative to image dimensions, with values in the range `[0, 1]`.

Expected converted dataset layout:

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

Example `data.yaml` (change `path` for your environment):

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

> **Split terminology:** Here, `val` is a **YOLO folder name** used for the original OPIXray test-directory images. Using that split for validation during fine-tuning means it is not an independent final test set. A future experiment should reserve a separate, untouched test split for final reporting.

## Training configuration

| Setting | Value |
|---|---|
| Starting weights | Pretrained `yolov8n.pt` |
| Epochs | 20 |
| Image size | 640 × 640 |
| Batch size | 8 |
| Platform | Google Colab |
| GPU | NVIDIA Tesla T4 |

Training used the Ultralytics `YOLO(...).train(...)` interface. Ultralytics relies on PyTorch internally; this project does not claim a custom neural-network architecture or manually implemented optimization loop.

## Installation and usage

### Requirements

Python 3.10 or later is recommended. A GPU is useful for training; single-image inference can also run on CPU.

```bash
pip install -r requirements.txt
```

The repository's [`requirements.txt`](requirements.txt) lists the packages used by the project. Package versions are not pinned, so reproducing results exactly may require matching the original environment.

The commands below assume you have obtained the OPIXray dataset and updated filesystem paths accordingly.

### 1. Prepare labels

```bash
python prepare_opixray_yolo.py \
  --root /path/to/OPIXray_full \
  --out /path/to/OPIXray_full_YOLO \
  --overwrite
```

This generates YOLO-format labels and the corresponding dataset layout. The script also provides Colab/Drive options; inspect its command-line arguments if running in another environment.

### 2. Train

```bash
python train_full_dataset.py \
  --data /path/to/OPIXray_full_YOLO/data.yaml \
  --epochs 20 \
  --imgsz 640 \
  --batch 8
```

Ultralytics saves training outputs under a `runs/detect/` directory. The best checkpoint from the original experiment is also included as [`best.pt`](best.pt).

### 3. Evaluate

```bash
python evaluate_full_dataset.py \
  --weights best.pt \
  --data /path/to/OPIXray_full_YOLO/data.yaml
```

This evaluates the checkpoint on the split referenced by the `val` field in `data.yaml`. Ensure that the evaluation setup matches the intended reporting protocol.

### 4. Run single-image inference

```bash
python inference_one_image.py \
  --weights best.pt \
  --image /path/to/sample_image.jpg
```

The inference script produces an annotated image with predicted bounding boxes, classes, and confidence scores. The original project saves output in a `runs/detect/` subdirectory.

### Google Colab note

If your files are in Google Drive, mount Drive in a notebook cell first:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then navigate to the repository and invoke scripts from a notebook with `!python`. The scripts support a `--no-mount` option for situations where Drive has already been mounted in the notebook. Avoid relying on a Drive mount initiated inside a separate subprocess.

## Repository contents

| File | Purpose |
|---|---|
| [`prepare_opixray_yolo.py`](prepare_opixray_yolo.py) | Convert OPIXray annotations into YOLO format |
| [`train_full_dataset.py`](train_full_dataset.py) | Fine-tune pretrained YOLOv8n weights |
| [`evaluate_full_dataset.py`](evaluate_full_dataset.py) | Evaluate trained weights |
| [`inference_one_image.py`](inference_one_image.py) | Run prediction on an individual X-ray image |
| [`Object_Detection.ipynb`](Object_Detection.ipynb) | Notebook with end-to-end coursework workflow and figures |
| [`best.pt`](best.pt) | Checkpoint from the final training run |
| [`requirements.txt`](requirements.txt) | Python dependencies |
| [`dataset_link.txt`](dataset_link.txt) | Dataset and results references |
| [`Karagoz_Cansu_Presentation.pptx`](Karagoz_Cansu_Presentation.pptx) | Project presentation |

## Limitations and next steps

- **Uneven class performance:** Straight Knife recall (0.498) was considerably lower than other categories. A targeted review of false negatives would help identify recurring failure patterns.
- **Stricter localization thresholds:** mAP@50:95 (0.405) was markedly lower than mAP@50 (0.862). Better bounding-box localization and targeted training experiments may help.
- **Evaluation protocol:** The original test directory was used as the YOLO validation split. A new, truly held-out test set is needed for an unbiased final assessment after model selection.
- **Generalization:** The reported results apply to the evaluated OPIXray images; performance on other X-ray systems, acquisition conditions, or object categories has not been established.
- **Deployment:** Real-time camera-stream throughput, memory consumption, robustness, and operational reliability have not been benchmarked.

Potential follow-up experiments include error analysis by class, augmentation experiments, comparisons between YOLO model sizes, a proper train/validation/test split, and an end-to-end inference demonstration.

## Acknowledgments

- [OPIXray dataset and project](https://github.com/OPIXray-author/OPIXray)
- [Ultralytics YOLO documentation](https://docs.ultralytics.com/)

**Author:** Cansu Karagöz  
**Project type:** Individual academic computer vision project
