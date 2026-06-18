Illicit Object Detection in X-ray Baggage Images using YOLOv8n and OPIXray
Course Project Practical Part
Author: Cansu Karagoz

============================================================
1. PROJECT DESCRIPTION
============================================================

This project implements and evaluates an object detection system for detecting prohibited knife/cutter-like objects in X-ray baggage images.

Selected method:
YOLOv8n using the Ultralytics framework

Selected dataset:
OPIXray full train/test split

Detected classes:
0: Folding_Knife
1: Straight_Knife
2: Scissor
3: Utility_Knife
4: Multi-tool_Knife

The model was trained and evaluated on the full OPIXray dataset converted to YOLO format.

Final experiment:
Training images: 7,109
Validation/Test images: 1,776
Model: YOLOv8n
Epochs: 20
Image size: 640 x 640
Batch size: 8
Platform: Google Colab
GPU used for final training: Tesla T4

============================================================
2. FILES INCLUDED
============================================================

The submission package contains:

Karagoz_Cansu_Presentation.pptx
    PowerPoint presentation of the practical project.

Object_Detection.ipynb
    Main notebook containing the complete practical workflow, final results, and figures.

requirements.txt
    Python dependencies required to run the scripts.

dataset_link.txt
    Links to the original OPIXray dataset, the Drive dataset copy, and final results folder.

best.pt
    Best trained YOLOv8n model weights.

prepare_opixray_yolo.py
    Script to convert original OPIXray annotations into YOLO format.

train_full_dataset.py
    Script to train YOLOv8n on the full OPIXray YOLO-format dataset.

evaluate_full_dataset.py
    Script to evaluate the trained model on the validation/test split.

inference_one_image.py
    Script to run the trained model on one image.

============================================================
3. INSTALLATION INSTRUCTIONS
============================================================

Recommended environment:
- Google Colab
- Python 3.10 or later
- GPU runtime recommended for training and evaluation
- CPU runtime works for single-image inference and can run evaluation slowly
- Ultralytics YOLOv8

In Google Colab, enable GPU when available:

Runtime -> Change runtime type -> Hardware accelerator -> GPU

To check GPU availability:

!nvidia-smi

Install the required dependencies:

pip install -r requirements.txt

or in Google Colab:

!pip install -r requirements.txt

The requirements.txt file contains:

ultralytics
opencv-python
pillow
matplotlib
pandas
pyyaml

If needed, Ultralytics can also be installed directly:

pip install ultralytics

To mount Google Drive in Colab:

from google.colab import drive
drive.mount("/content/drive")

Important Colab note:
The Python scripts include an optional Drive mount step, but Colab subprocesses cannot always mount Drive correctly. For the most reliable run, mount Drive manually in a notebook cell first, change to the Project folder, and add --no-mount to script commands:

%cd /content/drive/MyDrive/Project
!pip install -r requirements.txt
!python inference_one_image.py --no-mount

============================================================
4. DATASET INFORMATION
============================================================

Original dataset:
OPIXray

Original dataset link:
https://github.com/OPIXray-author/OPIXray

Dataset copy and final run/results folder:
See dataset_link.txt

The original OPIXray annotation format is:

image_name class_name xmin ymin xmax ymax

The dataset was converted to YOLO format:

class_id x_center y_center width height

All bounding box coordinates are normalized between 0 and 1.

Expected YOLO dataset structure:

OPIXray_full_YOLO/
    images/
        train/
        val/
    labels/
        train/
        val/
    data.yaml

The data.yaml file should contain:

path: /content/OPIXray_full_YOLO
train: images/train
val: images/val

names:
  0: Folding_Knife
  1: Straight_Knife
  2: Scissor
  3: Utility_Knife
  4: Multi-tool_Knife

If the dataset is stored in Google Drive, update the path, for example:

path: /content/drive/MyDrive/OPIXray_full_YOLO

============================================================
5. PREPARE DATASET IN YOLO FORMAT
============================================================

If starting from the original OPIXray dataset folders, run:

python prepare_opixray_yolo.py --root /content/drive/MyDrive/OPIXray_full --out /content/OPIXray_full_YOLO --overwrite

In Google Colab, after mounting Drive and changing to the Project folder, use:

!python prepare_opixray_yolo.py --no-mount --overwrite

Expected original OPIXray structure:

OPIXray_full/
    train/
        train_image/
        train_annotation/
    test/
        test_image/
        test_annotation/

The script creates:

OPIXray_full_YOLO/
    images/train/
    images/val/
    labels/train/
    labels/val/
    data.yaml

============================================================
6. TRAINING ON THE FULL DATASET
============================================================

To train YOLOv8n on the full OPIXray dataset:

python train_full_dataset.py --data /content/OPIXray_full_YOLO/data.yaml --epochs 20 --imgsz 640 --batch 8

In Google Colab, if the dataset is stored in Drive, use:

!python train_full_dataset.py --no-mount --data /content/drive/MyDrive/OPIXray_full_YOLO/data.yaml --epochs 20 --imgsz 640 --batch 8

The training script uses:

Model: YOLOv8n pretrained weights
Epochs: 20
Image size: 640
Batch size: 8

The trained model is saved by Ultralytics in a runs/detect/ folder.
The best model is saved as:

weights/best.pt

Note: Training on CPU is possible but not recommended because it can take a very long time. The submitted final model is already included as best.pt.

============================================================
7. EVALUATION ON THE FULL DATASET
============================================================

To evaluate the trained model on the validation/test set:

python evaluate_full_dataset.py --weights best.pt --data /content/OPIXray_full_YOLO/data.yaml

If using files stored in Google Drive from Colab:

!python evaluate_full_dataset.py --no-mount --weights /content/drive/MyDrive/opixray_full_yolov8n_20epochs_final/weights/best.pt --data /content/drive/MyDrive/OPIXray_full_YOLO/data.yaml

If the default Drive folders are already used, this shorter command is enough:

!python evaluate_full_dataset.py --no-mount

The evaluation reports:

Precision
Recall
mAP@50
mAP@50:95
Inference speed

============================================================
8. SINGLE-IMAGE INFERENCE
============================================================

To run the trained model on one image:

python inference_one_image.py --weights best.pt --image sample_image.jpg

Example using Google Drive paths from Colab:

!python inference_one_image.py --no-mount --weights /content/drive/MyDrive/opixray_full_yolov8n_20epochs_final/weights/best.pt --image /content/drive/MyDrive/OPIXray_full/test/test_image/example.jpg

If the default Drive folders are already used, this shorter command is enough and automatically selects one test image:

!python inference_one_image.py --no-mount

If no image path is provided, the script can automatically select one image from the available OPIXray test/validation folder, if the dataset folder exists.

The output prediction image is saved by Ultralytics in:

runs/detect/single_image_prediction/

The output image contains the detected object bounding boxes, class labels, and confidence scores.

============================================================
9. FINAL RESULTS
============================================================

The final YOLOv8n model achieved the following results on the full OPIXray validation/test split:

Overall results:
Precision: 0.859
Recall: 0.791
mAP@50: 0.862
mAP@50:95: 0.405
Inference time: 2.5 ms/image
End-to-end time: 5.2 ms/image
Inference FPS: 400 FPS
End-to-end FPS: 192.31 FPS

Class-level results:

Folding_Knife:
Precision 0.859, Recall 0.822, mAP@50 0.895, mAP@50:95 0.422

Straight_Knife:
Precision 0.798, Recall 0.498, mAP@50 0.681, mAP@50:95 0.268

Scissor:
Precision 0.962, Recall 0.970, mAP@50 0.979, mAP@50:95 0.517

Utility_Knife:
Precision 0.846, Recall 0.783, mAP@50 0.833, mAP@50:95 0.401

Multi-tool_Knife:
Precision 0.833, Recall 0.883, mAP@50 0.921, mAP@50:95 0.419

============================================================
10. SAVED RESULTS
============================================================

The final results folder contains:

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

These files are used in the PowerPoint presentation to show training curves, confusion matrix, qualitative predictions, and final metrics.

============================================================
11. NOTES
============================================================

A small subset was first used to verify that the dataset conversion, training, and evaluation pipeline worked correctly. After confirming the pipeline, the final experiment was performed using the full OPIXray train/test split.

The final model and results were saved to Google Drive to avoid losing files after the Colab runtime disconnected.

Files stored only in /content/ are temporary in Google Colab. For reproducibility, important files such as best.pt, result figures, and CSV metric files should be saved in Google Drive or included in the final submission ZIP.
