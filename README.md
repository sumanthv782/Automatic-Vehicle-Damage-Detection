# Automatic Vehicle Damage Detection

Multi-class object detection for automotive damage assessment, comparing three modeling approaches — a ResNet-based classifier (**DamageNet**), **Mask R-CNN** (via Detectron2), and **YOLOv8** — trained on the VehiDE dataset of annotated vehicle images.

## Overview

Given a photo of a vehicle, the goal is to localize and classify different types of visible damage (dents, scratches, broken glass, etc.). The repo explores this from three angles:

1. **DamageNet** — a ResNet50-based multi-label classifier fine-tuned to predict which damage types are present in an image.
2. **Mask R-CNN** — an instance segmentation model (via Detectron2) that draws pixel-level masks around each damaged region.
3. **YOLOv8** — a real-time object detector trained on bounding-box annotations converted from the dataset's polygon labels.

## Dataset

[VehiDE: Vehicle Damage Detection Dataset](https://www.kaggle.com/datasets/hendrichscullen/vehide-dataset-automatic-vehicle-damage-detection) (Kaggle)

Annotations are provided as polygons (VIA format) and are converted in-notebook to COCO-style bounding boxes/masks and to YOLO's normalized `.txt` label format.

### Damage classes

| Class | Meaning |
|---|---|
| `mat_bo_phan` | Missing part |
| `rach` | Tear |
| `mop_lom` | Dent |
| `tray_son` | Scratched paint |
| `thung` | Hole |
| `vo_kinh` | Broken glass |
| `be_den` | Broken light |

A `no_damage` class is added separately in `DamageNet_with_nodamage_class.ipynb` for images with no visible damage.

## Repository Contents

| File | Description |
|---|---|
| `automatic-vehicle-damage-detection.ipynb` | Main notebook: converts VIA polygon annotations to COCO/YOLO formats, trains DamageNet (ResNet50 classifier) and YOLOv8n, and visualizes predictions. |
| `Mask-R-CNN.ipynb` | Installs Detectron2, converts annotations to COCO format, registers the dataset, and trains/evaluates a Mask R-CNN model with COCO-style evaluation. |
| `DamageNet.ipynb` | Standalone DamageNet (ResNet50) training notebook using a multi-label `BCEWithLogitsLoss` setup. |
| `DamageNet_with_nodamage_class.ipynb` | Extends DamageNet with an explicit `no_damage` class and a focal-loss objective to handle class imbalance, plus per-class accuracy reporting. |
| `YOLOv8_colab.ipynb` | Google Colab version of the YOLO annotation conversion and YOLOv8 training/inference pipeline. |
| `train_coco.json`, `val_coco.json` | COCO-format annotations generated from the original VIA polygon annotations. |
| `0Train_no_damage_via_annos.json`, `0Val_no_damage_via_annos.json` | VIA-format annotations for the no-damage image subset. |
| `predictions/` | Sample test images used for running inference with the trained models. |
| `Proposal-Automatic-Vehicle-Damage-Detection.docx` | Original project proposal document. |

## Approach

### Annotation preprocessing
The raw VehiDE annotations are polygon coordinates (`x_all`, `y_all`) in VIA format. Each notebook includes conversion utilities to turn these into:
- **COCO bounding boxes** `(x, y, w, h)` for Mask R-CNN / classifier training
- **YOLO normalized `.txt` labels** (`class x_center y_center width height`) for YOLOv8 training

### DamageNet (classification)
- Backbone: `torchvision.models.resnet50`, final layer replaced with a linear layer sized to the number of damage classes.
- Multi-label formulation with `BCEWithLogitsLoss` (later versions use a custom `FocalLoss` to address class imbalance).
- Supports both GPU and TPU (`torch_xla`) training paths.
- Includes inference/visualization helpers that overlay predicted labels on test images.

### Mask R-CNN (instance segmentation)
- Built on [Detectron2](https://github.com/facebookresearch/detectron2).
- Dataset registered via `register_coco_instances` using the generated COCO annotations.
- Trained with Detectron2's `DefaultTrainer` and evaluated with `COCOEvaluator`.

### YOLOv8 (object detection)
- Uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (`ultralytics` package).
- Images and YOLO-format labels are organized into `yolo_dataset/images/{train,validation}` and `yolo_dataset/labels/{train,validation}`.
- A `dataset.yaml` is generated to describe the class names and splits before training.
- Includes a batch-inference script that loads trained weights and runs detection over the `predictions/` sample images.

## Environment

The notebooks were run on **Kaggle** and **Google Colab** and reference their filesystem conventions (`/kaggle/input/...`, `/kaggle/working/...`). Key dependencies:

- `torch`, `torchvision` (DamageNet, dataset utilities)
- `detectron2` (Mask R-CNN)
- `ultralytics` (YOLOv8)
- `opencv-python`, `matplotlib`, `pandas`, `tqdm`, `pyyaml`

To run locally, update the hardcoded `/kaggle/...` paths to point at your own copy of the VehiDE dataset, and install Detectron2 / Ultralytics per their respective installation guides (Detectron2 in particular requires a matching PyTorch/CUDA build).

## Usage

1. Download the [VehiDE dataset](https://www.kaggle.com/datasets/hendrichscullen/vehide-dataset-automatic-vehicle-damage-detection) and update the dataset paths in the notebook(s) you plan to run.
2. Run the annotation-conversion cells to generate COCO/YOLO-format labels (`train_coco.json`, `val_coco.json`, or the `yolo_dataset/` label files).
3. Train the model of your choice:
   - `automatic-vehicle-damage-detection.ipynb` or `DamageNet.ipynb` / `DamageNet_with_nodamage_class.ipynb` for the classification approach
   - `Mask-R-CNN.ipynb` for instance segmentation
   - `automatic-vehicle-damage-detection.ipynb` or `YOLOv8_colab.ipynb` for YOLOv8 detection
4. Run inference on the sample images in `predictions/` to visualize detected damage.
