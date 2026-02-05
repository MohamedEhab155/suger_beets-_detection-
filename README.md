# Sugar Beet and Weed Classification

A computer vision system for automated detection and classification of sugar beets and weeds in agricultural fields using YOLOv12s.

![Sugar Beet Field](https://stmaaprodfwsite.blob.core.windows.net/assets/sites/1/2024/02/Sugar-beet-plants-growing-on-black-fen-soil-Cambridgeshire-early-June-c-GNP-web.jpg)

## Overview

This project addresses the challenge of distinguishing sugar beet crops from weeds in real-time field conditions. Using the YOLOv12s object detection architecture, the model achieves high accuracy in identifying and localizing both sugar beets and various weed species, supporting precision agriculture workflows.

## Dataset

**Source:** [Roboflow Sugar Beets Dataset](https://universe.roboflow.com/vision-3gxqu/sugarbeets-zg7nc/dataset/2)

The dataset includes annotated images from agricultural fields with bounding boxes for sugar beets and multiple weed types. Images were split into training, validation, and test sets to ensure robust evaluation.

## Technical Stack

- **Framework:** Ultralytics YOLOv12s
- **Training Platform:** Kaggle (GPU-accelerated)
- **Language:** Python 3.x
- **Key Libraries:** PyTorch, OpenCV, NumPy

## Installation

```bash
git clone https://github.com/MohamedEhab155/suger_beets-_detection-.git
cd suger_beets-_detection-
pip install -r requirements.txt
```

## Usage

Basic inference example:

```python
from ultralytics import YOLO

# Load trained weights
model = YOLO('models/best.pt')

# Run detection
results = model('path/to/image.jpg')

# Display or save
results[0].show()
results[0].save(filename='output.jpg')
```

## Model Performance

The trained model demonstrates strong performance across key metrics:

### F1-Score
![F1 Curve](https://raw.githubusercontent.com/MohamedEhab155/suger_beets-_detection-/main/runs/detect/train/F1_curve.png)

The F1-score curve shows the balance between precision and recall at different confidence thresholds.

### Precision-Recall
![PR Curve](https://raw.githubusercontent.com/MohamedEhab155/suger_beets-_detection-/main/runs/detect/train/PR_curve.png)

This curve indicates the model's ability to maintain high precision while maximizing recall across both classes.

### Training Metrics
![Results](https://raw.githubusercontent.com/MohamedEhab155/suger_beets-_detection-/main/runs/detect/train/results.png)

Complete training results showing loss curves, mAP scores, and other evaluation metrics.

## Training Details

The model was trained using the following approach:

1. Downloaded and preprocessed the annotated dataset from Roboflow
2. Configured YOLOv12s with hyperparameters optimized for agricultural imagery
3. Trained on Kaggle's GPU infrastructure (see [notebook](https://www.kaggle.com/code/mohamedehab0122/using-kaggel/edit))
4. Monitored validation metrics to prevent overfitting
5. Selected best checkpoint based on mAP@0.5 performance

## Results

The model achieves reliable detection across varying field conditions including different lighting, crop densities, and growth stages. Inference speed is suitable for real-time applications on standard hardware.

## Potential Applications

- Automated weed mapping and treatment planning
- Crop health monitoring and yield estimation
- Integration with robotic weeding systems
- Data collection for precision agriculture analytics

## Future Improvements

- Fine-grained weed species classification
- Deployment optimization for edge devices (e.g., TensorRT, ONNX)
- Multi-temporal tracking of crop/weed growth
- Integration with UAV imagery for field-scale monitoring

## Acknowledgments

- Roboflow for dataset hosting and annotation tools
- Ultralytics team for YOLOv12 implementation
- Kaggle for computational resources

## License

This project is available for educational and research purposes.

---

**AI Engineer:** Mohamed Ehab  
**Contact:** [GitHub Profile](https://github.com/MohamedEhab155)