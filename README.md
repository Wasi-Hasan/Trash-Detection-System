# Constructing Robust Computer Vision Models with Custom-Labeled Data

## 📌 Project Overview
This repository contains the official implementation of our **Computer Vision & Deep Learning Framework** for automated **Trash and Waste Detection**. To identify the absolute best model for real-world deployment, we engineered an end-to-end pipeline benchmarking **3 distinct object detection paradigms** across **3 unique hyperparameter configurations each (9 experimental runs total)**.

We conducted extensive comparative training loops across anchor-free convolutional networks (**YOLOv8**), real-time vision transformers (**RT-DETR**), and focal-loss-driven architectures (**RetinaNet**), tuning optimizers, layer freezing, and batch layouts to maximize localization precision.

---

## 🛠️ Tech Stack & Key Frameworks
* **Core Libraries:** Python, PyTorch, Ultralytics, OpenCV
* **Development Environment:** Google Colab (GPU Accelerated: T4 / A100)
* **Dataset Format:** YOLO/Ultralytics compatible structured data (`data.yaml`)

---

## 📊 Dataset & Classes
The framework utilizes a custom-curated dataset comprising **10 distinct waste classes** annotated with high-precision bounding boxes for spatial ground-truth localization:
1. Aluminium Can
2. Cardboard Box
3. Food Wrapper
4. Glass Bottle
5. Paper Cup
6. Plastic Bag
7. Plastic Bottle
8. Styrofoam Cup
9. Tissue
10. Cloth

---

## 🚀 Experimental Configurations & Training Runs

We engineered a dynamic selection matrix comprising 9 unique execution pipelines to benchmark accuracy, convergence speed, and gradient scaling:

### 1. YOLOv8 Configuration Suite
* **Run 1.1 (AdamW):** Initialized with `yolov8s.pt` using the **AdamW** optimizer, Cosine Learning Rate scheduling, and a batch size of 32 to evaluate adaptive gradient convergence.
* **Run 1.2 (SGD):** Switched to the **SGD** optimizer with standard momentum and linear learning rate decay to contrast feature map optimization.
* **Run 1.3 (Frozen Layer Transfer Learning):** Utilized a lightweight `yolov8n.pt` backbone with the first 10 foundational feature extraction layers frozen (`freeze=10`) to speed up training on specialized target features.

### 2. Real-Time Detection Transformers (RT-DETR Suite)
* **Run 2.1 (Transformer + AdamW):** Leveraged the robust ViT backbone (`rtdetr-l.pt`) utilizing **AdamW** with a micro-tuned learning rate and cosine decay to stabilize self-attention matrix computations.
* **Run 2.2 (Transformer + SGD):** Executed the transformer architecture over an **SGD** optimizer to track multi-scale structural loss paths.
* **Run 2.3 (Batch8 Architecture):** Micro-batched the transformer processing loop (`batch=8`) to monitor constraint optimization on edge-simulated memory profiles.

### 3. RetinaNet Configuration Suite
* **Run 3.1 (RetinaNet + AdamW):** Deployed a feature-pyramid-driven RetinaNet architecture with **AdamW** optimization to target minor foreground-background class imbalances.
* **Run 3.2 (RetinaNet + SGD):** Evaluated standard focal loss minimization under localized **SGD** momentum profiles.
* **Run 3.3 (RetinaNet Frozen):** Frozen primary backbone feature blocks to implement rigid top-down and bottom-up multi-scale transfer learning hooks.

---


### Key Architecture Takeaways:
* **Real-Time Transformers (RT-DETR)** proved highly superior at contextualizing overlapping trash items, hitting peak metrics at **93.3% mAP@50** due to global token dependencies.
* **YOLOv8 Frameworks** delivered the fastest feature engineering and lowest execution latency, making them highly practical for high-frequency runtime deployment.
* **RetinaNet Models** effectively handled complex background noise suppression thanks to custom Focal Loss matrix adjustments.



