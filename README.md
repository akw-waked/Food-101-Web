# 🍽️ Food-101 Web App 

An interactive Streamlit application for visualizing **Food-101** image classification models with Grad-CAM explanations and recipe recommendations.
[Check the Live Web App](https://food-101-web-dl.streamlit.app/)

👉 **[Check the Live Web App](https://food-101-web-dl.streamlit.app/)**

---

# System Design

To make the project accessible to users and simulate a real-world application, I divided the project into two parts:

---

## 1. Training Application

### Summary

The training application focuses on building, training, and evaluating deep learning models using the **Food-101 dataset**.

- Models Used:
  - **Baseline CNN (ResNet18 from scratch)**
  - **Pretrained ResNet18 (Frozen layers)**
  - **Pretrained ResNet18 (Unfrozen all layers for fine-tuning)**

- Training Process:
  - Dataset split: Train / Validation / Test
  - Data augmentation: Rotation, horizontal flip, normalization
  - Hyperparameter tuning: Learning rate adjustments, batch size variations
  - Early stopping and checkpoint saving to optimize training
  - Visual training history: Accuracy and loss plots
  - Best model saved and exported for deployment

- Output:
  - Trained model checkpoint (`.pth` file)
  - Evaluation metrics (Accuracy, Classification Report, Confusion Matrix)
  - Training history and performance graphs
    
## 2. Deployment Application (Web App)

### Features

- Three model comparisons: Baseline, Pretrained Freeze, Pretrained Unfreeze
- Image transformation sliders (brightness, contrast, rotation)
- Grad-CAM visualizations for model interpretability
- Recipe recommendations based on predictions
- Fast and clean UI, optimized for performance

### Image

<img src="assets/predictedImage.png" alt="Predicted Image" width="600"/>
<br>
<img src="assets/API-Recipe.png" alt="API Recipe" width="600"/>


### Installation Web

```bash
git clone https://github.com/akw-waked/Food-101-Web
cd food101-web
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
