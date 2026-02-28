# 🐟 Multiclass Fish Image Classification

## 📌 Project Overview
This project focuses on classifying fish images into 11 distinct categories using Deep Learning. The objective was to build a highly accurate image classification system by training a Custom Convolutional Neural Network (CNN) from scratch and comparing its performance against five state-of-the-art pre-trained models using Transfer Learning. Finally, the best-performing models were evaluated, and a custom web application was deployed for real-time predictions.

## 🛠️ Skills & Technologies Used
* **Deep Learning:** TensorFlow, Keras, CNNs, Transfer Learning
* **Programming & Scripting:** Python
* **Web Deployment:** Streamlit
* **Data Processing & Visualization:** Pandas, NumPy, Matplotlib, `ImageDataGenerator`

## 🚀 Approach & Methodology
1.  **Data Preprocessing:** Images were loaded from distinct train/val/test directories. Pixel values were rescaled to a `[0, 1]` range. Data augmentation (rotations, zooming, flipping) was applied exclusively to the training set to improve model robustness.
2.  **Custom CNN:** Designed and trained a baseline CNN architecture featuring multiple `Conv2D` and `MaxPooling2D` layers, culminating in a `Dense` classifier. Implemented `Dropout` and `EarlyStopping` to prevent overfitting.
3.  **Transfer Learning:** Evaluated five pre-trained architectures (VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0). The base models were frozen, and custom classification heads (`GlobalAveragePooling2D` + `Dense`) were attached and trained.
4.  **Deployment:** Built an interactive web interface using Streamlit, allowing users to upload unseen fish images and receive real-time species predictions and confidence scores.

## 📊 Model Evaluation & Comparison Report
All models were evaluated against an unseen test dataset. The results are as follows:

| Model | Test Accuracy (%) |
| :--- | :--- |
| **MobileNet** | **99.18%** |
| **InceptionV3** | **97.52%** |
| **Custom CNN (Baseline)**| **97.00%** |
| **VGG16** | **77.06%** |
| **ResNet50** | **24.38%** |
| **EfficientNetB0** | **16.32%** |

**Key Insights:**
* **MobileNet** achieved the highest accuracy, proving highly efficient for this specific feature-extraction task.
* The **Custom CNN** performed exceptionally well, proving that a well-designed architecture with proper augmentation can rival heavier, pre-trained networks.
* Models like ResNet50 and EfficientNetB0 struggled under the constrained, fast-training parameters (frozen base layers, 3 epochs), indicating they would require extensive fine-tuning and unfreezing of deeper layers to converge properly on this dataset.

## 💻 How to Run the Application Locally
1. Clone this repository to your local machine.
2. Ensure you have the required libraries installed:
   ```bash
   pip install tensorflow streamlit numpy pillow