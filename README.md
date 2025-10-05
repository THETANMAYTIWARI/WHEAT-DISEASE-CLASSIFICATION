# WHEAT-DISEASE-CLASSIFICATION

### Dataset Link:- 
https://drive.google.com/drive/folders/1WRbDyVA_a6QdCsHfNY4pM7fRG3ws-r4r?usp=drive_link

🌾 Wheat Disease Classification using VGG19

This project focuses on automated wheat disease detection using deep learning (VGG19). It classifies images of wheat plants into four categories:

Crown and Root Rot

Healthy Wheat

Leaf Rust

Loose Smut

🧠 Model Overview

Base Model: VGG19 (pre-trained on ImageNet)

Custom Layers: AveragePooling2D → Flatten → Dense → Dropout → Softmax

Optimizer: Adam

Loss Function: Categorical Crossentropy

Accuracy Achieved: ~93% on validation data

🧰 Tech Stack

Languages: Python

Libraries: TensorFlow / Keras, NumPy, OpenCV, Matplotlib, Seaborn, Scikit-learn

Environment: Google Colab

📊 Dataset

The dataset consists of labeled wheat images stored in Google Drive with four disease classes.
Each image is preprocessed (resized to 224×224 and normalized) before training.

🚀 Model Training

Data Augmentation using ImageDataGenerator

70–30 train-test split

50 epochs with early performance improvement

Trained model saved as model.h5

🧾 Evaluation

Accuracy: 93.1%

Metrics: Confusion Matrix & Classification Report

Visualization: Accuracy & Loss curves

🔍 Testing

You can test the model on new wheat images using:

model = load_model("model.h5")


The model outputs the predicted disease name on the image.

📦 Output Example

Prediction: PREDICTION: LEAF RUST

Visual Output: Displays the input image with the predicted label overlay.

🏁 Results Summary
Metric	Training	Validation
Accuracy	93.8%	93.1%
Loss	0.19	0.21
💾 Files

model.h5 → Trained model

lb.pickle → Label binarizer

acc_plot.png, loss_plot.png → Accuracy/Loss graphs

📜 Author

Developed by TANMAY TIWARI — Leveraging Deep Learning for Smart Agriculture 🌱
