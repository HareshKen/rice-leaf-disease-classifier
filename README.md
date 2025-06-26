# Rice Leaf Disease Prediction using CNN (MobileNetV2)

This project uses a **Convolutional Neural Network (CNN)** with **MobileNetV2** as a feature extractor to automatically classify rice leaf diseases into three categories:

- **Bacterial Leaf Blight**
- **Brown Spot**
- **Leaf Smut**

The goal is to accurately detect these diseases using image classification techniques to assist farmers in minimizing crop loss.

---

## 📁 Folder Structure

Ensure your project directory follows this structure:
```
project_folder/
│
├── rice_leaf_cnn.py # Main Python script
├── rice_leaf_dataset/ # Dataset folder
│ ├── bacterial leaf blight/
│ ├── brown spot/
│ └── leaf smut/
│ └── *.jpg # Images per class
```

---

## ⚙️ Setup Instructions

### ✅ Prerequisites

Install the required Python libraries using:

```
pip install tensorflow matplotlib seaborn scikit-learn pandas
```
## How to Run the Project
1. Clone or download this repository into your local machine.

2. Download the rice leaf disease dataset and extract it into a folder named rice_leaf_dataset/.

3. Place the rice_leaf_cnn.py script in the root of your project_folder, alongside rice_leaf_dataset/.

4. Open a terminal or command prompt.

5. Navigate to the project directory:
```
cd path/to/project_folder
```
6. Run the script:
```
python rice_leaf_cnn.py
```
## After Training
Once training is complete (takes approximately 1–3 minutes depending on hardware), the script will:

Print Precision, Recall, F1-score, and Confusion Matrix

Display:

Training vs Validation Accuracy

Confusion Matrix (as a heatmap)

Save the trained model as:
```
rice_leaf_model.keras
```
