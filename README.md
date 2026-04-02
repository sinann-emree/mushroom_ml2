# 🍄 Mushroom Toxicity Classifier: End-to-End ML Web App

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Web App](https://img.shields.io/badge/Web_Interface-Enabled-005571?style=for-the-badge)

An end-to-end Machine Learning web application designed to predict whether a mushroom is **edible or poisonous** based on its morphological features. This project demonstrates a complete data science lifecycle, from rigorous data preprocessing to advanced model evaluation and web deployment.

## ✨ Key Features & Methodology

Unlike basic prediction models, this project emphasizes robust model validation and structured data engineering:

* **Comprehensive Data Preprocessing Pipeline:** The raw data undergoes a systematic 6-step cleaning and feature engineering process (`oniseleme_1` to `oniseleme_6`), ensuring high-quality input for the models.
* **Advanced Model Validation:** * **Holdout Method:** Standard train/test splitting for baseline evaluation.
    * **K-Fold Cross-Validation:** Ensures the model's reliability and prevents overfitting by training/testing across multiple data folds.
* **Statistical Significance Testing:** Implements the **McNemar's Test** (`model_mcnemar.py`) to statistically compare the performance of different machine learning models, proving that accuracy improvements are not just due to random chance.
* **Interactive Web UI:** A user-friendly web interface (`templates` & `app.py`) allowing users to input specific mushroom attributes and receive real-time toxicity predictions.

## 📂 Project Structure

* `data/`: Contains the raw and processed datasets.
* `models/`: Stores the serialized, pre-trained machine learning models (e.g., `.pkl` files).
* `templates/`: HTML files for the frontend web interface.
* `outputs/`: Stores generated evaluation metrics and visualization graphs.
* `oniseleme_*.py`: Step-by-step data preprocessing and feature engineering scripts.
* `model_*.py`: Distinct scripts for various evaluation methodologies (Holdout, K-Fold, McNemar).
* `app.py` / `main.py`: The main application scripts handling backend logic and serving the web interface.

## 🚀 Getting Started

### Prerequisites
Make sure you have Python installed on your system.

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/mushroom_ml2.git](https://github.com/YOUR_USERNAME/mushroom_ml2.git)
   cd mushroom_ml2
