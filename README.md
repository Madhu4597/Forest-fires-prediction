# Forest Fire Prediction  
_A Machine Learning Project for Wildfire Risk Estimation_  

---  
## Table of Contents  
- [Overview](#overview)  
- [Features](#features)  
- [Technologies Used](#technologies-used)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Deployment](#deployment)  
- [Live Demo](#live-demo)  
- [Examples](#examples)  
- [Contributing](#contributing)  
- [Acknowledgements](#acknowledgements)  

---  
## Overview  
Forest Fire Prediction is a machine learning project designed to predict the probability of wildfire occurrence based on meteorological and Fire Weather Index (FWI) data such as temperature, humidity, wind speed, rainfall, and indices like FFMC, DMC, DC, and ISI.  
The system helps in wildfire monitoring and prevention efforts by providing scalable and interpretable models, developed and tested in both VSCode and Jupyter Notebook environments.  
This project features a Streamlit-powered user interface for real-time predictions and interactive visualizations.  

Core goals:  
- Efficient preprocessing of forest fire datasets, including feature scaling and encoding.  
- Comparative evaluation of multiple ML algorithms for predictive performance.  
- User-friendly deployment through Streamlit, enabling interactive input and real-time fire risk estimation.  

---  
## Features  
- **Automated Data Preprocessing**: Handling missing values, categorical encoding (month/day), and scaling of numerical features.  
- **Exploratory Data Analysis (EDA)**: Visual and statistical summaries of meteorological and FWI features.  
- **Model Training & Evaluation**: Implements Logistic Regression, Decision Trees, Random Forest, K-Nearest Neighbors, and SVM.  
- **Performance Metrics Dashboard**: Includes accuracy, precision, recall, F1-score, ROC-AUC, PR-AUC, and confusion matrix visualizations.  
- **Hyperparameter Tuning & Error Analysis**: Optimization of model parameters and review of misclassifications.  
- **Deployment with Streamlit**: Interactive web interface for inputting feature values or uploading datasets and obtaining predictions.  
- **Modular Notebook & Script Design**: Clear structure in Jupyter Notebook and VSCode for easy experimentation and reproducibility.  

---  
## Technologies Used  
- **Python 3**  
- **pandas** & **numpy** — Data manipulation and preprocessing  
- **scikit-learn** — Model training, evaluation, and hyperparameter tuning  
- **matplotlib** & **seaborn** — Data visualization and plotting  
- **Jupyter Notebook** — Interactive development and experimentation  
- **VSCode** — Code editing and project management  
- **Streamlit** — Deployment and interactive user interface creation  

---  
## Installation  
1. Clone the repository:  
    ```
    git clone https://github.com/yourusername/forest-fire-prediction.git  
    cd forest-fire-prediction  
    ```  
2. Install required dependencies:  
    ```
    pip install -r requirements.txt  
    ```  

---  
## Usage  
1. Place the dataset file (e.g., `forestfires.csv`) inside the `data/` directory.  
2. To explore and train models, launch Jupyter Notebook:  
    ```
    jupyter notebook  
    ```  
3. Open the notebook file: `Forest-fire-prediction-project.ipynb` and run the cells sequentially to preprocess data, train models, and analyze results.  
4. Alternatively, run Python scripts in VSCode for code-based experimentation and automation.  
5. Findings and evaluation reports will be saved in the `results/` and visualization assets in the `assets/figures/` folders.  

---  
## Deployment  
1. To launch the interactive Streamlit application, run:  
    ```
    streamlit run app.py  
    ```  
2. Use the provided UI to input meteorological attributes and obtain real-time wildfire probability predictions.  

---  
## Live Demo  
Try the interactive web app for Forest Fire Prediction here:  
[Forest Fire Prediction Streamlit App](#)  

---  
## Examples  
- **Input**: Attributes such as temperature, humidity, wind speed, rainfall, and FWI indices (FFMC, DMC, DC, ISI).  
- **Output**: Wildfire risk prediction (`High Risk` / `Low Risk`) along with model performance visualizations.  

---  
## Contributing  
1. Fork the repository.  
2. Create a feature branch:  
    ```
    git checkout -b feature/your-feature  
    ```  
3. Commit your changes:  
    ```
    git commit -m "Add feature description"  
    ```  
4. Push to your branch:  
    ```
    git push origin feature/your-feature  
    ```  
5. Open a Pull Request for review.  

---  
## Acknowledgements  
- UCI Forest Fires Dataset  
- Fire Weather Index (FWI) methodology  
- scikit-learn, pandas, numpy, matplotlib, seaborn, Streamlit  
- Open-source ML community for resources and best practices  
---  
