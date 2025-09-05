# Forest Fire Prediction  
_A Machine Learning Project for Data-Driven Wildfire Risk Estimation_  
---

## Table of Contents  
- Overview  
- Features  
- Technologies Used  
- Installation  
- Usage  
- Deployment  
- Live Demo  
- Examples  
- Contributing  
- Acknowledgements  

---

## Overview  
Forest Fire Prediction is a machine learning project designed to estimate the probability of wildfire occurrence based on meteorological and Fire Weather Index (FWI) features such as temperature, humidity, wind speed, rainfall, and indices like FFMC, DMC, DC, and ISI.  
The system provides a reproducible pipeline developed and tested in both VSCode and Jupyter Notebook environments, with an interactive Streamlit application for real-time predictions and visualization.  

Core goals:  
- Efficient preprocessing of fire weather datasets (encoding, scaling, handling missing data).  
- Comparative evaluation of multiple ML algorithms for predictive accuracy.  
- Real-time deployment with Streamlit to support decision-makers in risk monitoring.  

---

## Features  
- **Automated Data Preprocessing**: Cleans, encodes categorical features (month/day), and scales numerical inputs.  
- **Exploratory Data Analysis (EDA)**: Statistical and graphical summaries for fire risk patterns.  
- **Model Training & Evaluation**: Implements Logistic Regression, Decision Trees, Random Forests, KNN, and SVM.  
- **Performance Metrics Dashboard**: Accuracy, precision, recall, F1-score, ROC-AUC, PR-AUC, and confusion matrix visualizations.  
- **Hyperparameter Tuning & Error Analysis**: Optimize thresholds and parameters to improve model reliability.  
- **Deployment with Streamlit**: Interactive UI for uploading datasets or entering feature values to predict fire probability.  
- **Modular Notebook & Script Design**: Clear structure for EDA, training, and application development.  

---

## Technologies Used  
- **Python 3**  
- **pandas** & **numpy** — Data preprocessing and manipulation  
- **scikit-learn** — Model training, evaluation, and hyperparameter tuning  
- **matplotlib** & **seaborn** — Data visualization and plots  
- **Jupyter Notebook** — Exploratory analysis and experimentation  
- **VSCode** — Code editing and project management  
- **Streamlit** — Web app deployment and real-time prediction interface  

---

## Installation  
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/forest-fire-prediction.git  
   cd forest-fire-prediction  
