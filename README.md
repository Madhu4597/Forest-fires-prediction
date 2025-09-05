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
Forest Fire Prediction is a machine learning project designed to predict the occurrence of wildfires based on meteorological and Fire Weather Index (FWI) features such as temperature, humidity, wind speed, rainfall, and indices like FFMC, DMC, DC, and ISI.  
The system aids in wildfire risk management by providing interpretable, data-driven models. The pipeline includes preprocessing, balancing imbalanced data, training multiple classifiers, and evaluating them with detailed performance metrics.  

Core goals:  
- Efficient preprocessing with feature encoding, scaling, and balancing.  
- Comparative evaluation of multiple ML algorithms.  
- User-friendly deployment through Streamlit for interactive input and predictions.  

---  
## Features  
- **Data Preprocessing**: Handling nulls, duplicates, class imbalance (RandomOverSampler), feature scaling, and cyclical encoding of month/day.  
- **Exploratory Data Analysis (EDA)**: Distribution checks, class imbalance analysis, and descriptive statistics.  
- **Model Training & Evaluation**: Implements KNN, Logistic Regression, Decision Tree, Random Forest, Extra Trees, and SVM with multiple kernels.  
- **Performance Metrics Dashboard**: Accuracy, precision, recall, F1-score, MCC, ROC-AUC, confusion matrix, and balanced accuracy.  
- **ROC Curve Visualizations**: Comparison of classifiers across thresholds.  
- **Deployment with Streamlit**: Interactive web application for real-time fire risk estimation.  

---  
## Technologies Used  
- **Python 3**  
- **pandas** & **numpy** — Data preprocessing and analysis  
- **scikit-learn** — Model training, evaluation, and metrics  
- **imblearn** — Handling class imbalance with oversampling  
- **matplotlib** & **seaborn** — Visualization and plotting  
- **Jupyter Notebook** — Interactive model experimentation  
- **VSCode** — Code editing and project management  
- **Streamlit** — Web application deployment  

---  
## Installation  
1. Clone the repository:  
    ```bash
    git clone https://github.com/yourusername/forest-fire-prediction.git  
    cd forest-fire-prediction  
    ```  
2. Install dependencies:  
    ```bash
    pip install -r requirements.txt  
    ```  

---  
## Usage  
1. Place the dataset file (e.g., `forestfires.csv`) inside the `data/` directory.  
2. Launch Jupyter Notebook for exploration and training:  
    ```bash
    jupyter notebook  
    ```  
   Open `Forest-fire-prediction-project.ipynb` and run all cells.  
3. Alternatively, run the Python script for end-to-end execution:  
    ```bash
    python "Forest fire prediction.py"  
    ```  
4. Outputs such as results tables, ROC plots, and metrics will be saved to `results/` and `assets/figures/`.  

---  
## Deployment  
1. Run the Streamlit application locally:  
    ```bash
    streamlit run app.py  
    ```  
2. Use the app UI to enter meteorological parameters or upload a dataset to predict fire occurrence.  

---  
## Live Demo  
Experience the live version of the app here:  
👉 [Forest Fire Prediction Streamlit App](https://forest-fires-prediction-jyjpnstx2tfnlbx9ktnctp.streamlit.app/)  

--- 
## Examples  
- **Input**: Meteorological conditions such as temperature, relative humidity, wind speed, rainfall, and FWI indices (FFMC, DMC, DC, ISI).  
- **Output**: Predicted wildfire risk (`High Risk` / `Low Risk`) with confidence metrics, confusion matrix, and ROC curve.  

---  
## Contributing  
1. Fork the repository.  
2. Create a feature branch:  
    ```bash
    git checkout -b feature/your-feature  
    ```  
3. Commit your changes:  
    ```bash
    git commit -m "Add feature description"  
    ```  
4. Push to your branch:  
    ```bash
    git push origin feature/your-feature  
    ```  
5. Open a Pull Request for review.  

---  
## Acknowledgements  
- **Dataset**: UCI Forest Fires Dataset & ISRO-derived data  
- **Fire Weather Index (FWI)** methodology for wildfire risk analysis  
- **Libraries**: scikit-learn, imblearn, pandas, numpy, matplotlib, seaborn, Streamlit  
- Open-source ML community for resources and inspiration  

---
