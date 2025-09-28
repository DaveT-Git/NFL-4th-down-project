# NFL 4th Down Decision Support Tool

## Overview
This repository contains a machine learning project designed to assist National Football League (NFL) coaches in making informed "go for it" or "kick" decisions on fourth down plays. Utilizing the nflfastR play-by-play dataset (2020-2024), the project develops a predictive model to estimate conversion success based on real-time game conditions. The methodology integrates unsupervised K-Means clustering to identify strategic patterns and evaluates six classification algorithms, with Logistic Regression selected as the optimal model (ROC-AUC = 0.682). The results are deployed in a Streamlit web application, offering a practical tool for decision support. This work was conducted as part of an Advanced Data Applications course project, documented in a ~10-page APA research paper.

## Features
- Predicts fourth down conversion success using features like yardline_100, ydstogo, score_differential, and game_seconds_remaining.
- Incorporates K-Means clusters to enhance model performance.
- Includes a Streamlit app for real-time predictions.
- Provides reproducible code and analysis for academic or professional use.

## Files

### Jupyter Notebooks
- **`Project_EDA.ipynb`**: This notebook contains the initial exploratory data analysis (EDA) of the nflfastR dataset. It covers data loading, filtering for fourth down pass and run plays (reducing the dataset from 246,218 to 3,932 instances), and variable selection (e.g., series_success as the target, four main features). It includes descriptive statistics, histograms (e.g., ydstogo distribution), and a correlation matrix to assess relationships (e.g., wp with score_differential). This file serves as the foundation for understanding data patterns and guiding feature engineering.
  
- **`Model_Iteration_First_Three_Feature_Sets.ipynb`**: This notebook iterates through six classification algorithms (Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, Support Vector Machine, and Gradient Boosting) across the first three feature sets: (1) four main features only, (2) four main features plus ep, and (3) four main features plus wp and ep. It details model training, evaluation metrics (accuracy, precision, recall, F1-score, ROC-AUC), and preliminary performance comparisons, setting the stage for identifying promising models.

- **`Final_Model_with_KMeans_Clusters.ipynb`**: This notebook finalizes the modeling process by adding K-Means clusters as a feature to the four main features. It implements standardization with StandardScaler, fits K-Means with 5 clusters, and re-evaluates the six algorithms. The notebook highlights Logistic Regression as the best performer (ROC-AUC = 0.682), includes cross-validation (5-fold), and extracts feature coefficients. It also prepares the model for deployment in the Streamlit app.

### Other Files
- **`fourth_down_dataframe.csv`**: Filtered dataframe used for modeling that focuses exclusivel on 4th down plays where teams attempted a run or a pass, excluding field goal or punt attemps.
- Streamlit App folder
  - **`app.py`**: The Streamlit application script that loads the trained Logistic Regression model, scaler, and K-Means objects to predict conversion success based on user inputs.
  - **`logreg_model.joblib`, `scaler.joblib`, `kmeans.joblib`, `features.joblib`**: Serialized objects containing the trained model, scaler, cluster model, and feature list, respectively, required for the app.    

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/NFL_4th_Down_Decision_Tool.git
   cd NFL_4th_Down_Decision_Tool
