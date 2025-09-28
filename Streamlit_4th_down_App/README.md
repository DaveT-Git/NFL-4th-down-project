# NFL 4th Down Decision Support Tool - Streamlit App

## Overview
This subfolder contains the Streamlit web application for the NFL 4th Down Decision Support Tool, a machine learning-based solution to predict fourth down conversion success in real-time. The app leverages a trained Logistic Regression model (ROC-AUC = 0.682) with K-Means clusters, built using the nflfastR dataset (2020-2024). Users can input game conditions (yardline, yards to go, score differential, and time remaining) to receive a success probability and recommendation ("Go for it" or "Punt or kick"). This app is part of a broader project documented in a ~10-page APA research paper and hosted in the parent GitHub repository.

## Files
- **`app.py`**: The main Python script for the Streamlit app, which loads the model and provides the user interface.
- **`logreg_model.joblib`**: Serialized file containing the trained Logistic Regression model.
- **`scaler.joblib`**: Serialized StandardScaler object used for feature standardization.
- **`kmeans.joblib`**: Serialized K-Means clustering model for assigning game situation clusters.
- **`features.joblib`**: Serialized list of feature names used in the model.

## Setup Instructions
To run the Streamlit app, follow these steps:

1. **Clone the Repository**:
   Navigate to the parent repository and clone it:
   ```bash
   git clone https://github.com/yourusername/NFL_4th_Down_Decision_Tool.git
   cd NFL_4th_Down_Decision_Tool

2. **Install Dependencies**: Ensure you have Python 3.6 or higher installed. Install the required packages using pip:

     **`pip install streamlit joblib pandas scikit-learn`**

3. **Navigate to the Streamlit Subfolder**: Move into the streamlit_app directory:

    **`cd streamlit_app`**

4. **Run the App**: Launch the Streamlit app by:

    **`streamlit run app.py`**

   This will start a local server, and a browser window will open automatically (e.g., http://localhost:8501).

6. **Use the App**:
   
	•  Enter the game conditions in the provided input fields.

	•  Click “Predict Success Probability” to see the predicted success rate and recommendation.

	•  Explore different scenarios to test the model’s behavior.


