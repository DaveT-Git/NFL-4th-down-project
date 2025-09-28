
import streamlit as st
import pandas as pd
import joblib

# Load the saved components
logreg_model = joblib.load('logreg_model.joblib')
scaler = joblib.load('scaler.joblib')
kmeans = joblib.load('kmeans.joblib')
features = joblib.load('features.joblib')  # ['yardline_100', 'ydstogo', 'score_differential', 'game_seconds_remaining', 'cluster']
main_features = features[:-1]  # ['yardline_100', 'ydstogo', 'score_differential', 'game_seconds_remaining']

# App title and description
st.title("NFL 4th Down Decision Tool")
st.write("Enter game conditions to predict 4th down conversion success and get a recommendation.")

# User inputs for the four main features
yardline_100 = st.number_input("Yards to Opponent's Endzone (yardline_100)", min_value=0.0, max_value=100.0, value=45.0)
ydstogo = st.number_input("Yards to Go (ydstogo)", min_value=0.0, max_value=50.0, value=3.0)
score_differential = st.number_input("Score Differential (positive if leading)", min_value=-50.0, max_value=50.0, value=-7.0)
game_seconds_remaining = st.number_input("Seconds Remaining in Game (game_seconds_remaining)", min_value=0.0, max_value=3600.0, value=120.0)

# Button to trigger prediction
if st.button("Predict Success Probability"):
    # Prepare unseen data as a DataFrame
    unseen_data = pd.DataFrame({
        'yardline_100': [yardline_100],
        'ydstogo': [ydstogo],
        'score_differential': [score_differential],
        'game_seconds_remaining': [game_seconds_remaining]
    })

    # Scale the four main features using the pre-fitted scaler
    unseen_data_scaled = scaler.transform(unseen_data[main_features])

    # Assign cluster using the loaded K-Means on scaled data (since K-Means was fit on scaled data)
    unseen_data['cluster'] = kmeans.predict(unseen_data_scaled)

    # Full input for model (four main + cluster)
    unseen_data_full = unseen_data[features]

    # Make prediction
    proba = logreg_model.predict_proba(unseen_data_full)[:, 1][0]  # Probability of success (class 1)

    # Recommendation based on threshold
    threshold = 0.5  # Adjustable threshold (e.g., 0.6 for conservative)
    recommendation = "Go for it" if proba > threshold else "Punt or kick"

    # Display results
    st.subheader("Prediction Results")
    st.write(f"**Predicted Success Probability:** {proba:.3f} (or {proba * 100:.1f}%)")
    st.write(f"**Recommendation:** {recommendation}")
    if proba > threshold:
        st.success("High chance of success – consider going for it!")
    else:
        st.warning("Low chance of success – safer to punt or kick.")

    # Optional: Show input details for debugging
    st.write("**Input Details:**", unseen_data_full.to_dict())