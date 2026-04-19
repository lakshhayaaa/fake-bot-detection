# prediction.py
# This is the CHEF of our restaurant.
# It loads the ML model and does the actual prediction.

import pickle          # pickle is used to load our saved model.pkl file
import numpy as np     # numpy helps us work with numbers/arrays
from app.config import MODEL_PATH   # import the model path from our settings file

# ── LOAD THE MODEL ──────────────────────────────────────────
# We load the model ONCE when the app starts (not every time someone calls /predict)
# This is faster — imagine re-reading a whole book every time someone asks you a question
# Better to read it once and keep it in memory!

def load_model():
    """Opens the model.pkl file and loads the trained ML model into memory."""
    with open(MODEL_PATH, "rb") as f:   # "rb" means "read binary" — .pkl files are binary
        model = pickle.load(f)
    return model

# Load the model right now, when this file is first imported
model = load_model()

# ── THE PREDICTION FUNCTION ──────────────────────────────────
def predict_user(features: dict):
    """
    Takes a dictionary of features, runs it through the model,
    and returns whether the user is a Bot or Human + the probability.
    
    features: a dictionary like {"Number_of_followers": 1.6, "Number_of_following": 0.7, ...}
    """
    
    # Step 1: Convert the dictionary into a list of values (in the correct order!)
    # The model was trained on columns in a specific order — we must match that order.
    feature_order = [
        "Number_of_followers", "Number_of_following", "tfidf_similarity",
        "Number_of_Activity", "Number_of_Issue", "Number_of_Pull_Request",
        "Number_of_Repository", "Number_of_Commit", "Number_of_Active_day",
        "Periodicity_of_Activities", "Number_of_Connection_Account",
        "Median_Response_Time", "issue_activity_ratio", "pr_activity_ratio",
        "commit_per_repo", "follower_following_ratio", "activity_per_day"
    ]
    
    # Build an ordered list of values from the features dictionary
    input_values = [features[col] for col in feature_order]
    
    # Step 2: Convert to a 2D numpy array
    # Why 2D? Because the model expects shape (1, 17) — 1 row, 17 features
    # Think of it like a table with 1 row and 17 columns
    input_array = np.array(input_values).reshape(1, -1)
    
    # Step 3: Ask the model to predict
    prediction = model.predict(input_array)[0]        # 0 = Human, 1 = Bot
    probability = model.predict_proba(input_array)[0]  # [prob_human, prob_bot]
    
    # Step 4: Build a nice result to return
    label = "Bot" if prediction == 1 else "Human"
    confidence = round(float(probability[prediction]) * 100, 2)  # e.g. 91.45%
    
    return {
        "prediction": label,           # "Bot" or "Human"
        "confidence": confidence,      # e.g. 91.45 (means 91.45% sure)
        "bot_probability": round(float(probability[1]) * 100, 2),    # always show bot %
        "human_probability": round(float(probability[0]) * 100, 2)   # always show human %
    }