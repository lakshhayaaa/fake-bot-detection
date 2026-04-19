# predict.py (routes)
# This is the WAITER — it handles the /predict API endpoint.
# 
# When someone sends a POST request to /predict with user data,
# this file receives it, validates it, and returns the prediction.

from fastapi import APIRouter          # APIRouter lets us organize routes in separate files
from app.schemas.user_schema import UserFeatures   # our form/schema
from app.services.prediction import predict_user   # our prediction function

# Create a router (think of it as a mini-app just for prediction routes)
router = APIRouter()

# ── THE /predict ENDPOINT ────────────────────────────────────
# @router.post means: "when someone sends a POST request to /predict, run this function"
# POST is used when you're SENDING data (like submitting a form)
# GET is used when you're just reading data (like opening a webpage)

@router.post("/predict")
def predict(user: UserFeatures):
    """
    Receives user features, runs prediction, returns result.
    
    Example input (JSON):
    {
        "Number_of_followers": 1.6,
        "Number_of_following": 0.69,
        ...
    }
    
    Example output:
    {
        "prediction": "Bot",
        "confidence": 91.45,
        "bot_probability": 91.45,
        "human_probability": 8.55
    }
    """
    
    # user is a UserFeatures object — convert it to a plain dictionary
    # .model_dump() turns it into {"Number_of_followers": 1.6, ...}
    features = user.model_dump()
    
    # Call the prediction service (the chef) with the features
    result = predict_user(features)
    
    # Return the result — FastAPI automatically converts this dict to JSON
    return result