# user_schema.py
# This is our "form" — it defines exactly what data we expect
# when someone calls our /predict endpoint.
# 
# Pydantic is the library that does this validation automatically.
# If someone sends wrong data, it automatically returns an error message.

from pydantic import BaseModel

class UserFeatures(BaseModel):
    # These are ALL the features our trained model expects.
    # The names MUST exactly match the column names in preprocessed_data.csv
    
    Number_of_followers: float
    Number_of_following: float
    tfidf_similarity: float
    Number_of_Activity: float
    Number_of_Issue: float
    Number_of_Pull_Request: float
    Number_of_Repository: float
    Number_of_Commit: float
    Number_of_Active_day: float
    Periodicity_of_Activities: float
    Number_of_Connection_Account: float
    Median_Response_Time: float
    issue_activity_ratio: float
    pr_activity_ratio: float
    commit_per_repo: float
    follower_following_ratio: float
    activity_per_day: float