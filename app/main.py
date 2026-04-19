# main.py
# This is the FRONT DOOR of our entire API.
# It creates the FastAPI app and registers all our routes.

from fastapi import FastAPI
from app.routes.predict import router as predict_router  # import our predict route

# Create the FastAPI app
# title and description show up in the automatic documentation page
app = FastAPI(
    title="Fake Bot Detection API",
    description="Send GitHub user features → Get Bot or Human prediction",
    version="1.0.0"
)

# ── CONNECT THE ROUTES ───────────────────────────────────────
# "include_router" is like saying: "hey app, also handle all routes from predict_router"
# prefix="/api" means all routes from predict_router will start with /api
# So /predict becomes /api/predict
app.include_router(predict_router, prefix="/api")

# ── HOME ROUTE ───────────────────────────────────────────────
@app.get("/")
def home():
    return {
        "message": "Fake Bot Detection API is running!",
        "docs": "Go to /docs to test the API in your browser"
    }