# config.py
# This file stores all the settings/configuration for our app.
# Think of it like a settings menu — one place to change things.

import os

# This is the path to our trained ML model file
# os.path.join builds the correct file path for any operating system (Mac/Windows/Linux)
MODEL_PATH = os.path.join("app", "models", "model.pkl")