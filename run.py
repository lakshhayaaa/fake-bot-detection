# run.py
# This is how you START the API server.
# Run this file with: python run.py
# Then open your browser at: http://127.0.0.1:8000

import uvicorn  # uvicorn is the server that runs FastAPI apps

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",   # "look in app/main.py, find the variable called 'app'"
        host="0.0.0.0",   # accept connections from anywhere
        port=8000,         # run on port 8000
        reload=True        # auto-restart when you save changes
    )