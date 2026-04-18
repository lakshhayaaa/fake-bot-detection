from fastapi import FastAPI

app = FastAPI()
@app.get("/")
def read_root():
    return {"message": "Fake bot prediction system is up and running"}
