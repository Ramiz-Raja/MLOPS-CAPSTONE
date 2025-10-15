import os, joblib
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

app = FastAPI(title='Titanic Predictor')

class InputRow(BaseModel):
    Pclass: int
    Sex: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked_Q: int = 0
    Embarked_S: int = 0

MODEL_PATH = '/tmp/model.joblib'

def try_download_from_wandb():
    try:
        import wandb
        api = wandb.Api()
        # Attempt to download the latest artifact. This requires WANDB_API_KEY be set in env
        art = api.artifact('titanic-rf:latest', type='model')
        path = art.download(root='/tmp')
        # find a joblib file in the downloaded artifact
        for root, _, files in __import__('os').walk(path):
            for f in files:
                if f.endswith('.joblib'):
                    src = __import__('os').path.join(root, f)
                    dst = MODEL_PATH
                    __import__('shutil').copy(src, dst)
                    return True
        return False
    except Exception as e:
        print('W&B model download failed:', e)
        return False

@app.on_event('startup')
def load_model():
    global model
    model = None
    # Try to load model from local file first
    if os.path.exists('rf_titanic.joblib'):
        model = joblib.load('rf_titanic.joblib')
        print('Loaded model from rf_titanic.joblib')
        return
    # If not found, try downloading from W&B (requires WANDB_API_KEY env var)
    ok = try_download_from_wandb()
    if ok and os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print('Loaded model from W&B artifact')
    else:
        print('No model available at startup. /predict will return an error until model is trained or placed locally.')

@app.post("/predict")
def predict(row: InputRow):
    if model is None:
        return {"error": "Model not loaded yet. Please train first."}

    df = pd.DataFrame([row.dict()])
    preds = model.predict(df)

    # Get probabilities in a simpler format
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)[0]  # first row
        proba = [float(x) for x in proba]
    else:
        proba = None

    return {
        "prediction": int(preds[0]),
        "probabilities": proba
    }
