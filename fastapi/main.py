import torch
import pickle
from contextlib import asynccontextmanager

from fastapi import FastAPI
from scalar_fastapi import get_scalar_api_reference

from network import SimpleModel


# deep learning model
dl_model = None

# machine learning model
ml_model = None


# life span
@asynccontextmanager
async def lifespan(_: FastAPI):
    global dl_model, ml_model

    with open("trained_model.pkl", "rb") as f:
        ml_model = pickle.load(f)

    dl_model = SimpleModel()
    dl_model.load_state_dict(torch.load("model_checkpoint.ckpt"))
    dl_model.eval()
    yield
    dl_model = None
    ml_model = None


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)


@app.post("/deep_learning")
def deep_learning(data: list[float]):
    if len(data) != 10:
        return {"detail": "Invalid input: Expecting a list of 10 float numbers."}

    input_tensor = torch.tensor([data], dtype=torch.float32)
    with torch.no_grad():
        prediction = dl_model(input_tensor)

    return {"prediction": prediction.item()}


@app.post("/machine_learning")
def machine_learning(data: list[float]):
    if len(data) != 10:
        return {"detail": "Invalid input: Expecting a list of 10 float numbers."}

    prediction = ml_model.predict([data])[0]
    return {"prediction": prediction}


@app.get("/docs", include_in_schema=False)
def scalar_docs():
    return get_scalar_api_reference(openapi_url=app.openapi_url, title="FastAPI")
