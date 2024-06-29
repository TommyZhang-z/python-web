import pickle
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI
from network import SimpleModel


ml_model = None
dl_model = SimpleModel()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_model
    with open("trained_model.pkl", "rb") as file:
        ml_model = pickle.load(file)

    dl_model.load_state_dict(torch.load("model_checkpoint.ckpt"))
    dl_model.eval()
    yield
    ml_model = None


app = FastAPI(lifespan=lifespan)


@app.post("/deep_learning")
async def deep_learning(data: list[float]):
    if len(data) != 10 or not all(isinstance(x, float) for x in data):
        return {"detail": "Invalid input: Expecting a list of 10 float numbers."}

    input_tensor = torch.tensor([data], dtype=torch.float32)

    with torch.no_grad():
        prediction = dl_model(input_tensor)

    prediction_value = prediction.item()

    return {"prediction": prediction_value}


@app.post("/machine_learning")
async def machine_learning(data: list[float]):
    if len(data) != 10 or not all(isinstance(x, float) for x in data):
        return {"detail": "Invalid input: Expecting a list of 10 float numbers."}

    prediction = ml_model.predict([data])

    return {"prediction": prediction[0]}
