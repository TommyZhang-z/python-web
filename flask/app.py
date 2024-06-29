import pickle
from flask import Flask, request, jsonify
import torch
from network import SimpleModel


app = Flask(__name__)


with open("trained_model.pkl", "rb") as file:
    ml_model = pickle.load(file)

dl_model = SimpleModel()
dl_model.load_state_dict(torch.load("model_checkpoint.ckpt"))
dl_model.eval()


@app.route("/deep_learning", methods=["POST"])
def deep_learning():
    data = request.json
    if len(data) != 10 or not all(isinstance(x, float) for x in data):
        return (
            jsonify({"detail": "Invalid input: Expecting a list of 10 float numbers."}),
            400,
        )

    input_tensor = torch.tensor([data], dtype=torch.float32)

    with torch.no_grad():
        prediction = dl_model(input_tensor)

    prediction_value = prediction.item()

    return jsonify({"prediction": prediction_value})


@app.route("/machine_learning", methods=["POST"])
def machine_learning():
    data = request.json
    if len(data) != 10 or not all(isinstance(x, float) for x in data):
        return (
            jsonify({"detail": "Invalid input: Expecting a list of 10 float numbers."}),
            400,
        )

    prediction = ml_model.predict([data])

    return jsonify({"prediction": prediction[0]})
