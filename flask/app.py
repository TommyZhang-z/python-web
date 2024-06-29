import torch
import pickle
from flask import Flask, request, jsonify
from network import SimpleModel


# deep learning model
dl_model = SimpleModel()
dl_model.load_state_dict(torch.load("model_checkpoint.ckpt"))
dl_model.eval()

# machine learning model
with open("trained_model.pkl", "rb") as f:
    ml_model = pickle.load(f)


app = Flask(__name__)


@app.route("/deep_learning", methods=["POST"])
def deep_learning():
    data = request.json
    if (
        not isinstance(data, list)
        or len(data) != 10
        or not all(isinstance(i, float) for i in data)
    ):
        return {"detail": "Invalid input: Expecting a list of 10 float numbers."}

    input_tensor = torch.tensor([data], dtype=torch.float32)
    with torch.no_grad():
        prediction = dl_model(input_tensor)

    return jsonify({"prediction": prediction.item()})


@app.route("/machine_learning", methods=["POST"])
def machine_learning():
    data = request.json
    if (
        not isinstance(data, list)
        or len(data) != 10
        or not all(isinstance(i, float) for i in data)
    ):
        return {"detail": "Invalid input: Expecting a list of 10 float numbers."}

    prediction = ml_model.predict([data])[0]
    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
