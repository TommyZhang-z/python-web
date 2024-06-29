from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt

import pickle
import json
import torch
from .network import SimpleModel


# machine learning model
with open("trained_model.pkl", "rb") as file:
    ml_model = pickle.load(file)

# deep learning model
dl_model = SimpleModel()
dl_model.load_state_dict(torch.load("model_checkpoint.ckpt"))
dl_model.eval()


# use DRF to simplify the code
@csrf_exempt
def deep_learning(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            if (
                not isinstance(data, list)
                or len(data) != 10
                or not all(isinstance(x, float) for x in data)
            ):
                return HttpResponseBadRequest(
                    "Invalid input: Expecting a list of 10 float numbers."
                )

            input_tensor = torch.tensor([data], dtype=torch.float32)

            with torch.no_grad():
                prediction = dl_model(input_tensor)

            prediction_value = prediction.item()

            return JsonResponse({"prediction": prediction_value})
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            return HttpResponseBadRequest(f"Invalid input: {str(e)}")
    else:
        return HttpResponseBadRequest("Invalid request method: Only POST is allowed.")


# use DRF to simplify the code
@csrf_exempt
def machine_learning(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            if (
                not isinstance(data, list)
                or len(data) != 10
                or not all(isinstance(x, float) for x in data)
            ):
                return HttpResponseBadRequest(
                    "Invalid input: Expecting a list of 10 float numbers."
                )

            prediction = ml_model.predict([data])[0]

            return JsonResponse({"prediction": prediction})
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            return HttpResponseBadRequest(f"Invalid input: {str(e)}")
    else:
        return HttpResponseBadRequest("Invalid request method: Only POST is allowed.")
