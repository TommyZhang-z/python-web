import numpy as np
import json
import torch
import pickle

from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt

from .network import SimpleModel


# deep learning model
dl_model = SimpleModel()
dl_model.load_state_dict(torch.load("model_checkpoint.ckpt"))
dl_model.eval()


# machine learning model
with open("trained_model.pkl", "rb") as file:
    ml_model = pickle.load(file)


# recommend to use drf instead
@csrf_exempt
def deep_learning(request):
    if request.method == "POST":
        try:
            # load data from request
            data = json.loads(request.body)

            # check if data is a list
            # check if data length is 10
            # check if all elements in data are floats
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

            return JsonResponse({"prediction": prediction.item()})
        except Exception as e:
            return HttpResponseBadRequest({"message": str(e)})
    else:
        return HttpResponseBadRequest({"message": "Please send a POST request."})


@csrf_exempt
def machine_learning(request):
    if request.method == "POST":
        try:
            # load data from request
            data = json.loads(request.body)

            # check if data is a list
            # check if data length is 10
            # check if all elements in data are floats
            if (
                not isinstance(data, list)
                or len(data) != 10
                or not all(isinstance(x, float) for x in data)
            ):
                return HttpResponseBadRequest(
                    "Invalid input: Expecting a list of 10 float numbers."
                )

            # make predictions
            prediction = ml_model.predict([data])[0]

            return JsonResponse({"prediction": prediction.tolist()})
        except Exception as e:
            return HttpResponseBadRequest({"message": str(e)})
    else:
        return HttpResponseBadRequest({"message": "Please send a POST request."})
