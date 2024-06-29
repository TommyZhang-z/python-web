import torch
from network import SimpleModel


model = SimpleModel()
model.load_state_dict(torch.load("model_checkpoint.ckpt"))
model.eval()

with torch.no_grad():
    x_test = torch.tensor(
        [
            0.1931,
            0.1561,
            -1.4582,
            1.0371,
            0.9861,
            -0.4463,
            0.3062,
            1.0581,
            0.0601,
            1.3870,
        ],
        dtype=torch.float32,
    )
    predictions = model(x_test)


print("Predictions:")
print(predictions)
