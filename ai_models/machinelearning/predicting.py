import numpy as np
import pickle

# Load the trained model from the .pkl file
with open("trained_model.pkl", "rb") as file:
    model = pickle.load(file)

# Generate some random input data for prediction
X_new = np.array(
    [
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
        ]
    ]
)

# Make predictions
predictions = model.predict(X_new)

print("Predictions:")
print(predictions)
