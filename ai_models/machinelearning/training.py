from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import pickle

# Generate random regression data
X, y = make_regression(n_samples=100, n_features=10, noise=0.1)

# Define and train the model
model = LinearRegression()
model.fit(X, y)

# Export the trained model to a .pkl file
with open("trained_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model training complete and exported as trained_model.pkl")
