from tensorflow.keras.models import load_model
from dataloader.load_data import load_server_data

model_name = "results/boneage_fl_model.keras"
SERVER_PATH = 'data/boneage_clahe_server.parquet'

model = load_model(model_name)


# Load server data
X_server, y_server, male_server = load_server_data(SERVER_PATH)

# If your model expects gender as an input feature, concatenate it
# Example: model takes image and gender as inputs
# Adjust according to your model’s input structure
if isinstance(model.input, list):  # model has multiple inputs
    X_eval = [X_server, male_server]
else:
    X_eval = X_server

# Evaluate the model
loss, *metrics = model.evaluate(X_eval, y_server, verbose=1)
print("Evaluation Loss:", loss)
print("Other Metrics:", metrics)