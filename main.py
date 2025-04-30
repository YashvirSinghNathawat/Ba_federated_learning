from dataloader.load_data import load_client_data, load_server_data
from training.federated import federated_training
import sys
import os
import json

CLIENT_PATHS = [
    'data/boneage_clahe_client_1.parquet',
    'data/boneage_clahe_client_2.parquet', 
    'data/boneage_clahe_client_3.parquet'
]
SERVER_PATH = 'data/boneage_clahe_server.parquet'
os.makedirs("results", exist_ok=True)

def main():
    # Load data
    clients_data = [load_client_data(path) for path in CLIENT_PATHS]
    server_data = load_server_data(SERVER_PATH) 
    
    num_rounds = 1000
    
    # Print info for each client
    for idx, (X_train, X_test, y_train, y_test, male_train, male_test) in enumerate(clients_data, 1):
        print(f"\nClient {idx} Data:")
        print(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
        print(f"X_test shape: {X_test.shape}, dtype: {X_test.dtype}")
        print(f"y_train shape: {y_train.shape}, dtype: {y_train.dtype}")
        print(f"y_test shape: {y_test.shape}, dtype: {y_test.dtype}")
        print(f"male_train shape: {male_train.shape}, dtype: {male_train.dtype}")
        print(f"male_test shape: {male_test.shape}, dtype: {male_test.dtype}")
    
    # Print server data info
    X, male, y = server_data
    print("\nServer Data:")
    print(f"X shape: {X.shape}, dtype: {X.dtype}")
    print(f"male shape: {male.shape}, dtype: {male.dtype}")
    print(f"y shape: {y.shape}, dtype: {y.dtype}")
    
    model, history, time_metrics, best_round = federated_training(
        clients_data=clients_data,
        server_data=server_data,
        num_rounds=num_rounds
    )
    # Save model
    model.save('results/boneage_fl_model.keras')
    
    
    # Report best round
    print(f"\nBest round: {best_round + 1}")
    print(f"Best MAE: {min(history['server_mae']):.4f}")

    # Optionally save history and metrics (e.g., as JSON or pickle)
    with open("results/history.json", "w") as f:
        json.dump(history, f)
    with open("results/time_metrics.json", "w") as f:
        json.dump(time_metrics, f)
    
    print("Training is done")

if __name__ == "__main__":
    main()
    # with open('output.txt', 'w') as f:
    #     original_stdout = sys.stdout  # Save original stdout
    #     sys.stdout = f
    #     try:
    #         main()
    #     finally:
    #         sys.stdout = original_stdout
        
