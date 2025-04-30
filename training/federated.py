import time
import gc
import numpy as np
from models.aggregation import aggregate_weights
from models.boneage_model import BoneAgeRegressor
from training.evaluation import evaluate_model

def federated_training(clients_data, server_data, num_rounds=10, client_epochs=1, save_rounds=20):

    # Initialize timers dictionary
    time_metrics = {
        'round_times': [],
        'client_training_times': [[] for _ in range(len(clients_data))],
        'aggregation_times': [],
        'evaluation_times': []
    }
    
    # Initialize global model
    regressor = BoneAgeRegressor(activation='relu', dropout_rate=0.2)
    global_model = regressor.build_model(input_shape=(224, 224, 3), weights=None)
    global_weights = global_model.get_weights()

    # Initialize persistent client models
    client_models = []
    for i in range(len(clients_data)):
        print(f"Initializing Client {i+1} model...")
        model = regressor.build_model(input_shape=(224, 224, 3), weights=None)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        client_models.append(model)
    
    # Prepare server data
    X_server, y_server, male_server  = server_data
    
    # Track metrics
    history = {
        'round': [],
        'client_train_loss': [[] for _ in range(len(clients_data))],
        'client_train_mae': [[] for _ in range(len(clients_data))],
        'client_val_loss': [[] for _ in range(len(clients_data))],
        'client_val_mae': [[] for _ in range(len(clients_data))],
        'server_loss': [],
        'server_mae': []
    }
    
    best_server_mae = float('inf')
    best_round = -1
    best_weights = None
    
    for round_num in range(num_rounds):
        round_start_time = time.time()
        
        # Collect garbage
        gc.collect()
        
        print(f"\n=== Round {round_num + 1}/{num_rounds} ===")
        
        # Client training
        client_weights = []
        client_metrics = []

        for i, (X_train, X_val, y_train, y_val, male_train, male_val) in enumerate(clients_data):
            print(f"\nTraining Client {i+1}")
            client_start_time = time.time()
            
            # Update weights
            if round_num > 0:
                client_models[i].set_weights(global_weights)
            
            # Train for specified epochs
            client_history = client_models[i].fit(
                [X_train, male_train], y_train,
                validation_data=([X_val, male_val], y_val),
                epochs=client_epochs,
                batch_size=32,
                verbose=1
            )
            
            # Store metrics
            history['client_train_loss'][i].append(client_history.history['loss'][-1])
            history['client_train_mae'][i].append(client_history.history['mae'][-1])
            history['client_val_loss'][i].append(client_history.history['val_loss'][-1])
            history['client_val_mae'][i].append(client_history.history['val_mae'][-1])
            
            # Collect weights
            client_weights.append(client_models[i].get_weights())
            client_metrics.append({
                'train_loss': client_history.history['loss'][-1],
                'train_mae': client_history.history['mae'][-1],
                'val_loss': client_history.history['val_loss'][-1],
                'val_mae': client_history.history['val_mae'][-1]
            })
            
            client_training_time = time.time() - client_start_time
            time_metrics['client_training_times'][i].append(client_training_time)
            
        
        # Aggregate weights
        aggregation_start = time.time()
        global_weights = aggregate_weights(client_weights)
        time_metrics['aggregation_times'].append(time.time() - aggregation_start)
        
        # Evaluate on server data
        eval_start = time.time()
        global_model.set_weights(global_weights)
        server_metrics = evaluate_model(global_model, X_server, male_server, y_server)
        time_metrics['evaluation_times'].append(time.time() - eval_start)
        
        history['round'].append(round_num)
        history['server_loss'].append(server_metrics['loss'])
        history['server_mae'].append(server_metrics['mae'])
        
        print(f"\nServer Metrics - Loss: {server_metrics['loss']:.4f}, MAE: {server_metrics['mae']:.4f}")
        
        # Track best model (based on server MAE)
        if server_metrics['mae'] < best_server_mae and round_num < save_rounds:
            best_server_mae = server_metrics['mae']
            best_round = round_num
            best_weights = [w.copy() for w in global_weights]

        # Round timing
        round_time = time.time() - round_start_time
        time_metrics['round_times'].append(round_time)
        print(f"\nRound {round_num+1} completed in {round_time:.2f} seconds")
        print("Client training times:", [f"{t[-1]:.2f}s" for t in time_metrics['client_training_times']])
        print(f"Aggregation time: {time_metrics['aggregation_times'][-1]:.2f}s")
        print(f"Evaluation time: {time_metrics['evaluation_times'][-1]:.2f}s")
        
    
    # Set final model to best weights
    if best_weights is not None:
        global_model.set_weights(best_weights)
        print(f"\nBest model found at round {best_round + 1} with server MAE: {best_server_mae:.4f}")
    else:
        print("\nNo model saved within the specified rounds")
    
    return global_model, history, time_metrics, best_round