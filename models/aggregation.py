import numpy as np
from typing import List, Callable

def fed_avg(client_weights: List[List[np.ndarray]]) -> List[np.ndarray]:
    """Standard Federated Averaging"""
    return [np.mean(weights, axis=0) for weights in zip(*client_weights)]

def fed_avg_weighted(client_weights: List[List[np.ndarray]], 
                    client_samples: List[int]) -> List[np.ndarray]:
    """Sample-weighted Federated Averaging"""
    total_samples = sum(client_samples)
    return [np.average(weights, axis=0, weights=client_samples) 
           for weights in zip(*client_weights)]

def fed_prox(global_weights: List[np.ndarray],
            client_weights: List[List[np.ndarray]],
            mu: float = 0.01) -> List[np.ndarray]:
    """FedProx aggregation with proximal term"""
    return [
        global_layer + (mu * np.mean([client_layer - global_layer 
                                    for client_layer in client_layers], axis=0))
        for global_layer, client_layers in zip(global_weights, zip(*client_weights))
    ]

STRATEGIES = {
    'fed_avg': fed_avg,
    'fed_avg_weighted': fed_avg_weighted,
    'fed_prox': fed_prox
}