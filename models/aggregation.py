import numpy as np
def aggregate_weights(client_weights):
    """Average weights from all clients"""
    avg_weights = []
    for weights_list in zip(*client_weights):
        avg_weights.append(np.mean(weights_list, axis=0))
    return avg_weights