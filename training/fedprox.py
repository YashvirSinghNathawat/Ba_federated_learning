import tensorflow as tf
from tensorflow.keras import losses

def train_fedprox(model, X, y, male, global_weights, optimizer, mu=0.01):
    """Custom training loop for FedProx"""
    with tf.GradientTape() as tape:
        # Standard prediction loss
        preds = model([X, male], training=True)
        loss = losses.MSE(y, preds)
        
        # Add proximal term
        prox_term = 0
        for layer, global_layer in zip(model.trainable_variables, global_weights):
            prox_term += tf.reduce_sum(tf.square(layer - global_layer))
        total_loss = loss + (mu/2) * prox_term
    
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))