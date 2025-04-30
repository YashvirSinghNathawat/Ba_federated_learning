import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_parquet_with_arrays(path, expected_shape, expected_dtype='float32'):
    """Load Parquet file and reconstruct numpy arrays"""
    df = pd.read_parquet(path)
    
    for col in df.columns:
        # Check if column contains serialized arrays
        sample = df[col].iloc[0]
        if isinstance(sample, bytes) and len(sample) == np.prod(expected_shape) * np.dtype(expected_dtype).itemsize:
            df[col] = df[col].apply(
                lambda x: np.frombuffer(x, dtype=expected_dtype).reshape(expected_shape)
            )
    return df

# Change img change from 1 to 3
def to_rgb(images):
    return np.stack([np.stack([img]*3, axis=-1) for img in images])

def load_client_data(path, expected_shape=(224, 224)):
    """Load and preprocess client data"""
    df = load_parquet_with_arrays(path, expected_shape, 'uint8')
    df['image'] = df['image'].apply(lambda x: x.astype('float32') / 255.0)
    
    X = to_rgb(np.stack(df['image'].values))
    y = df['boneage'].values
    male = df['male'].values
    
    return train_test_split(X, y, male, test_size=0.2, random_state=42)

def load_server_data(path, expected_shape=(224, 224)):
    """Load and preprocess server data (no train-test split)"""
    df = load_parquet_with_arrays(path, expected_shape, 'uint8')
    df['image'] = df['image'].apply(lambda x: x.astype('float32') / 255.0)
    
    X = to_rgb(np.stack(df['image'].values))
    y = df['boneage'].values
    male = df['male'].values
    
    return X, y, male