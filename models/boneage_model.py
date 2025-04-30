from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, GlobalAveragePooling2D
from tensorflow.keras.applications import InceptionV3
class BoneAgeRegressor:
    def __init__(self, activation='relu', dropout_rate=0.2):
        self.activation = activation
        self.dropout_rate = dropout_rate

    def __dense_block(self, X, units):
        X = Dense(units, activation=self.activation)(X)
        X = Dropout(self.dropout_rate)(X)
        return X

    def build_model(self, input_shape, gender_input_shape=(1,), weights='imagenet'):
        # Image input
        img_input = Input(shape=input_shape, name='image_input')

        # Gender input
        gender_input = Input(shape=gender_input_shape, name='gender_input')

        # Base CNN Model (InceptionV3)
        base_cnn = InceptionV3(include_top=False, input_shape=input_shape, weights=weights)
        cnn_features = base_cnn(img_input)
        cnn_features = GlobalAveragePooling2D()(cnn_features)
        cnn_features = Dropout(self.dropout_rate)(cnn_features)

        # Gender embedding
        gender_features = Dense(32, activation=self.activation)(gender_input)

        # Concatenate features
        combined_features = Concatenate(axis=-1)([cnn_features, gender_features])

        # Fully connected layers
        X = self.__dense_block(combined_features, 1024)
        X = self.__dense_block(X, 1024)

        # Output layer (linear activation for regression)
        output = Dense(1, activation='linear')(X)

        # Define the model
        model = Model(inputs=[img_input, gender_input], outputs=output)
        # Always compile the model
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        return model