#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the dataset.
    Args:
        file_path (str): Path to the dataset CSV file.
    Returns:
        tuple: Preprocessed feature and target arrays.
    """
    # Load dataset
    weather_data = pd.read_csv(file_path)

    # Convert 'Formatted Date' to datetime and set as index
    weather_data['Formatted Date'] = pd.to_datetime(weather_data['Formatted Date'])
    weather_data.set_index('Formatted Date', inplace=True)

    # Drop unnecessary columns
    data = weather_data.drop(['Summary', 'Simplified Summary'], axis=1)

    # Encode 'Precip Type'
    label_encoder = LabelEncoder()
    data['Precip Type'] = label_encoder.fit_transform(data['Precip Type'])  # 0: none, 1: rain

    # Normalize features
    scaler = MinMaxScaler()
    features = data.drop('Precip Type', axis=1).values
    features_scaled = scaler.fit_transform(features)

    return features_scaled, data['Precip Type'].values, scaler


def create_sequences(features_scaled, target, time_steps):
    """
    Create time-series sequences for the LSTM model.
    Args:
        features_scaled (array): Scaled feature data.
        target (array): Target data.
        time_steps (int): Number of time steps for sequence creation.
    Returns:
        tuple: Feature and target sequences.
    """
    X, y = [], []
    for i in range(time_steps, len(features_scaled)):
        X.append(features_scaled[i - time_steps:i])
        y.append(target[i])

    return np.array(X), np.array(y)


def build_lstm_model(input_shape):
    """
    Build and compile the LSTM model.
    Args:
        input_shape (tuple): Shape of the input data.
    Returns:
        keras.Model: Compiled LSTM model.
    """
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Train and evaluate the LSTM model.
    Args:
        model (keras.Model): The LSTM model.
        X_train, y_train, X_test, y_test: Training and testing datasets.
    Returns:
        keras.callbacks.History: Training history.
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=64,
        callbacks=[early_stopping],
        verbose=1
    )
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
    return history


def save_model(model, file_path):
    """
    Save the trained model.
    Args:
        model (keras.Model): Trained LSTM model.
        file_path (str): File path to save the model.
    """
    model.save(file_path)
    print(f"Model saved at {file_path}")


def main():
    # File path
    file_path = r'C:\Users\S K KEERTHI RAGAV\OneDrive\Desktop\B.TECH CSE SEM 5\Principles of AI\new_modified_weatherHistory.csv'

    # Parameters
    time_steps = 24

    # Load and preprocess data
    features_scaled, target, scaler = load_and_preprocess_data(file_path)

    # Create time-series sequences
    X, y = create_sequences(features_scaled, target, time_steps)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build LSTM model
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    # Train and evaluate
    history = train_and_evaluate_model(model, X_train, y_train, X_test, y_test)

    # Save the model
    save_model(model, 'rain_prediction_lstm_model.h5')


if _name_ == "_main_":
    main()
