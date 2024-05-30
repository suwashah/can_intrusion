import numpy as np
import tensorflow as tf
import logging as log
import datetime
import os
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier, LogisticRegression
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from joblib import dump, load

trained_model_folder = "TrainedModels\\"
# Create a folder for log files if it doesn't exist
log_folder = 'logs'
os.makedirs(log_folder, exist_ok=True)

# Configure logging to save log file in the folder
log_file = os.path.join(log_folder, 'log_file.txt')

# Configure logging
log.basicConfig(filename=log_file,
                level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def build_autoencoder(input_dim, encoding_dim):
    # Input layer
    input_layer = Input(shape=(input_dim,))
    # Encoder layer
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    # Decoder layer
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    # Autoencoder model
    autoencoder = Model(input_layer, decoded)
    # Encoder model
    encoder = Model(input_layer, encoded)
    # Compile autoencoder
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder, encoder


def SVC_Scan(x_train, y_train, x_test, y_test):
    start_ts = datetime.datetime.now()
    log.info("Starting SVC model...")
    # Initialize SVC with non-linear kernel
    # rbf=Radial Basis Function (non-linear)
    svc_classifier = SVC(kernel='rbf')
    # Train the classifier
    svc_classifier.fit(x_train, y_train)

    train_score, test_score = svc_classifier.score(
        x_train, y_train), svc_classifier.score(x_test, y_test)
    # Predict on the test set
    y_pred = svc_classifier.predict(x_test)
    end_ts = datetime.datetime.now()
    log.info('SVC model finished. Elapsed time: %s',
             end_ts - start_ts)
    return y_pred, train_score, test_score


def MLP_Scan(x_train, y_train, x_test, y_test):
    # Initialize MLP Classifier
    mlp_classifier = MLPClassifier(hidden_layer_sizes=(
        100, 50), max_iter=500)  # Example parameters
    # Train the classifier
    model_filename = trained_model_folder+"mlp_model.joblib"
    if os.path.exists(model_filename):
        print("Loading existing MLP model...")
        mlp_classifier = load(model_filename)
        print("MLP Model loaded successfully.")
    else:
        # If the model doesn't exist, train a new one
        mlp_classifier.fit(x_train, y_train)
        dump(mlp_classifier, model_filename)

    train_score, test_score = mlp_classifier.score(
        x_train, y_train), mlp_classifier.score(x_test, y_test)
    # Predict on the test set
    y_pred = mlp_classifier.predict(x_test)
    return y_pred, train_score, test_score


def SGD_Scan(x_train, y_train, x_test, y_test):
   # Initialize SGD Classifier
    sgd_classifier = SGDClassifier(random_state=42)
    # Train the classifier
    model_filename = trained_model_folder+"sgd_model.joblib"
    if os.path.exists(model_filename):
        print("Loading existing SGD model...")
        sgd_classifier = load(model_filename)
        print("SGD Model loaded successfully.")
    else:
        # If the model doesn't exist, train a new one
        sgd_classifier.fit(x_train, y_train)
        dump(sgd_classifier, model_filename)
    train_score, test_score = sgd_classifier.score(
        x_train, y_train), sgd_classifier.score(x_test, y_test)

    # Predict on the test set
    y_pred = sgd_classifier.predict(x_test)
    return y_pred, train_score, test_score


def Logistic_regression_Scan(x_train, y_train, x_test, y_test):
   # Initialize SGD Classifier
    logreg = LogisticRegression(random_state=42)
    model_filename = trained_model_folder+"logr_model.joblib"
    if os.path.exists(model_filename):
        print("Loading existing LOGR model...")
        logreg = load(model_filename)
        print("LOGR Model loaded successfully.")
    else:
        # If the model doesn't exist, train a new one
        logreg.fit(x_train, y_train)
        dump(logreg, model_filename)

    train_score, test_score = logreg.score(
        x_train, y_train), logreg.score(x_test, y_test)
    # Predict on the test set
    y_pred = logreg.predict(x_test)
    return y_pred, train_score, test_score


def Linear_regression_Scan(x_train, y_train, x_test, y_test):
    # Initialize Linear Regression model
    linear_reg_classifier = LinearRegression()
    model_filename = trained_model_folder+"lrg_model.joblib"
    # Train the classifier
    if os.path.exists(model_filename):
        print("Loading existing Linear Regression model...")
        linear_reg_classifier = load(model_filename)
        print("Linear Regression Model loaded successfully.")
    else:
        # If the model doesn't exist, train a new one
        linear_reg_classifier.fit(x_train, y_train)
        dump(linear_reg_classifier, model_filename)

    train_score, test_score = linear_reg_classifier.score(
        x_train, y_train), linear_reg_classifier.score(x_test, y_test)
    # Predict probabilities on the test set
    y_pred_prob = linear_reg_classifier.predict(x_test)

    # Convert probabilities to binary predictions using a threshold
    threshold = 0.5
    y_pred_binary = (y_pred_prob >= threshold).astype(int)
    return y_pred_binary, train_score, test_score


def CNN_Scan(x_train, y_train, x_test, y_test):
    start_ts = datetime.datetime.now()
    print("Starting CNN model...")

    print(x_train)
    # Reshape the input data for CNN
    x_train_reshaped = np.expand_dims(x_train, axis=-1)
    print("X_train completed")
    x_test_reshaped = np.expand_dims(x_test, axis=-1)

    # print(x_train_reshaped)
    # Define the input shape
    input_shape = x_train_reshaped.shape[1:]
    input_shape = (64, 1)
    # Define the input layer
    inputs = Input(shape=input_shape)
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        GlobalAveragePooling1D(),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(x_train_reshaped, y_train, epochs=10, batch_size=32, verbose=1)
    # Evaluate the model
    train_score = model.evaluate(x_train_reshaped, y_train, verbose=0)
    test_score = model.evaluate(x_test_reshaped, y_test, verbose=0)
    # Predict on the test set
    y_pred_prob = model.predict(x_test_reshaped)
    y_pred_binary = (y_pred_prob >= 0.5).astype(int)
    end_ts = datetime.datetime.now()
    log.info('CNN model finished. Elapsed time: %s',
             end_ts - start_ts)
    return y_pred_binary.flatten(), train_score, test_score
