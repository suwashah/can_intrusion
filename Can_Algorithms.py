
import tensorflow as tf
import logging as log
import datetime
import os
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Flatten
from tensorflow.keras.layers import Input


# Create a folder for log files if it doesn't exist
log_folder = 'logs'
os.makedirs(log_folder, exist_ok=True)

# Configure logging to save log file in the folder
log_file = os.path.join(log_folder, 'log_file.txt')

# Configure logging
log.basicConfig(filename=log_file,
                level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def SVC_Scan(x_train, y_train, x_test):
    start_ts = datetime.datetime.now()
    log.info("Starting SVC model...")
    # Initialize SVC with non-linear kernel
    # rbf=Radial Basis Function (non-linear)
    svc_classifier = SVC(kernel='rbf')
    # Train the classifier
    svc_classifier.fit(x_train, y_train)
    # Predict on the test set
    y_pred = svc_classifier.predict(x_test)
    end_ts = datetime.datetime.now()
    log.info('SVC model finished. Elapsed time: %s',
             end_ts - start_ts)
    return y_pred


def MLP_Scan(x_train, y_train, x_test):
    start_ts = datetime.datetime.now()
    log.info("Starting MLP model...")
    # Initialize MLP Classifier
    mlp_classifier = MLPClassifier(hidden_layer_sizes=(
        100, 50), max_iter=500)  # Example parameters

    # Train the classifier
    mlp_classifier.fit(x_train, y_train)
    # Predict on the test set
    y_pred = mlp_classifier.predict(x_test)
    end_ts = datetime.datetime.now()
    log.info('MLP model finished. Elapsed time: %s',
             end_ts - start_ts)
    return y_pred


def SGD_Scan(x_train, y_train, x_test):
    start_ts = datetime.datetime.now()
    log.info("Starting SGD model...")
   # Initialize SGD Classifier
    sgd_classifier = SGDClassifier()

    # Train the classifier
    sgd_classifier.fit(x_train, y_train)

    # Predict on the test set
    y_pred = sgd_classifier.predict(x_test)
    end_ts = datetime.datetime.now()
    log.info('SGD model finished. Elapsed time: %s',
             end_ts - start_ts)
    return y_pred


def Linear_regression_Scan(x_train, y_train, x_test):
    start_ts = datetime.datetime.now()
    log.info("Starting Linear regression model...")
    # Initialize Linear Regression model
    linear_reg_classifier = LinearRegression()

    # Train the classifier
    linear_reg_classifier.fit(x_train, y_train)

    # Predict probabilities on the test set
    y_pred_prob = linear_reg_classifier.predict(x_test)

    # Convert probabilities to binary predictions using a threshold
    threshold = 0.5
    y_pred_binary = (y_pred_prob >= threshold).astype(int)
    end_ts = datetime.datetime.now()
    log.info('Linear regression model finished. Elapsed time: %s',
             end_ts - start_ts)
    return y_pred_binary


def CNN_Scan(x_train, y_train, x_test):
    start_ts = datetime.datetime.now()
    log.info("Starting CNN model...")
    # Reshape the input data for CNN
    x_train_reshaped = np.expand_dims(x_train, axis=-1)
    x_test_reshaped = np.expand_dims(x_test, axis=-1)

    # Define the input shape
    input_shape = x_train_reshaped.shape[1:]

    # Define the input layer
    inputs = Input(shape=input_shape)

    # Define the CNN architecture
    model = Sequential([
        inputs,
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        GlobalAveragePooling1D(),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(x_train_reshaped, y_train, epochs=10, batch_size=32, verbose=0)

    # Predict on the test set
    y_pred_prob = model.predict(x_test_reshaped)
    y_pred_binary = (y_pred_prob >= 0.5).astype(int)
    end_ts = datetime.datetime.now()
    log.info('CNN model finished. Elapsed time: %s',
             end_ts - start_ts)
    return y_pred_binary.flatten()
