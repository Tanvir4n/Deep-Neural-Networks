import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

# Load Breast Cancer Dataset (e.g., from scikit-learn or CSV)
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X, y = data.data, data.target

# Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape data for CNN input (e.g., [samples, height, width, channels])
# Assuming each feature is treated like a pixel in a 1D image
X = X.reshape(-1, int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1])), 1)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 1: Deep Learning Model (CNN for Feature Extraction)
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(X.shape[1], X.shape[2], 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')  # 2 classes (benign/malignant)
])

cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Extract Features from CNN
feature_extractor = Sequential(cnn_model.layers[:-1])  # Remove the output layer
X_train_features = feature_extractor.predict(X_train)
X_test_features = feature_extractor.predict(X_test)

# Step 2: Machine Learning Model (Random Forest Classifier)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_features, y_train)

# Predict and Evaluate
y_pred = rf_model.predict(X_test_features)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Hybrid Model Accuracy:", accuracy_score(y_test, y_pred))
