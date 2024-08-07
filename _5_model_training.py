import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Step 1: Load the data
data = pd.read_csv('C:\\Song_Success_Predictor\\pre_processing\\selected_columns.csv')

# Step 2: Prepare data
X = data['encoded_lyrics'].values
y = data['rating'].values

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = [np.fromstring(x[1:-1], dtype=float, sep=', ') for x in X_train]
X_test = [np.fromstring(x[1:-1], dtype=float, sep=', ') for x in X_test]

# Step 4: Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Build the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(len(X_train[1]),)),
    Dense(32, activation='relu'),
    Dense(6, activation='softmax')  # Output layer for classification (6 neurons for 6 classes)
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 6: Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=2)

# Step 7: Evaluate the model
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print("Accuracy on Test Data:", accuracy)

model.save('C:\\Song_Success_Predictor\\trained_model.keras') 