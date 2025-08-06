import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Load the LFW dataset
lfw_data = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# Features and labels
X = lfw_data.images
y = lfw_data.target
target_names = lfw_data.target_names
n_classes = len(target_names)

# Normalize and reshape input data
X = X.reshape(-1, X.shape[1], X.shape[2], 1) / 255.0

# One-hot encode the target
y = to_categorical(y, n_classes)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=X.shape[1:]),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(n_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Function to predict and display a face
def predict_and_display(index):
    sample = X_test[index].reshape(1, X.shape[1], X.shape[2], 1)
    prediction = model.predict(sample)
    predicted_label = target_names[np.argmax(prediction)]
    actual_label = target_names[np.argmax(y_test[index])]
    
    plt.imshow(X_test[index].reshape(X.shape[1], X.shape[2]), cmap='gray')
    plt.title(f"Predicted: {predicted_label}\nActual: {actual_label}")
    plt.axis('off')
    plt.show()

# Show prediction result for one test sample
predict_and_display(5)

# Save the trained model
model.save("lfw_face_recognition_cnn.h5")
