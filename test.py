# # test.py
# import os
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import load_model

# #  Define Paths
# MODEL_PATH = "alzheimers_detection_model.h5"
# CLASSES = ["Mild Dementia", "Moderate Dementia", "Non Demented", "Very mild Dementia"]
# IMG_HEIGHT, IMG_WIDTH = 224, 224

# #  Check if Model Exists
# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError(f" Model file '{MODEL_PATH}' not found! Train the model first.")

# #  Load Model
# model = load_model(MODEL_PATH)

# #  Preprocess Image
# def preprocess_image(image_path):
#     image = tf.io.read_file(image_path)
#     image = tf.image.decode_jpeg(image, channels=3)
#     image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
#     return tf.expand_dims(image / 255.0, axis=0)

# #  Predict Function
# def predict_image(image_path):
#     if not os.path.exists(image_path):
#         print(f" Error: Image '{image_path}' not found!")
#         return

#     image = preprocess_image(image_path)
#     prediction = model.predict(image)
#     predicted_class = np.argmax(prediction)
#     predicted_label = CLASSES[predicted_class]

#     # Display Image + Prediction
#     plt.imshow(plt.imread(image_path))
#     plt.title(f"Predicted: {predicted_label}")
#     plt.axis('off')
#     plt.show()

#     print(f" Prediction: {predicted_label}")

# #  Test an Image (Update path)
# test_image_path = "images.jpeg"
# predict_image(test_image_path)
# Import Libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

# Load Model
model = load_model("alzheimers_detection_model.h5")

# Define Classes
classes = ["Mild Dementia", "Moderate Dementia", "Non Demented", "Very mild Dementia"]

# Load Dataset (Same Preprocessing as Train)
IMG_HEIGHT, IMG_WIDTH = 224, 224

def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    return image / 255.0  

# ✅ Load Test Dataset (Fixed File Name Issue)
test_paths = np.load("test_paths.npy", allow_pickle=True)
test_labels = np.load("test_labels.npy", allow_pickle=True)

# ✅ Ensure Labels are Categorical Integers (Fixed Label Encoding Issue)
if test_labels.ndim > 1:
    test_labels = np.argmax(test_labels, axis=1)  # Convert one-hot back to integers

# Prepare TensorFlow Dataset
test_dataset = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
test_dataset = test_dataset.map(lambda x, y: (preprocess_image(x), y)).batch(32).prefetch(tf.data.AUTOTUNE)

# Evaluate Model
loss, accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Predictions
predictions = model.predict(test_dataset)
y_pred = np.argmax(predictions, axis=1)

# ✅ Print Classification Report
print("Classification Report:")
print(classification_report(test_labels, y_pred, target_names=classes))

# ✅ Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(test_labels, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

