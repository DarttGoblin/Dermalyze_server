import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# === Static Inputs ===
image_path = 'akiec.jpg'
age = 60
sex = 'female'
localization = 'foot'

# === Preprocessing Settings ===
IMG_SIZE = (224, 224)

# === Preprocess Image ===
img = load_img(image_path, target_size=IMG_SIZE)
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# === Preprocess Age ===
age_array = np.array([[age / 100.0]], dtype=np.float32)

# === Preprocess Sex ===
sex_map = {'female': 0, 'male': 1, 'unknown': 2}
sex_array = np.array([[sex_map.get(sex, 2)]], dtype=np.float32)

# === Preprocess Localization ===
loc_categories = ['abdomen', 'back', 'chest', 'ear', 'face', 'foot',
                  'hand', 'lower extremity', 'neck', 'scalp', 'trunk',
                  'upper extremity', 'unknown']

loc_vector = np.zeros((1, len(loc_categories)), dtype=np.float32)
index = loc_categories.index(localization) if localization in loc_categories else loc_categories.index('unknown')
loc_vector[0, index] = 1.0

# === Load Model from SavedModel format ===
model = tf.saved_model.load('multiclass_skin_classifier')
infer = model.signatures["serving_default"]

# === Predict ===
output = infer(
    keras_tensor=tf.constant(img_array),
    keras_tensor_1=tf.constant(age_array),
    keras_tensor_2=tf.constant(sex_array),
    keras_tensor_3=tf.constant(loc_vector)
)

pred = list(output.values())[0].numpy()
class_index = np.argmax(pred)
confidence = pred[0][class_index]

class_names = ['akiec', 'bcc', 'mel']

print(f"Predicted class: {class_names[class_index]} (index {class_index})")
print(f"Confidence: {confidence:.4f}")