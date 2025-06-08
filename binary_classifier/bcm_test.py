import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

print("Loading model...")
model = tf.keras.models.load_model('binary_skin_classifier.keras', compile=False)
infer = model.signatures['serving_default']
print("Model loaded.")

# === Preprocess Image ===
img_path = 'normal.jpg'
img = load_img(img_path, target_size=(224, 224))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# === Preprocess Metadata ===
age = np.array([[0.5]], dtype=np.float32)  # normalized age
sex = np.array([[1]], dtype=np.float32)    # 0=female, 1=male
loc = np.zeros((1, 12), dtype=np.float32)
loc[0, 3] = 1  # example localization one-hot

# === Predict using correct input names ===
output = infer(
    keras_tensor=tf.constant(img_array),
    keras_tensor_1=tf.constant(age),
    keras_tensor_2=tf.constant(sex),
    keras_tensor_3=tf.constant(loc)
)

# === Interpret Output ===
pred = list(output.values())[0].numpy()[0][0]
label = 'abnormal' if pred >= 0.5 else 'normal'

# === Output Results ===
result_text = f"Prediction: {label}\nProbability: {pred:.4f}\n"
print(result_text)

with open('result.txt', 'w') as f:
    f.write(result_text)
