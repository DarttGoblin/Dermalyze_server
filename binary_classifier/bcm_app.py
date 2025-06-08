from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

print("Loading model...")
model = tf.keras.models.load_model('binary_skin_classifier.keras')
print("Model loaded successfully.")

@app.route('/predict', methods=['POST'])
def predict():
    # Log that a prediction request was received
    print("Received trigger from frontend")

    # Dummy hardcoded input (until real integration)
    img = load_img('normal.jpg', target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    age = np.array([[0.5]])  # normalized
    sex = np.array([[1]])    # female
    loc = np.zeros((1, 12))  # dummy one-hot region
    loc[0, 3] = 1            # example: 4th category active

    pred = model.predict([img_array, age, sex, loc])[0][0]
    label = 'abnormal' if pred >= 0.5 else 'normal'

    print(f"Prediction: {label}, Probability: {pred:.4f}")

    return jsonify({
        'probability': float(pred),
        'prediction': label
    })

if __name__ == '__main__':
    app.run(debug=True)