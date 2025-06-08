from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from io import BytesIO
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load model using SavedModel format
model = tf.saved_model.load('multiclass_skin_classifier')
infer = model.signatures["serving_default"]

# Class names
class_names = ['akiec', 'bcc', 'mel']

# Localization categories
loc_categories = ['abdomen', 'back', 'chest', 'ear', 'face', 'foot',
                  'hand', 'lower extremity', 'neck', 'scalp', 'trunk',
                  'upper extremity', 'unknown']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load and preprocess image
        image_file = request.files['image']
        image = Image.open(BytesIO(image_file.read())).convert('RGB')
        image = image.resize((224, 224))
        img_array = img_to_array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

        # Preprocess metadata
        age = float(request.form.get('age', 50)) / 100.0
        age_array = np.array([[age]], dtype=np.float32)

        sex = request.form.get('sex', 'unknown').lower()
        sex_map = {'female': 0, 'male': 1, 'unknown': 2}
        sex_array = np.array([[sex_map.get(sex, 2)]], dtype=np.float32)

        localization = request.form.get('localization', 'unknown').lower()
        loc_vector = np.zeros((1, len(loc_categories)), dtype=np.float32)
        index = loc_categories.index(localization) if localization in loc_categories else loc_categories.index('unknown')
        loc_vector[0, index] = 1.0

        # Make prediction
        output = infer(
            keras_tensor=tf.constant(img_array),
            keras_tensor_1=tf.constant(age_array),
            keras_tensor_2=tf.constant(sex_array),
            keras_tensor_3=tf.constant(loc_vector)
        )

        preds = list(output.values())[0].numpy()
        class_index = int(np.argmax(preds))
        confidence = float(preds[0][class_index])
        predicted_label = class_names[class_index]

        return jsonify({
            'prediction': predicted_label,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
