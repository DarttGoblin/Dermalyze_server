from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from io import BytesIO
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load both models
binary_model = tf.saved_model.load('models/binary_skin_classifier').signatures["serving_default"]
multiclass_model = tf.saved_model.load('models/multiclass_skin_classifier').signatures["serving_default"]

# Class names
binary_classes = ['normal', 'abnormal']
multi_classes = ['akiec', 'bcc', 'mel']

loc_categories = ['back', 'lower extremity', 'trunk', 'upper extremity', 'abdomen', 'face', 
                  'chest', 'foot', 'neck', 'scalp', 'hand', 'ear', 'genital', 'acral']

@app.route('/predict', methods=['POST'])
def predict_combined():
    try:
        # Load and preprocess image
        image_file = request.files['image']
        image = Image.open(BytesIO(image_file.read())).convert('RGB')
        image = image.resize((224, 224))
        img_array = img_to_array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

        # Metadata
        age = float(request.form.get('age', 50)) / 100.0
        sex = request.form.get('sex', '').lower()
        localization = request.form.get('localization', '').lower()

        sex_map = {'female': 0, 'male': 1}
        if sex not in sex_map:
            return jsonify({'error': f'Invalid sex value: {sex}'}), 400
        if localization not in loc_categories:
            return jsonify({'error': f'Invalid localization value: {localization}'}), 400

        age_array = np.array([[age]], dtype=np.float32)
        sex_array = np.array([[sex_map[sex]]], dtype=np.float32)
        loc_vector = np.zeros((1, len(loc_categories)), dtype=np.float32)
        loc_vector[0, loc_categories.index(localization)] = 1.0

        # Predict with binary model
        binary_output = binary_model(
            keras_tensor=tf.constant(img_array),
            keras_tensor_1=tf.constant(age_array),
            keras_tensor_2=tf.constant(sex_array),
            keras_tensor_3=tf.constant(loc_vector)
        )

        # binary_preds = list(binary_output.values())[0].numpy()
        # binary_index = int(np.argmax(binary_preds))
        # binary_conf = float(binary_preds[0][binary_index])
        # binary_label = binary_classes[binary_index]
        binary_preds = list(binary_output.values())[0].numpy()
        binary_conf = float(binary_preds[0][0])  # Confidence of 'abnormal'
        binary_label = 'abnormal' if binary_conf >= 0.5 else 'normal'
        confidence = binary_conf if binary_label == 'abnormal' else 1.0 - binary_conf

        # If prediction is normal
        if binary_label == 'normal':
            return jsonify({
                'abnormal': False,
                'prediction': binary_label,
                'confidence': binary_conf
            })

        # Else predict with multiclass model
        multi_output = multiclass_model(
            keras_tensor=tf.constant(img_array),
            keras_tensor_1=tf.constant(age_array),
            keras_tensor_2=tf.constant(sex_array),
            keras_tensor_3=tf.constant(loc_vector)
        )

        multi_preds = list(multi_output.values())[0].numpy()
        multi_index = int(np.argmax(multi_preds))
        multi_conf = float(multi_preds[0][multi_index])
        multi_label = multi_classes[multi_index]

        return jsonify({
            'abnormal': True,
            'prediction': multi_label,
            'confidence': multi_conf
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
