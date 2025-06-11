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

# Binary model localization categories (original)
binary_loc_categories = ['back', 'lower extremity', 'trunk', 'upper extremity', 'abdomen', 'face', 
                         'chest', 'foot', 'neck', 'scalp', 'hand', 'ear', 'genital', 'acral']

# Multiclass model localization categories (fixed to match working test)
multi_loc_categories = ['abdomen', 'back', 'chest', 'ear', 'face', 'foot',
                        'hand', 'lower extremity', 'neck', 'scalp', 'trunk',
                        'upper extremity', 'unknown']

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

        age_array = np.array([[age]], dtype=np.float32)

        # === BINARY MODEL PREDICTION ===
        # Sex preprocessing for binary model (original way)
        sex_map_binary = {'female': 0, 'male': 1}
        if sex not in sex_map_binary:
            return jsonify({'error': f'Invalid sex value: {sex}'}), 400
        sex_array_binary = np.array([[sex_map_binary[sex]]], dtype=np.float32)

        # Localization for binary model (original way)
        if localization not in binary_loc_categories:
            return jsonify({'error': f'Invalid localization value: {localization}'}), 400
        binary_loc_vector = np.zeros((1, len(binary_loc_categories)), dtype=np.float32)
        binary_loc_vector[0, binary_loc_categories.index(localization)] = 1.0

        # Predict with binary model
        binary_output = binary_model(
            keras_tensor=tf.constant(img_array),
            keras_tensor_1=tf.constant(age_array),
            keras_tensor_2=tf.constant(sex_array_binary),
            keras_tensor_3=tf.constant(binary_loc_vector)
        )

        binary_preds = list(binary_output.values())[0].numpy()
        binary_conf = float(binary_preds[0][0])  # Confidence of 'abnormal'
        binary_label = 'abnormal' if binary_conf >= 0.5 else 'normal'

        # If prediction is normal
        if binary_label == 'normal':
            return jsonify({
                'abnormal': False,
                'prediction': binary_label,
                'confidence': 1.0 - binary_conf  # Confidence of being normal
            })

        # === MULTICLASS MODEL PREDICTION ===
        # Sex preprocessing for multiclass model (with unknown support)
        sex_map_multi = {'female': 0, 'male': 1, 'unknown': 2}
        sex_array_multi = np.array([[sex_map_multi.get(sex, 2)]], dtype=np.float32)

        # Localization for multiclass model (with unknown fallback)
        multi_loc_vector = np.zeros((1, len(multi_loc_categories)), dtype=np.float32)
        multi_index = multi_loc_categories.index(localization) if localization in multi_loc_categories else multi_loc_categories.index('unknown')
        multi_loc_vector[0, multi_index] = 1.0

        print("DEBUG MULTICLASS SHAPES →")
        print("  img_array:", img_array.shape)
        print("  age_array:", age_array.shape)
        print("  sex_array_multi:", sex_array_multi.shape)
        print("  multi_loc_vector:", multi_loc_vector.shape)
        print("  localization used:", localization if localization in multi_loc_categories else 'unknown')

        # Predict with multiclass model
        multi_output = multiclass_model(
            keras_tensor=tf.constant(img_array),
            keras_tensor_1=tf.constant(age_array),
            keras_tensor_2=tf.constant(sex_array_multi),
            keras_tensor_3=tf.constant(multi_loc_vector)
        )

        multi_preds = list(multi_output.values())[0].numpy()
        multi_class_index = int(np.argmax(multi_preds))
        multi_conf = float(multi_preds[0][multi_class_index])
        multi_label = multi_classes[multi_class_index]

        return jsonify({
            'abnormal': True,
            'prediction': multi_label,
            'confidence': multi_conf
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)