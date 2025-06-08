import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split

# Configs
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = '/content/drive/MyDrive/binary_data'  # contains 'normal' and 'abnormal'
CSV_PATH = '/content/drive/MyDrive/binary_data/HAM10000_metadata_balanced.csv'
EPOCHS = 10

# Load metadata
df = pd.read_csv(CSV_PATH)

# Filter for images in binary_data dir
all_filenames = []
for label in ['normal', 'abnormal']:
    label_dir = os.path.join(DATA_DIR, label)
    files = [f for f in os.listdir(label_dir) if f.endswith('.jpg')]
    all_filenames.extend(files)

# Filter dataframe
df = df[df['image_id'].apply(lambda x: f'{x}.jpg' in all_filenames)].copy()

# Label binary target
df['binary_label'] = df['dx'].apply(lambda x: 1 if x in ['mel', 'bcc', 'akiec'] else 0)

# Encode metadata
df['sex'] = LabelEncoder().fit_transform(df['sex'].fillna('unknown'))
df['localization'] = OneHotEncoder(sparse_output=False).fit_transform(df['localization'].fillna('unknown').values.reshape(-1, 1)).tolist()
df['age'] = df['age'].fillna(df['age'].mean()) / 100.0  # Normalize

from tqdm import tqdm

def load_data(df):
    images, ages, sexes, locs, labels = [], [], [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        label = 'abnormal' if row['binary_label'] == 1 else 'normal'
        path = os.path.join(DATA_DIR, label, row['image_id'] + '.jpg')
        img = load_img(path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        images.append(img_array)
        ages.append(row['age'])
        sexes.append(row['sex'])
        locs.append(row['localization'])
        labels.append(row['binary_label'])
    return (
        np.array(images),
        np.array(ages).reshape(-1, 1),
        np.array(sexes).reshape(-1, 1),
        np.array(locs),
        np.array(labels)
    )

images, ages, sexes, locs, labels = load_data(df)

# Train/val split
X_train, X_val, age_train, age_val, sex_train, sex_val, loc_train, loc_val, y_train, y_val = train_test_split(
    images, ages, sexes, locs, labels, test_size=0.2, stratify=labels, random_state=42
)

# Build model
img_input = layers.Input(shape=IMG_SIZE + (3,))
age_input = layers.Input(shape=(1,))
sex_input = layers.Input(shape=(1,))
loc_input = layers.Input(shape=(locs.shape[1],))

base = EfficientNetV2B0(include_top=False, weights='imagenet', input_tensor=img_input)
x = layers.GlobalAveragePooling2D()(base.output)

meta = layers.Concatenate()([age_input, sex_input, loc_input])
meta = layers.Dense(32, activation='relu')(meta)

combined = layers.Concatenate()([x, meta])
combined = layers.Dense(64, activation='relu')(combined)
combined = layers.Dropout(0.5)(combined)
output = layers.Dense(1, activation='sigmoid')(combined)

model = Model(inputs=[img_input, age_input, sex_input, loc_input], outputs=output)

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(
    [X_train, age_train, sex_train, loc_train], y_train,
    validation_data=([X_val, age_val, sex_val, loc_val], y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

model.save('binary_skin_classifier.keras')  # ✅ Best option (saves everything)
model.save('binary_skin_classifier.h5')  # Explicit legacy formatQ