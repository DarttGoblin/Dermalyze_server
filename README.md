🩺 Skin Lesion Classification Project
This project aims to build deep learning models for detecting and classifying skin lesions from images to support early skin cancer diagnosis.

🚀 Project Overview
We developed two models:

Binary Classification Model
Distinguishes between normal and abnormal skin lesions.

Multi-class Classification Model
Classifies images into multiple specific skin cancer classes (e.g., AKIEC, BCC, MEL, etc.).

📂 Dataset Structure
Datasets are large and excluded from the repository via .gitignore. The folder structure looks like this:

bash
Copier
Modifier
.other_files/ # Miscellaneous files (excluded)
binary_classifier/
binary_data/
abnormal/ # Images labeled as abnormal lesions
normal/ # Images labeled as normal lesions
multi_classifier/
multiclass_data/
akiec/ # Class AKIEC images
bcc/ # Class BCC images
mel/ # Class MEL images
... # Other skin cancer classes
Note: You must prepare and place your dataset manually in these folders.

🧠 Model Training Summary
Binary Classification Model
Trained for 10 epochs.

Training accuracy increased from ~66% to ~93%.

Validation accuracy stayed low (~50%), indicating overfitting.

Validation loss fluctuated, confirming poor generalization.

Multi-class Classification Model
Also trained for 10 epochs.

Training accuracy improved to ~97%.

Validation accuracy was unstable and low (~16% to 57%), also indicating overfitting.

Validation loss was inconsistent.

⚠️ Key Observations
Both models show strong learning on training data but fail to generalize well.

Likely causes:

Insufficient regularization (e.g., no/low dropout, weight decay).

Dataset diversity and size limitations.

Model complexity and training setup.

Lack of auxiliary input integration (like patient metadata).

🔧 How to Use This Project
Prepare your datasets and place them inside the folders described above.

Use your training scripts to train either the binary or multi-class model.

Monitor training and validation metrics to detect overfitting.

Experiment with techniques such as data augmentation, dropout, or improved model architectures.

📈 Future Improvements
Add data augmentation to increase dataset diversity.

Use stronger regularization techniques (dropout, L2 regularization).

Incorporate patient metadata (age, sex, skin region) for better predictions.

Explore advanced model architectures or multimodal models.

Balance dataset classes to reduce bias.

📝 Summary
This project demonstrates the challenges of training skin lesion classifiers and highlights the importance of data quality, diversity, and appropriate model design to avoid overfitting and improve real-world performance.
