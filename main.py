import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 1. LOAD DATASET
# =========================
def load_dataset(dataset_path):
    images = []
    labels = []
    class_names = os.listdir(dataset_path)

    for label, class_name in enumerate(class_names):
        class_folder = os.path.join(dataset_path, class_name)
        for file in os.listdir(class_folder):
            img_path = os.path.join(class_folder, file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # resize
            img = cv2.resize(img, (128, 128))
            
            # grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels), class_names


# =========================
# 2. HOG FEATURE EXTRACTION
# =========================
def extract_hog_features(images, pixels_per_cell=(8,8)):
    features = []
    for img in images:
        hog_feature = hog(
            img,
            orientations=9,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=(2,2),
            block_norm='L2-Hys'
        )
        features.append(hog_feature)
    return np.array(features)


# =========================
# 3. TRAIN + GRID SEARCH
# =========================
def train_model(X_train, y_train):
    param_grid = {
        'kernel': ['linear', 'rbf'],
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001]
    }

    model = SVC(decision_function_shape='ovo')

    grid = GridSearchCV(model, param_grid, cv=5, verbose=2, n_jobs=-1)
    grid.fit(X_train, y_train)

    print("\nBest Parameters:", grid.best_params_)
    return grid.best_estimator_


# =========================
# 4. EVALUASI
# =========================
def evaluate_model(model, X_test, y_test, class_names):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("\nAccuracy:", acc)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


# =========================
# 5. MAIN PIPELINE
# =========================
def run_experiment(dataset_path, pixels_per_cell=(8,8), test_size=0.2):
    print("\n=== Loading Dataset ===")
    images, labels, class_names = load_dataset(dataset_path)

    print("\n=== Extracting HOG Features ===")
    features = extract_hog_features(images, pixels_per_cell)

    print("\n=== Splitting Data ===")
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, stratify=labels, random_state=42
    )

    print("\n=== Training Model ===")
    model = train_model(X_train, y_train)

    print("\n=== Evaluating Model ===")
    evaluate_model(model, X_test, y_test, class_names)


# =========================
# 6. RUN 3 EXPERIMENTS
# =========================
if __name__ == "__main__":
    dataset_path = "dataset_bunga"

    print("\n========= PERCOBAAN 1 =========")
    run_experiment(dataset_path, pixels_per_cell=(8,8), test_size=0.3)

    print("\n========= PERCOBAAN 2 =========")
    run_experiment(dataset_path, pixels_per_cell=(8,8), test_size=0.2)

    print("\n========= PERCOBAAN 3 =========")
    run_experiment(dataset_path, pixels_per_cell=(16,16), test_size=0.2)