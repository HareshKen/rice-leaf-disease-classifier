import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Input

def build_model(input_shape=(224, 224, 3), num_classes=3):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False

    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model, base_model

def fine_tune_model(base_model, layers_to_unfreeze=40):
    base_model.trainable = True
    for layer in base_model.layers[:-layers_to_unfreeze]:
        layer.trainable = False

def plot_confusion_matrix(cm, labels):
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(6, 4))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()

def predict_image(model, img_path, labels):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    class_label = labels[class_index]
    confidence = prediction[0][class_index]
    print(f"Prediction: {class_label} ({confidence:.2f} confidence)")

def main():
    print("🚀 Starting Improved Rice Leaf Disease Classifier...")

    image_size = (224, 224)
    batch_size = 16

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.3,
        shear_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_gen = datagen.flow_from_directory(
        'rice_leaf_dataset',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )

    val_gen = datagen.flow_from_directory(
        'rice_leaf_dataset',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )

    model, base_model = build_model(input_shape=(224, 224, 3), num_classes=3)

    model.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=True),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    y_val = val_gen.classes
    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=np.unique(y_val),
                                         y=y_val)
    class_weights = dict(enumerate(class_weights))
    print("Class Weights:", class_weights)

    early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=2, min_lr=1e-6)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        callbacks=[early_stop, reduce_lr],
        class_weight=class_weights
    )

    print("🔧 Fine-tuning MobileNetV2...")
    fine_tune_model(base_model, layers_to_unfreeze=60)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5, amsgrad=True),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history_ft = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        callbacks=[early_stop, reduce_lr],
        class_weight=class_weights
    )

    for key in history_ft.history:
        history.history[key].extend(history_ft.history[key])

    val_gen.reset()
    Y_pred = model.predict(val_gen)
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = val_gen.classes
    labels = list(val_gen.class_indices.keys())

    print("\n📄 Classification Report:\n", classification_report(y_true, y_pred, target_names=labels))
    cm = confusion_matrix(y_true, y_pred)
    print("🧮 Confusion Matrix:\n", cm)

    # Accuracy plot
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    plot_confusion_matrix(cm, labels)

    # Save model
    model.save("rice_leaf_model.keras")
    print("✅ Model saved as 'rice_leaf_model.keras'.")

    # Predict one image (optional)
    if os.path.exists("test_image.jpg"):
        predict_image(model, "test_image.jpg", labels)

if __name__ == "__main__":
    main()
