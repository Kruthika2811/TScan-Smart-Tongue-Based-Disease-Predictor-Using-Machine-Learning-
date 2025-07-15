# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# import os

# train_path = 'dataset/'

# datagen = ImageDataGenerator(
#     rescale=1./255,
#     validation_split=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True
# )

# train_data = datagen.flow_from_directory(
#     train_path,
#     target_size=(128, 128),
#     batch_size=32,
#     class_mode='categorical',
#     subset='training'
# )

# val_data = datagen.flow_from_directory(
#     train_path,
#     target_size=(128, 128),
#     batch_size=32,
#     class_mode='categorical',
#     subset='validation'
# )

# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
#     MaxPooling2D(2, 2),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dense(3, activation='softmax')
# ])

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# callbacks = [
#     EarlyStopping(patience=5, restore_best_weights=True),
#     ModelCheckpoint('model/tongue_disease_model.h5', save_best_only=True)
# ]

# model.fit(train_data, epochs=10, validation_data=val_data, callbacks=callbacks)

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Create folder to save model
os.makedirs('model', exist_ok=True)

# Path to your dataset folder
train_path = 'dataset/'

# Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Load train and validation data
train_data = datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),  # MobileNetV2 default
    batch_size=4,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=4,
    class_mode='categorical',
    subset='validation'
)

# Load MobileNetV2 as base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze it

# Add classification head
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint('model/tongue_model.h5', save_best_only=True)
]

# Train
model.fit(train_data, validation_data=val_data, epochs=10, callbacks=callbacks)
