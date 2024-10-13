import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2


class CNNGPR:
    def __init__(self):
        # Inicializa o modelo CNN
        self.model = self._build_cnn_model()

    def _build_cnn_model(self):
        # Constrói e compila o modelo CNN
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
        base_model.trainable = False  # Congela as camadas do modelo base

        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(10, activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, train_dir, validation_dir, epochs=10, batch_size=32):
        # Configura os geradores de dados para treinamento e validação
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=30,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )

        validation_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(256, 256),
            batch_size=batch_size,
            class_mode='categorical',
            color_mode='grayscale'
        )

        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(256, 256),
            batch_size=batch_size,
            class_mode='categorical',
            color_mode='grayscale'
        )

        # Treina o modelo
        history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size
        )

        return history

    def predict(self, image):
        # Faz a predição em uma imagem
        image = cv2.resize(image, (256, 256))
        image = image.reshape(1, 256, 256, 1) / 255.0
        prediction = self.model.predict(image)
        return np.argmax(prediction[0])

    def save_model(self, filepath):
        # Salva o modelo treinado
        self.model.save(filepath)

    def load_model(self, filepath):
        # Carrega um modelo salvo
        self.model = tf.keras.models.load_model(filepath)

    def evaluate(self, test_dir, batch_size=32):
        # Avalia o modelo em um conjunto de testes
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(256, 256),
            batch_size=batch_size,
            class_mode='categorical',
            color_mode='grayscale'
        )

        results = self.model.evaluate(test_generator)
        return dict(zip(self.model.metrics_names, results))
