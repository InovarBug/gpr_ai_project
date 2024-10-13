import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
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
        input_img = tf.keras.Input(shape=(256, 256, 1))
        # Encoder
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        # Decoder
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        model = tf.keras.Model(input_img, decoded)
        model.compile(optimizer='adam', loss='binary_crossentropy')
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
            class_mode=None,
            color_mode='grayscale'
        )
        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(256, 256),
            batch_size=batch_size,
            class_mode=None,
            color_mode='grayscale'
        )

        # Treina o modelo
        print("Iniciando o treinamento do modelo")
        history = self.model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=len(validation_generator)
        )

        print("Salvando o modelo treinado como cnn_model_trained.h5")
        self.save_model('cnn_model_trained.h5')
        print("Modelo salvo com sucesso")
        return history

    def predict(self, image):
        # Faz a predição em uma imagem
        image = cv2.resize(image, (256, 256))
        image = image.reshape(1, 256, 256, 1) / 255.0
        prediction = self.model.predict(image)
        return prediction[0]

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
if __name__ == '__main__':
    cnn_gpr = CNNGPR()
    cnn_gpr.train(train_dir='data/train', validation_dir='data/validation', epochs=5, batch_size=2)
