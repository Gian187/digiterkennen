# model.py

import tensorflow as tf
from tensorflow.keras import layers, models


def create_model(input_shape=(28, 28, 1), num_classes=26):
    """
    Erstellt ein Convolutional Neural Network (CNN) Modell für die Bildklassifikation.

    :param input_shape: Die Form der Eingabebilder (Höhe, Breite, Kanäle)
    :param num_classes: Die Anzahl der Ausgabeklassen (26 für Buchstaben A-Z)
    :return: Das Keras-Modell
    """

    model = models.Sequential()

    # 1. Convolutional Layer mit MaxPooling
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    # 2. Convolutional Layer mit MaxPooling
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # 3. Convolutional Layer mit MaxPooling
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Flatten Layer, um in das Dense Layer zu gehen
    model.add(layers.Flatten())

    # 1. Dense Layer
    model.add(layers.Dense(64, activation='relu'))

    # Dropout Layer zur Regularisierung (Vermeidung von Overfitting)
    model.add(layers.Dropout(0.5))

    # Ausgabe Layer für die Klassifikation
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Kompiliere das Modell
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model
