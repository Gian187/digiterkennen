import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from model import create_model  # Importiere das Modell aus model.py

# Pfad zum Hauptordner mit den Unterordnern A-Z
folder_path = r"C:/1BHEL_HTLInn/KISY/Python/Digit_Erkennen"
img_size = 28  # Skalierung auf 28x28 Pixel

# Lade und verarbeite die Bilder und Labels
images_array = np.load("images.npy")
labels_array = np.load("labels.npy")

# Normalisiere die Bilder
images_array = images_array.reshape(-1, 28, 28, 1)  # Anpassung auf die Eingabeform für CNNs
images_array = images_array.astype('float32')  # Um sicherzustellen, dass die Werte als float32 vorliegen
labels_array = labels_array.astype('int32')  # Um sicherzustellen, dass die Labels als Integer vorliegen

# Splitte die Daten in Trainings- und Testdaten
x_train, x_test, y_train, y_test = train_test_split(images_array, labels_array, test_size=0.1, random_state=42)

# Erstelle das Modell
model = create_model(input_shape=(28, 28, 1), num_classes=26)

# Trainiere das Modell
model.fit(x_train, y_train, epochs=30, validation_data=(x_test, y_test))

# Teste das Modell
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Genauigkeit: {test_acc * 100:.2f}%")

# Speichern des Modells
model.save('model.h5')
print(f"Modell gespeichert")
