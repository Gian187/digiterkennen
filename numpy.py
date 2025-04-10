import os
import cv2
import numpy as np


def prepare_data(folder, image_size=28):
    images = []
    labels = []
    label_dict = {}  # Dictionary zur Speicherung der Label-Zuordnung

    # Durch alle Unterordner iterieren
    for idx, subfolder in enumerate(sorted(os.listdir(folder))):
        subfolder_path = os.path.join(folder, subfolder)

        if os.path.isdir(subfolder_path):
            label_dict[idx] = subfolder  # Speichere die Label-Zuordnung

            for filename in sorted(os.listdir(subfolder_path)):
                filepath = os.path.join(subfolder_path, filename)

                if os.path.isfile(filepath):
                    # Bild einlesen und in Graustufen konvertieren
                    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

                    # Bild skalieren auf das richtige Format
                    resized_img = cv2.resize(image, (image_size, image_size))

                    # Normalisieren (Werte zwischen 0 und 1)
                    normalized_img = resized_img / 255.0

                    # Falls CNN: Kanal hinzufügen (Height, Width, 1)
                    normalized_img = normalized_img.reshape(image_size, image_size, 1)

                    images.append(normalized_img)
                    labels.append(idx)  # Verwenden des Index als Label

    # Konvertiere in NumPy-Arrays
    images_array = np.array(images, dtype=np.float32)
    labels_array = np.array(labels, dtype=np.int32)  # Flaches Array

    # Speichern der Dateien
    np.save("images.npy", images_array)
    np.save("labels.npy", labels_array)
    np.save("label_dict.npy", label_dict)  # Speichern der Label-Zuordnung

    print(f"Daten gespeichert: {len(images)} Bilder verarbeitet.")
    print(f"Shape der Bilder: {images_array.shape}")  # Debug-Info
    print(f"Shape der Labels: {labels_array.shape}")  # Debug-Info
    print(f"Label-Zuordnung: {label_dict}")

    return images_array, labels_array, label_dict


folder_path = "C:/1BHEL_HTLInn/KISY/Python/Digit_Erkennen/BIGDATASET"
images, labels, label_dict = prepare_data(folder_path, image_size=28)
