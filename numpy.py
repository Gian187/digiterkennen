import os
import cv2
import numpy as np


# Funktion zum Vorbereiten der Bilddaten
def prepare_data(folder, image_size=28):
    images = []  # Liste zum Speichern der verarbeiteten Bilder
    labels = []  # Liste zum Speichern der Labels für jedes Bild
    label_dict = {}  # Dictionary zur Speicherung der Zuordnung von Labels zu Ordnernamen

    # Durch alle Unterordner im angegebenen Ordner iterieren
    # Angenommen, jeder Unterordner enthält Bilder einer bestimmten Klasse
    for idx, subfolder in enumerate(sorted(os.listdir(folder))):
        subfolder_path = os.path.join(folder, subfolder)  # Pfad des Unterordners

        # Sicherstellen, dass es sich um einen Ordner handelt
        if os.path.isdir(subfolder_path):
            label_dict[idx] = subfolder  # Speichern des Ordners als Label-Zuordnung (idx -> Ordnername)

            # Durch alle Dateien im Unterordner iterieren
            for filename in sorted(os.listdir(subfolder_path)):
                filepath = os.path.join(subfolder_path, filename)  # Pfad der Bilddatei

                # Sicherstellen, dass es sich um eine Datei handelt
                if os.path.isfile(filepath):
                    # Bild mit OpenCV einlesen und in Graustufen umwandeln
                    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

                    # Bild auf die gewünschte Größe anpassen (image_size x image_size)
                    resized_img = cv2.resize(image, (image_size, image_size))

                    # Normalisierung der Bildwerte (Werte werden zwischen 0 und 1 skaliert)
                    normalized_img = resized_img / 255.0

                    # Falls ein CNN verwendet wird, den Kanal hinzufügen (Form: HxWx1)
                    normalized_img = normalized_img.reshape(image_size, image_size, 1)

                    # Das Bild und das zugehörige Label (Index des Unterordners) speichern
                    images.append(normalized_img)
                    labels.append(idx)  # Label ist der Index des Unterordners (z.B. 0 für "A", 1 für "B", etc.)

    # Konvertiere die Listen in NumPy-Arrays für die spätere Verwendung im Modell
    images_array = np.array(images, dtype=np.float32)  # Bilder als float32 Array
    labels_array = np.array(labels, dtype=np.int32)  # Labels als int32 Array

    # Speichern der Bilder, Labels und der Label-Zuordnung in Dateien (für spätere Nutzung)
    np.save("images.npy", images_array)  # Speichert die Bilder in einer .npy-Datei
    np.save("labels.npy", labels_array)  # Speichert die Labels in einer .npy-Datei
    np.save("label_dict.npy", label_dict)  # Speichert die Zuordnung der Labels zu Ordnernamen

    # Debugging-Ausgabe, um zu überprüfen, wie viele Bilder verarbeitet wurden
    print(f"Daten gespeichert: {len(images)} Bilder verarbeitet.")
    print(f"Shape der Bilder: {images_array.shape}")  # Form der Bilddaten (z.B. (1000, 28, 28, 1))
    print(f"Shape der Labels: {labels_array.shape}")  # Form der Label-Daten (z.B. (1000,))
    print(f"Label-Zuordnung: {label_dict}")  # Zeigt die Zuordnung von Label-Index zu Ordnernamen

    # Rückgabe der vorbereiteten Daten
    return images_array, labels_array, label_dict


# Hauptpfad zum Ordner, der die Daten enthält (Bilder in Unterordnern, die Buchstaben oder Zahlen repräsentieren)
folder_path = "C:/1BHEL_HTLInn/KISY/Python/Digit_Erkennen/BIGDATASET"

# Aufruf der Funktion zur Vorbereitung der Daten
images, labels, label_dict = prepare_data(folder_path, image_size=28)
