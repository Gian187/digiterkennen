import cv2
import numpy as np
import os
from PIL import Image


def extract_letters(image_path, output_folder):
    # Bild laden
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Konturen finden
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ordner für Ausgabe erstellen
    os.makedirs(output_folder, exist_ok=True)

    letter_index = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Zu kleine oder große Konturen überspringen
        if w < 10 or h < 10:
            continue

        # Buchstaben extrahieren
        letter_image = image[y:y + h, x:x + w]

        # Bild speichern
        letter_pil = Image.fromarray(letter_image)
        letter_pil.save(os.path.join(output_folder, f'Briola_{letter_index:02d}.png'))
        letter_index += 1

    print(f"{letter_index} Buchstaben gespeichert in {output_folder}")


# Beispielaufruf
extract_letters("page_1.png", "output_letters")