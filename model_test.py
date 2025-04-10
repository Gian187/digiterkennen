import numpy as np
import tkinter as tk
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Lade das trainierte Modell
model = load_model("model.h5")

def predict_image(image):
    image = image.resize((28, 28)).convert('L')  # Größe anpassen und in Graustufen konvertieren
    image = np.array(image)
    image = image / 255.0  # Normalisierung
    image = image.reshape(1, 28, 28, 1)  # Passende Form für das Modell

    prediction = model.predict(image)
    predicted_label = np.argmax(prediction)
    confidence = np.max(prediction)  # Höchste Wahrscheinlichkeit
    return predicted_label, confidence


class DrawApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Buchstaben erkennen")

        self.canvas = tk.Canvas(root, width=280, height=280, bg='white')
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)  # B1-Motion bedeuted self.draw aufrufen wenn Benutzer mit linker Maustaste über die Leinwand fährt

        self.button_predict = tk.Button(root, text="Erkennen", command=self.predict)
        self.button_predict.pack()

        self.button_clear = tk.Button(root, text="Löschen", command=self.clear_canvas)
        self.button_clear.pack()

        self.image = Image.new("RGB", (280, 280), "white")
        self.draw_image = ImageDraw.Draw(self.image)

    def draw(self, event):
        x, y = event.x, event.y     # Aktuelle Mauskoordianten
        self.canvas.create_oval(x, y, x + 8, y + 8, fill='black', width=5)      # Zeichnet Kreis an den aktuellen Koordinaten
        self.draw_image.ellipse([x, y, x + 8, y + 8], fill='black')         # Speichert Bild in PIL Imgae für spätere Verarbeitung

    def predict(self):
        predicted_label, confidence = predict_image(self.image)
        letter = chr(predicted_label + ord('A'))

        plt.imshow(self.image.resize((28, 28)).convert('L'), cmap='gray')
        plt.title(f"Vorhersage: {letter} (Genauigkeit: {confidence:.2%})")
        plt.axis("off")
        plt.show()

    def clear_canvas(self):
        self.canvas.delete("all")       # Alles von der Leinwand löschen
        self.image = Image.new("RGB", (280, 280), "white")      # erstellt ein neues weißes PIL-Image
        self.draw_image = ImageDraw.Draw(self.image)        # Damit man wieder drauf zeichnen kann


root = tk.Tk()
app = DrawApp(root)
root.mainloop()
