import numpy as np
import tkinter as tk
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Lade das trainierte Modell, das später für die Vorhersage genutzt wird.
# Das Modell wurde vorher mit handgeschriebenen Buchstaben trainiert.
model = load_model("model.h5")


# Funktion zur Vorhersage eines Bildes (z.B. ein gezeichneter Buchstabe)
def predict_image(image):
    # Bildgröße auf 28x28 Pixel anpassen und es in Graustufen umwandeln
    image = image.resize((28, 28)).convert('L')  # 'L' bedeutet Graustufen
    image = np.array(image)  # Bild in ein numpy-Array umwandeln
    image = image / 255.0  # Bild normalisieren (Werte zwischen 0 und 1)
    image = image.reshape(1, 28, 28, 1)  # Das Bild wird für das Modell in die passende Form gebracht

    # Vorhersage des Modells durchführen
    prediction = model.predict(image)
    predicted_label = np.argmax(prediction)  # Der Index mit der höchsten Wahrscheinlichkeit
    confidence = np.max(prediction)  # Die höchste Wahrscheinlichkeit (Vertrauensniveau)
    return predicted_label, confidence  # Gibt das vorhergesagte Label und die Wahrscheinlichkeit zurück


# Klasse für die Zeichen-App
class DrawApp:
    def __init__(self, root):
        # Initialisierung der App, sie wird in einem Fenster (root) angezeigt
        self.root = root
        self.root.title("Buchstaben erkennen")  # Titel des Fensters

        # Erstelle ein Zeichenfeld (Canvas), auf dem der Benutzer schreiben kann
        self.canvas = tk.Canvas(root, width=280, height=280, bg='white')
        self.canvas.pack()  # Das Canvas wird im Fenster angezeigt
        self.canvas.bind("<B1-Motion>", self.draw)  # B1-Motion: Zeichnen bei gedrückter linker Maustaste

        # Button zur Vorhersage des gezeichneten Buchstabens
        self.button_predict = tk.Button(root, text="Erkennen", command=self.predict)
        self.button_predict.pack()

        # Button zum Löschen der Zeichenfläche
        self.button_clear = tk.Button(root, text="Löschen", command=self.clear_canvas)
        self.button_clear.pack()

        # Ein leeres Bild (weiß) wird erstellt, auf dem der Benutzer schreiben kann
        self.image = Image.new("RGB", (280, 280), "white")
        self.draw_image = ImageDraw.Draw(self.image)  # Ein Zeichenwerkzeug wird für das Bild erstellt

    # Diese Funktion wird aufgerufen, wenn der Benutzer mit der Maus über das Canvas fährt
    def draw(self, event):
        # Die aktuellen Koordinaten der Maus holen
        x, y = event.x, event.y
        # Zeichnet einen kleinen Kreis auf das Canvas
        self.canvas.create_oval(x, y, x + 8, y + 8, fill='black', width=5)
        # Speichert die gezeichneten Punkte im Bild
        self.draw_image.ellipse([x, y, x + 8, y + 8], fill='black')

    # Funktion zur Vorhersage des gezeichneten Buchstabens
    def predict(self):
        # Die Vorhersage des Modells durchführen
        predicted_label, confidence = predict_image(self.image)

        # Das vorhergesagte Label (Buchstabe) von der Zahl ableiten
        letter = chr(predicted_label + ord('A'))  # Umwandlung der Zahl in den entsprechenden Buchstaben

        # Das Bild wird auf 28x28 Pixel herunterskaliert und in Graustufen dargestellt
        plt.imshow(self.image.resize((28, 28)).convert('L'), cmap='gray')
        plt.title(f"Vorhersage: {letter} (Genauigkeit: {confidence:.2%})")  # Anzeige der Vorhersage und der Genauigkeit
        plt.axis("off")  # Achsen ausblenden
        plt.show()  # Bild und Vorhersage anzeigen

    # Funktion zum Löschen des Canvas (Bild wird zurückgesetzt)
    def clear_canvas(self):
        self.canvas.delete("all")  # Löscht alles vom Canvas
        # Ein neues leeres Bild erstellen, um wieder zu zeichnen
        self.image = Image.new("RGB", (280, 280), "white")
        self.draw_image = ImageDraw.Draw(self.image)  # Neues Zeichenwerkzeug für das neue Bild


# Hauptprogramm: Erstelle das Fenster (root) und starte die App
root = tk.Tk()  # Erstelle das Fenster
app = DrawApp(root)  # Instanziiere die Zeichen-App
root.mainloop()  # Starte die grafische Benutzeroberfläche
