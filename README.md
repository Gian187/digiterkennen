
# 🧠 Buchstabenerkennung mit CNN (A-Z)

Dieses Projekt erkennt handgeschriebene Buchstaben (A bis Z) mithilfe eines Convolutional Neural Networks (CNN). Die Daten werden zuerst in NumPy-Form gebracht, das Modell wird trainiert und anschließend kann man es in einer kleinen GUI testen.

---

## 📦 Benötigte Libraries

Bitte installiere folgende Python-Bibliotheken, falls sie noch nicht vorhanden sind:

```bash
pip install numpy opencv-python tensorflow scikit-learn matplotlib pillow
```

---

## 🗂️ Projektstruktur

```plaintext
.
├── numpy.py              # Wandelt Bilddaten in NumPy-Arrays um
├── train_algorithm.py    # Trainiert das CNN-Modell mit den vorbereiteten Daten
├── model.py              # Enthält den CNN-Architekturcode
├── model_test.py         # Test-GUI zum Zeichnen und Erkennen von Buchstaben
├── model.h5              # (Wird erstellt) Das trainierte Modell
└── images.npy, labels.npy, label_dict.npy  # (Werden erstellt) Trainingsdaten im NumPy-Format
```

---

## 🧭 Schritt-für-Schritt-Anleitung

### 1. Daten vorbereiten (`numpy.py`)

Stelle sicher, dass deine Bilder in Unterordnern (z.B. `A`, `B`, ..., `Z`) liegen. Jeder Ordner repräsentiert eine Klasse.

📍 **Pfade anpassen:**  
In `numpy.py`, Zeile 77:

```python
folder_path = "C:/1BHEL_HTLInn/KISY/Python/Digit_Erkennen/BIGDATASET"
```

🔁 Ändere den Pfad zu dem Ort, wo deine Bildordner gespeichert sind.

Dann ausführen:

```bash
python numpy.py
```

---

### 2. Modell trainieren (`train_algorithm.py`)

📍 **Optional: Pfad prüfen**  
In `train_algorithm.py`, Zeile 7:

```python
folder_path = r"C:/1BHEL_HTLInn/KISY/Python/Digit_Erkennen"
```

Der Pfad wird nur als Referenz verwendet und kann ignoriert werden, solange `images.npy` und `labels.npy` im gleichen Verzeichnis liegen.

Dann ausführen:

```bash
python train_algorithm.py
```

🔁 Das Modell wird trainiert und als `model.h5` gespeichert.

---

### 3. Modell testen & GUI nutzen (`model_test.py`)

Ein Fenster öffnet sich, in dem du Buchstaben zeichnen und erkennen lassen kannst.

📍 **Wichtig:** Stelle sicher, dass `model.h5` im gleichen Ordner wie `model_test.py` liegt.

Dann ausführen:

```bash
python model_test.py
```

---

## ✏️ Hinweis zur GUI

- Zeichne mit der Maus einen Buchstaben.
- Klicke auf **"Erkennen"**, um eine Vorhersage zu bekommen.
- Klicke auf **"Löschen"**, um neu zu zeichnen.

---

## ✅ Ergebnis

Das Modell erkennt gezeichnete Buchstaben mit einer Genauigkeit, die abhängig von der Qualität und Menge deiner Trainingsdaten ist. Die GUI zeigt das Ergebnis mit prozentualer Sicherheit an.

## Dieses Projekt darf nicht für kommerzielle Zwecke verwendet werden

---
