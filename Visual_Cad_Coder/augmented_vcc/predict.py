#bei trainiertem Modell line 52 anpassen
import torch
from PIL import Image
from torchvision import transforms
from model import init_convnext, classes #Modellinitialisierung und Klassen importiert

def load_image(image_path, image_size=224): #Funktion zum Laden und Vorverarbeiten des Bildes
    transform = transforms.Compose([ #Transformationskette
        transforms.Resize((image_size, image_size)), #Bild auf feste Größe 224x224 gebracht
        transforms.ToTensor() #PIL-Bild wird zu [3,H,W] Tensor (Format wegen Struktur des Bildes)
    ])
    image = Image.open(image_path).convert("RGB") #Bild wird geöffnet und auf drei Kanäle erzwungen
    image_tensor = transform(image) #Bild wird transformiert
    image_tensor = image_tensor.unsqueeze(0)  # Batch-Dimension wird am Anfang hinzugefügt: [B,3,H,W]
    return image_tensor #vorverarbeitete Bild zurückgegeben

def predict(image_path, checkpoint_path=None): #None als Standard, damit Funktion ohne trainierte Gewichte getestet werden kann
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = init_convnext(device=device)
    model.eval()

    if checkpoint_path is not None: #trainierte Gewichte werden geladen, wenn angegeben
        state_dict = torch.load(checkpoint_path, map_location=device) #geseicherter Modellzustand wird geladen
        model.load_state_dict(state_dict) #geladene Gewichte werden übertragen
        print(f"Checkpoint geladen: {checkpoint_path}")
    else:
        print("Kein Checkpoint angegeben - verwende aktuelles Modellgewicht.")

    image = load_image(image_path).to(device) #Bild wird geladen, transformiert und auf Device verschoben

    with torch.no_grad(): #Gradientenberechnung wird bei Prediction ausgeschaltet
        output = model(image)

    output = output.squeeze(0).cpu()
    #Batch-Dimension wird entfernt und Tensor wird auf CPU verschoben

    print("\nVorhergesagte Counts:")
    for class_name, value in zip(classes, output): #Namen werden mit Werten gepaart
        print(f"{class_name}: {value.item():.3f}") #reelle Vorhersagen werden ausgegeben 

    rounded_counts = torch.round(output).to(torch.int64) #Rundung zur nächsten ganzen Zahl

    print("\nGerundete Counts:")
    for class_name, value in zip(classes, rounded_counts):
        print(f"{class_name}: {value.item()}")

    print(f"\nGesamtsumme aller vorhergesagten Faces: {rounded_counts.sum().item()}")

if __name__ == "__main__":
    image_path = "test_image.png"
    checkpoint_path = None
    # checkpoint_path = "checkpoints/best_model.pth" sobald trainiertes Modell existiert
    predict(image_path=image_path, checkpoint_path=checkpoint_path)#Vorhersage gestartet
    