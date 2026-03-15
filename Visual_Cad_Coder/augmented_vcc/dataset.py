#Zeile 25-29 muss ergänzt werden

import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

classes = [
    "ExtrudeSide",
    "ExtrudeEnd",
    "CutSide",
    "CutEnd",
    "Fillet",
    "Chamfer",
    "RevolveSide",
    "RevolveEnd"
]#Die zu erkennenden Klassen der Faces

class CADCountDataset(Dataset):
    def __init__(self, image_dir, annotations_file, image_size=224, transform=None):
        image_dir = "data/images"
        #Ordner mit den Bildern
        annotations_file = "data/annotations.json"
        #JSON Datei mit Bildnamen und CountVektor
        image_size= None 
        #gewünschte Zielgröße der Bilder
        transform = None
        #Umwandlung von PIL-Bild in Tensor kann hier angegeben werden
        self.image_dir=Path(image_dir)
        self.annotations_file=Path(annotations_file)

        with open(self.annotations_file, "r", encoding="utf-8") as f:
            #Datei wird geöffnet und als Variable f gespeichert
            self.samples =json.load(f)
        if transform is None:
            #Transformoperation, wenn keine definiert wurde
            self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
                ])
        else:
            #optionale benutzerdefinierte Transformoperation
            self.transform = transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        sample = self.samples[index]
        #Bild mit Annotation gespeichert
        image_path=self.image_dir / sample["image"]
        #baut den vollständigen Pfad zum Bild
        image = Image.open(image_path).convert("RGB")
        #Bilder werden auf drei Kanäle vereinheitlicht
        if self.transform is not None:
            image = self.transform(image)
        counts = torch.tensor(sample["counts"], dtype=torch.float32)
        #Count Liste wird in einen Tensor umgewandelt
        return image, counts
