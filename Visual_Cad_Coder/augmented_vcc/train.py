#Aufgabe: Lädt Daten, splittet sie in Train/Val, erstellt DataLoader, initialisiert Modell
#trainiert über mehrere Epochen, bewertet mit Validierungsdaten und speichert letzte und beste Modell
import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from model import init_convnext
from dataset import CADCountDataset

def main():
    if torch.cuda.is_available(): #GPU wird genutzt falls cuda-fähig
        device = "cuda"
    else:
        device = "cpu"
    print(f"Verwendetes Device: {device}")

    image_dir="data/images" #Hier liegen Bilder
    annotations_file="data/annotations.json" #Dateinamen mit Labels in Datei

    batch_size=8 #Anzahl der gleichzeitig verarbeiteten Bildern (Training in Patches und nicht Einzelbildern)
    num_epochs=10 #Anzahl der Trainierdurchläufe
    learning_rate=1e-4 #Gibt an, wie stark Gewichte pro Schritt angepasst werden
    weight_decay=1e-4 #Je höher, desto stärker werden hohe Gewichte bestraft
    val_split=0.2 #Prozentsatz, der für Validierung genutzt wird
    random_seed=42 #tbc
    num_workers=0 #Anzahl der Hintergrundprozesse, die Daten laden (0 für Stabilität)

    save_dir=Path("checkpoints") #Hier sollen Modellgewichte gespeichert werden
    save_dir.mkdir(parents=True, exist_ok=True) #(Parent-)Ordner werden ggf. erstellt
    print("Lade Dataset...")
    full_dataset=CADCountDataset( #Dataset Objekt erzeugt
        #JSON wird geladen, Bildpfade vorbereitet und Transform wird gesetzt
        image_dir=image_dir,
        annotations_file=annotations_file,
        image_size=224,
    )
    dataset_size=len(full_dataset) #Anzahl Samples im ganzen Dataset
    val_size=int(dataset_size*val_split)
    train_size=dataset_size-val_size

    generator=torch.Generator().manual_seed(random_seed) 
    #Zufallsgenerator mit festem Seed für reproduzierbaren Split
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator
    )
    print(f"Gesamte Samples_ {dataset_size}")
    print(f"Train-Samples: {train_size}")
    print(f"Val-Samples: {val_size}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True, #Trainingsdaten werden pro Epoche gemischt
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False, #Bei Validierung ist Reihenfolge egal
        num_workers=num_workers
    )
    print("Initialisiere Modell...")
    model=init_convnext(device=device)
    loss_fn=torch.nn.SmoothL1Loss()
    optimizer=torch.optim.AdamW( 
        #Optimizer definiert, wie Modellgewichte angepasst werden
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    best_val_loss=float("inf") #unendlich
    for epoch in range(num_epochs):
        model.train() #Modell wird in Trainingsmodus gesetzt
        train_loss_sum=0.0
        for images, targets in train_loader: 
            #Iteration über alle Trainingsbatches, images->Bild und targets->Count-Vektoren
            images = images.to(device)
            targets = targets.to(device)
            outputs=model(images) #prediction wird berechnet
            loss=loss_fn(outputs, targets) #Fehler zwischen prediciton und ground truth
            optimizer.zero_grad() #Gradienten zurückgesetzt
            loss.backward() #Gradienten der trainierbaren Parameter berechnet
            optimizer.step() #Gewichte werden mit Gradienten aktualisiert
            train_loss_sum += loss.item()*images.size(0)
        avg_train_loss = train_loss_sum/len(train_loader.dataset)
        model.eval() #Modell wird in Evaluationsmodus gesetzt
        val_loss_sum = 0.0
        with torch.no_grad(): #Gradienten innerhalb des Blocks werden nicht aktualisiert
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)

                outputs = model(images)
                loss = loss_fn(outputs, targets)

                val_loss_sum += loss.item() * images.size(0)
        avg_val_loss=val_loss_sum/len(val_loader.dataset)
        print(
            f"Epoch [{epoch+1}/{num_epochs}]"
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )
        last_model_path = save_dir / "last_model.pth"
        torch.save(model.state_dict(), last_model_path)

        if avg_val_loss<best_val_loss:
            best_val_loss=avg_val_loss
            best_model_path = save_dir/"best_model.pth"
            torch.save(model.state_dict(), best_model_path) #Gewichte des besten paths gespeichert
            print(f"Bestes Modell gespeichert: {best_model_path}")
    print("Training abgeschlossen.")
    print(f"Letztes Modell: {save_dir/'last_model.pth'}")
    print(f"Bestes Modell: {save_dir / 'best_model.pth'}")

if __name__ == "__main__": 
    #Damit bei Import nicht sofort Initialisierung und Training gestartet wird
    main()
