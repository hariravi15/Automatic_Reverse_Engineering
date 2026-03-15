#Hier muss Model ggf. von Tiny auf Small/Base/Large geändert werden (s.line 36)
import torch
import torch.nn as nn
import torchvision.models as models

classes = [
    "ExtrudeSide",
    "ExtrudeEnd",
    "CutSide",
    "CutEnd",
    "Fillet",
    "Chamfer",
    "RevolveSide",
    "RevolveEnd"
] #Die zu erkennenden Klassen der Faces

class CountHead(nn.Module):
    #Linear Head für Konvertierung des Features in Output
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        #Konstruktr der Elternklasse aufgerufen
        self.fc = nn.Linear(in_dim, out_dim)
        #Schicht erstellt, um lineare Transformation aus Eingabe- zum Ausgabeverktor zu berechnen 
        self.act = nn.Softplus()
        #Aktivierungsfunktion für nichtnegative Outputs
    
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return self.act(self.fc(x))
        #Vektor wird in reelle Counts umgewandelt
    
def init_convnext(device ="cuda"):
    #cuda für GPU Nutzung
    model = models.convnext_tiny(
        weights=models.ConvNeXt_Tiny_Weights.DEFAULT
    )
    #Erstmal nur das Tiny Model - kann zu small/base/large gewechselt werden,
    #indem das geladene Backbone und der Weights-Name geändert werden
    in_dim = model.classifier[2].in_features
    #Dimension im letzten Linear-Layer wird festgelegt 
    model.classifier[2] = CountHead(in_dim, len(classes))
    #CountHead wird in den letzten Linear-Layer gesetzt
    model = model.to(device)
    return model
if __name__=="__main__":
    #wird ausgeführt, wenn .py direkt gestartet wird, nicht wenn von einer anderen Datei importiert
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    #device wird festgelegt
    model = init_convnext(device)
    #print("Modell gestartet")
