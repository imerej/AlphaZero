import torch.nn as nn
import torch.nn.functional as F



class ResNet(nn.Module):
    def __init__(self, game, num_resBlocks, num_hidden, device):
        #nb de convolutional block ie longueur du backbone
        # et leur taille
        
        super().__init__()
        #super verifie que tout le system est correctement mis en place
        #ie il inititalise nn.module
        
        self.device = device
        
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            #cest une couche convolutionnelle
            
            #input channels
            #le premier 3 permet de prendre trois grilles en parametre : j1, j2, vide
            
            #output channels
            #num hidden est le nb de filtres, de feature maps
            
            # idee : implementer en n dimension ?
            
            #kernel size donne la taille des filtres ie la taille de lendroit zoomé
            nn.BatchNorm2d(num_hidden),
            #normalisation augmente la vitesee dentrainement askip
            nn.ReLU()
            #enleve les valeurs negatives
        )
        #nn.sequential permet de definir une sequence de fonctions a appliquer
        #ca cree donc un nouveau bloc comme suite des trois differents blocs
        
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
            #crée num_resBlocks residual blocks, chacun avec num_hidden filtres,
            #et les garde comme liste de modules a entrainer
            #comme partie principale du nn : comme backbone
        )
        
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            # 4D : (batch_size, channels, height, width) -> 1D
            nn.Linear(32 * game.action_size, game.action_size)
            #fully connected layer
        )#32 est arbitraire mais empiriquement pas mauvais
        
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.action_size, 1),
            nn.Tanh()
        )
        
        self.to(device)
        #deplace le modele vers le device : GPU/CPU
        
    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value
    
class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x
        #on applique le saut : sortie = entree + F(entree) 

