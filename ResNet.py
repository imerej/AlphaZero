import torch.nn as nn
import torch.nn.functional as F


# on choisit la conv suivant la dimension de la grille encodee
# (1D pour une ligne, 2D pour le morpion classique, 3D pour le 3x3x3...)
CONV = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
BATCHNORM = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}


class ResNet(nn.Module):
    def __init__(self, game, num_resBlocks, num_hidden, device):
        #nb de convolutional block ie longueur du backbone
        # et leur taille

        super().__init__()
        #super verifie que tout le system est correctement mis en place
        #ie il inititalise nn.module

        self.device = device

        n = len(game.get_encoded_shape())
        Conv, BatchNorm = CONV[n], BATCHNORM[n]
        # avant c'etait toujours conv2d sur la grille aplatie en (length, length^(dim-1))
        # du coup en 3D deux cases voisines dans le cube n'etaient pas forcement voisines pour le filtre

        self.startBlock = nn.Sequential(
            Conv(3, num_hidden, kernel_size=3, padding=1),
            #cest une couche convolutionnelle

            #input channels
            #le premier 3 permet de prendre trois grilles en parametre : j1, j2, vide

            #output channels
            #num hidden est le nb de filtres, de feature maps

            #kernel size donne la taille des filtres ie la taille de lendroit zoomé
            BatchNorm(num_hidden),
            #normalisation augmente la vitesee dentrainement askip
            nn.ReLU()
            #enleve les valeurs negatives
        )
        #nn.sequential permet de definir une sequence de fonctions a appliquer
        #ca cree donc un nouveau bloc comme suite des trois differents blocs

        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden, Conv, BatchNorm) for i in range(num_resBlocks)]
            #crée num_resBlocks residual blocks, chacun avec num_hidden filtres,
            #et les garde comme liste de modules a entrainer
            #comme partie principale du nn : comme backbone
        )

        self.policyHead = nn.Sequential(
            Conv(num_hidden, 32, kernel_size=3, padding=1),
            BatchNorm(32),
            nn.ReLU(),
            nn.Flatten(),
            # (batch_size, channels, *grille) -> (batch_size, channels * nb cases)
            nn.Linear(32 * game.action_size, game.action_size)
            #fully connected layer
        )#32 est arbitraire mais empiriquement pas mauvais

        self.valueHead = nn.Sequential(
            Conv(num_hidden, 3, kernel_size=3, padding=1),
            BatchNorm(3),
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
    def __init__(self, num_hidden, Conv=nn.Conv2d, BatchNorm=nn.BatchNorm2d):
        super().__init__()
        self.conv1 = Conv(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = BatchNorm(num_hidden)
        self.conv2 = Conv(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = BatchNorm(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x
        #on applique le saut : sortie = entree + F(entree)
