from NDGame import *
from ResNet import *
from MCTS import *
from AlphaZero import *





game = NDGame(3,3)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(game, 4, 64, device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)


args = {
    'C': 3,
    'num_searches': 20,
    'num_iterations': 3,
    'num_selfPlay_iterations': 200,
    'num_epochs': 4,
    'batch_size': 32,
    'temperature': 1.1,
    'dirichlet_epsilon': 1,
    'dirichlet_alpha': 3,
    'nom': "3x3x3"
}


alphaZero = AlphaZero(model, optimizer, game, args)
alphaZero.learn()

