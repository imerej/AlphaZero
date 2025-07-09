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

'''
game = NDGame(6,2)
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
for k in range(1):

    argss = args
    player = 1

    model = ResNet(game, 4, 64, device)
    model.load_state_dict(torch.load( f"model_{k}_{argss['num_searches']}_{argss['num_iterations']}_{argss['num_selfPlay_iterations']}_{argss['num_epochs']}_{argss['nom']}.pt", map_location=device))


    model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    args = {
        'C': 2,
        'num_searches': 5,
        'dirichlet_epsilon': 0.,
        'dirichlet_alpha': 0.3
    }
    model.eval()

    mcts = MCTS(game, args, model)


    def joue_contre_rand(n_parties):
        for i in trange(n_parties):
            state = game.get_initial_state()
            player = 1
            while True:
                if player == -1:
                    valid_moves = game.get_valid_moves(state)
                    tab = game.get_valid_moves(state)
                    indices = np.where(tab == 1)[0]        # prend les indices valant 1
                    action = np.random.choice(indices)

                else:
                    neutral_state = game.change_perspective(state, player)
                    mcts_probs = mcts.search(neutral_state)
                    action = np.argmax(mcts_probs)

                state = game.get_next_state(state, action, player)

                value, is_terminal = game.get_value_and_terminated(state, action)
                if is_terminal:
                    if value == 1:
                        resultats.append(player)
                    else:
                        resultats.append(0)
                    break

                player = game.get_opponent(player)

    resultats = []
    joue_contre_rand(300)
    nb_zero = resultats.count(0)
    nb_un = resultats.count(1)
    nb_moins_un = resultats.count(-1)
    print("dans simulation ", k, "avec les args", a)
    print("Nombre de 1 :", nb_un)
    print("Nombre de -1 :", nb_moins_un)
    print("Nombre de 0 :", nb_zero)
    '''
