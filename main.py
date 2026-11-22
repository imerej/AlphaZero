import argparse

from NDGame import *
from ResNet import *
from MCTS import *
from AlphaZero import *
from minmax import MinMax


# exemples :
#   python main.py train --length 3 --dim 3
#   python main.py eval --length 3 --dim 3 --model checkpoints/model_2_60_3_200_4_3x3x3_k3.pt
#   python main.py play --length 3 --dim 3 --model ...
#   python main.py eval --length 3 --dim 3 --model ... --adversaire minmax
#   python main.py play --length 3 --dim 3 --joueur minmax


def get_parser():
    parser = argparse.ArgumentParser(description="alphazero sur le morpion en n dimensions")
    parser.add_argument('mode', choices=['train', 'eval', 'play'])
    parser.add_argument('--length', type=int, default=3, help="cote de l'hypercube")
    parser.add_argument('--dim', type=int, default=3)
    parser.add_argument('--k', type=int, default=None, help="nb a aligner (par defaut = length)")

    parser.add_argument('--res_blocks', type=int, default=4)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--model', type=str, default=None, help="fichier .pt a charger")
    parser.add_argument('--optimizer', type=str, default=None, help="pour reprendre un entrainement")

    parser.add_argument('--C', type=float, default=2)
    parser.add_argument('--searches', type=int, default=60)
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--selfplay', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--temperature', type=float, default=1.25)
    parser.add_argument('--epsilon', type=float, default=0.25, help="poids du bruit de dirichlet")
    parser.add_argument('--alpha', type=float, default=0.3, help="parametre du bruit de dirichlet")
    parser.add_argument('--max_symmetries', type=int, default=None,
                        help="nb de symetries gardees par position (par defaut toutes)")
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--parties', type=int, default=100, help="nb de parties pour eval")
    parser.add_argument('--first', action='store_true', help="en mode play, l'humain commence")
    parser.add_argument('--joueur', choices=['alphazero', 'minmax'], default='alphazero',
                        help="l'ia evaluee (eval) ou contre qui on joue (play)")
    parser.add_argument('--adversaire', choices=['rand', 'minmax'], default='rand', help="adversaire en mode eval")
    parser.add_argument('--profondeur', type=int, default=2, help="profondeur du minmax")
    parser.add_argument('--seed', type=int, default=None)
    return parser


def charge_modele(game, a, device):
    model = ResNet(game, a.res_blocks, a.hidden, device)
    if a.model is not None:
        model.load_state_dict(torch.load(a.model, map_location=device))
    model.eval()
    return model


def coup_aleatoire(game, state):
    indices = np.where(game.get_valid_moves(state) == 1)[0]        # prend les indices valant 1
    return np.random.choice(indices)


def coup_mcts(game, mcts, state, player):
    neutral_state = game.change_perspective(state, player)
    mcts_probs = mcts.search(neutral_state)
    return int(np.argmax(mcts_probs))


def partie(game, joueur1, joueur2):
    # joueur1 et joueur2 sont des fonctions (state, player) -> action
    # renvoie 1 si joueur1 gagne, -1 si joueur2 gagne, 0 si nul
    state = game.get_initial_state()
    player = 1
    while True:
        action = joueur1(state, player) if player == 1 else joueur2(state, player)
        state = game.get_next_state(state, action, player)
        value, is_terminal = game.get_value_and_terminated(state, action)
        if is_terminal:
            return player if value == 1 else 0
        player = game.get_opponent(player)


def joueur_aleatoire(game):
    return lambda s, p: coup_aleatoire(game, s)


def joueur_mcts(game, mcts):
    return lambda s, p: coup_mcts(game, mcts, s, p)


def joueur_minmax(game, profondeur):
    m = MinMax(game, profondeur)
    return lambda s, p: m.get_action(s, p)


def match(game, ia, adversaire, n_parties):
    # ia joue la moitie des parties en premier et l'autre moitie en second
    resultats = {'premier': [], 'second': []}
    for i in trange(n_parties):
        if i % 2 == 0:
            resultats['premier'].append(partie(game, ia, adversaire))
        else:
            resultats['second'].append(-partie(game, adversaire, ia))
        # dans les deux cas 1 = victoire de ia
    return resultats


def joue_contre_rand(game, mcts, n_parties):
    return match(game, joueur_mcts(game, mcts), joueur_aleatoire(game), n_parties)


def affiche_resultats(resultats):
    for ordre, r in resultats.items():
        if not r:
            continue
        print(f"{ordre} : {r.count(1)} victoires, {r.count(0)} nuls, {r.count(-1)} defaites sur {len(r)}")


def demande_coup(game, state):
    while True:
        txt = input(f"ton coup ({game.dim} coordonnees entre 0 et {game.length - 1}) : ")
        try:
            indices = tuple(int(x) for x in txt.replace(',', ' ').split())
        except ValueError:
            continue
        if len(indices) != game.dim or not game.in_bounds(indices):
            print("coordonnees invalides")
            continue
        if state[indices] != 0:
            print("case deja prise")
            continue
        return int(np.ravel_multi_index(indices, game.shape))


def joue_contre_humain(game, ia, humain_commence):
    state = game.get_initial_state()
    player = 1
    humain = 1 if humain_commence else -1
    if game.dim > 2:
        print("les tranches sont affichees de gauche a droite (premiere coordonnee = numero de tranche)")
    while True:
        print(game.display(state))
        print()
        if player == humain:
            action = demande_coup(game, state)
        else:
            action = ia(state, player)
            print("l'ia joue", game.to_base(action))
        state = game.get_next_state(state, action, player)
        value, is_terminal = game.get_value_and_terminated(state, action)
        if is_terminal:
            print(game.display(state))
            if value == 0:
                print("match nul")
            elif player == humain:
                print("gagne")
            else:
                print("perdu")
            return
        player = game.get_opponent(player)


if __name__ == '__main__':
    a = get_parser().parse_args()

    if a.seed is not None:
        np.random.seed(a.seed)
        random.seed(a.seed)
        torch.manual_seed(a.seed)

    game = NDGame(a.length, a.dim, a.k)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = charge_modele(game, a, device)

    if a.mode == 'train':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        if a.optimizer is not None:
            optimizer.load_state_dict(torch.load(a.optimizer, map_location=device))

        args = {
            'C': a.C,
            'num_searches': a.searches,
            'num_iterations': a.iterations,
            'num_selfPlay_iterations': a.selfplay,
            'num_epochs': a.epochs,
            'batch_size': a.batch_size,
            'temperature': a.temperature,
            'dirichlet_epsilon': a.epsilon,
            # avant c'etait 1 : a la racine on ne regardait que le bruit et plus du tout le reseau
            'dirichlet_alpha': a.alpha,
            'augment': not a.no_augment,
            'max_symmetries': a.max_symmetries,
            'nom': str(game),
        }

        alphaZero = AlphaZero(model, optimizer, game, args)
        alphaZero.learn()

    else:
        args = {
            'C': a.C,
            'num_searches': a.searches,
            'dirichlet_epsilon': 0.,
        }
        mcts = MCTS(game, args, model)

        if a.joueur == 'minmax':
            ia = joueur_minmax(game, a.profondeur)
        else:
            if a.model is None:
                print("pas de --model : reseau non entraine")
            ia = joueur_mcts(game, mcts)

        if a.mode == 'eval':
            if a.adversaire == 'minmax':
                adversaire = joueur_minmax(game, a.profondeur)
            else:
                adversaire = joueur_aleatoire(game)
            print(f"{a.joueur} contre {a.adversaire}")
            affiche_resultats(match(game, ia, adversaire, a.parties))
        else:
            joue_contre_humain(game, ia, a.first)
