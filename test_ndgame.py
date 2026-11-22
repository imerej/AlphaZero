import itertools
import numpy as np
import torch

from NDGame import NDGame
from ResNet import ResNet
from MCTS import MCTS
from minmax import MinMax


# lance avec : python -m pytest test_ndgame.py  (ou juste python test_ndgame.py)


def lignes_gagnantes(game):
    # toutes les lignes de k cases, calculees betement pour comparer avec check_win
    lignes = set()
    for depart in itertools.product(range(game.length), repeat=game.dim):
        for d in itertools.product([-1, 0, 1], repeat=game.dim):
            if not any(d):
                continue
            cases = [tuple(x + i * y for x, y in zip(depart, d)) for i in range(game.k)]
            if all(game.in_bounds(c) for c in cases):
                lignes.add(frozenset(cases))
    return lignes


def gagnant_naif(game, state, lignes):
    for l in lignes:
        valeurs = {state[c] for c in l}
        if len(valeurs) == 1 and 0 not in valeurs:
            return int(valeurs.pop())
    return 0


def gagnants(state, lignes):
    # sur une grille aleatoire les deux joueurs peuvent avoir une ligne
    res = set()
    for l in lignes:
        valeurs = {state[c] for c in l}
        if len(valeurs) == 1 and 0 not in valeurs:
            res.add(int(valeurs.pop()))
    return res


def test_nombre_de_lignes():
    # formule connue : ((n+2)^d - n^d) / 2 lignes gagnantes pour le n^d
    for n, d in [(3, 2), (3, 3), (4, 3), (3, 4)]:
        game = NDGame(n, d)
        assert len(lignes_gagnantes(game)) == ((n + 2) ** d - n ** d) // 2


def test_check_win_parties_aleatoires():
    rng = np.random.default_rng(0)
    for n, d, k in [(3, 2, 3), (3, 3, 3), (4, 3, 4), (5, 2, 4), (4, 2, 3), (3, 4, 3)]:
        game = NDGame(n, d, k)
        lignes = lignes_gagnantes(game)
        for _ in range(100):
            state = game.get_initial_state()
            player = 1
            while True:
                action = rng.choice(np.where(game.get_valid_moves(state) == 1)[0])
                state = game.get_next_state(state, action, player)
                value, fini = game.get_value_and_terminated(state, action)
                # le coup qui vient d'etre joue gagne ssi une ligne est complete
                # (avant ce coup personne n'avait gagne puisque la partie continuait)
                attendu = gagnant_naif(game, state, lignes)
                assert (value == 1) == (attendu == player)
                if fini:
                    break
                player = -player


def test_trou_dans_la_ligne():
    # X X . X X sur une ligne de 5 avec k = 4 : pas gagne
    game = NDGame(5, 2, 4)
    state = game.get_initial_state()
    for j in [0, 1, 3, 4]:
        state[0, j] = 1
    assert not game.check_win(state, 4)
    state[0, 2] = 1
    assert game.check_win(state, 2)


def test_symetries():
    for n, d in [(3, 2), (3, 3), (4, 2)]:
        game = NDGame(n, d)
        assert len(game.symmetries) == 2 ** d * len(list(itertools.permutations(range(d))))
        lignes = lignes_gagnantes(game)
        rng = np.random.default_rng(1)
        for _ in range(50):
            state = rng.choice([-1, 0, 1], size=game.shape).astype(float)
            probs = rng.random(game.action_size)
            probs /= probs.sum()
            g = gagnants(state, lignes)
            vus = set()
            for s, p in game.augment(state, probs):
                # meme gagnant apres symetrie
                assert gagnants(s, lignes) == g
                assert np.isclose(p.sum(), 1)
                vus.add(s.tobytes())
            # l'identite est dans le groupe
            assert state.tobytes() in vus


def test_policy_suit_la_grille():
    # on met un seul pion et toute la proba sur sa case :
    # apres chaque symetrie la proba doit toujours etre sur la case du pion
    game = NDGame(3, 3)
    for action in range(game.action_size):
        state = game.get_initial_state()
        state = game.get_next_state(state, action, 1)
        probs = np.zeros(game.action_size)
        probs[action] = 1
        for s, p in game.augment(state, probs):
            assert s[game.to_base(int(np.argmax(p)))] == 1


def test_reseau_formes():
    for n, d in [(5, 1), (3, 2), (3, 3), (3, 4)]:
        game = NDGame(n, d)
        model = ResNet(game, 1, 8, torch.device('cpu'))
        model.eval()
        x = torch.tensor(game.get_encoded_state(game.get_initial_state())).unsqueeze(0)
        policy, value = model(x)
        assert policy.shape == (1, game.action_size)
        assert value.shape == (1, 1)


def test_mcts_coup_gagnant():
    # avec assez de recherches le mcts doit trouver le coup gagnant immediat
    # meme avec un reseau pas entraine
    torch.manual_seed(0)
    game = NDGame(3, 2)
    model = ResNet(game, 1, 8, torch.device('cpu'))
    model.eval()
    mcts = MCTS(game, {'C': 2, 'num_searches': 200}, model)
    state = game.get_initial_state()
    for a, p in [(0, 1), (3, -1), (1, 1), (4, -1)]:
        state = game.get_next_state(state, a, p)
    # X X .
    # O O .
    # . . .   -> X doit jouer 2
    probs = mcts.search(state)
    assert np.argmax(probs) == 2


def test_minmax_lignes():
    for n, d, k in [(3, 2, 3), (3, 3, 3), (4, 3, 4), (5, 2, 4)]:
        game = NDGame(n, d, k)
        assert len(MinMax(game).lignes) == len(lignes_gagnantes(game))


def test_minmax_gagne_et_bloque():
    game = NDGame(3, 2)
    m = MinMax(game, 2, aleatoire=False)
    state = game.get_initial_state()
    for a, p in [(0, 1), (3, -1), (1, 1), (4, -1)]:
        state = game.get_next_state(state, a, p)
    # X X .
    # O O .
    # . . .
    assert m.get_action(state, 1) == 2   # X gagne en 2
    state[0, 0] = 0
    state = game.get_next_state(state, 6, 1)
    # . X .
    # O O .
    # X . .   -> X doit bloquer en 5
    assert m.get_action(state, 1) == 5
    # et O joue 5 pour gagner
    assert m.get_action(state, -1) == 5


def test_minmax_3d():
    # en 3x3x3 le premier coup doit etre le centre (13 lignes passent par lui)
    game = NDGame(3, 3)
    m = MinMax(game, 2, aleatoire=False)
    assert m.get_action(game.get_initial_state(), 1) == 13


def test_heuristique_symetrique():
    game = NDGame(4, 3)
    m = MinMax(game)
    rng = np.random.default_rng(2)
    for _ in range(20):
        state = rng.choice([-1, 0, 1], size=game.shape).astype(float)
        assert m.heuristique(state) == -m.heuristique(-state)


if __name__ == '__main__':
    for nom, f in list(globals().items()):
        if nom.startswith('test_'):
            f()
            print(nom, 'ok')
