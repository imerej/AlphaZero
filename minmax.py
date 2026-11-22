import itertools
import numpy as np


# joueur minmax avec heuristique, pour avoir un adversaire un peu plus serieux
# que l'aleatoire pour evaluer alphazero
# meme idee que le score du minmax en C du puissance 4 : on compte les facons
# potentielles de gagner, mais sur toutes les lignes de l'hypercube


VICTOIRE = 10 ** 9


class MinMax:
    def __init__(self, game, profondeur=2, aleatoire=True):
        self.game = game
        self.profondeur = profondeur
        # profondeur 2 = mon coup puis la reponse de l'adversaire, ensuite on evalue
        self.aleatoire = aleatoire
        # si plusieurs coups ont le meme score on en tire un au hasard
        # (sinon le minmax joue toujours la meme partie)

        self.lignes = self.get_lignes()
        # tableau (nb de lignes, k) : indices (aplatis) des cases de chaque ligne gagnante

        # poids d'une ligne suivant le nb de pions dessus
        # une ligne avec 2 pions vaut beaucoup plus que deux lignes avec 1 pion
        self.poids = np.array([0] + [10 ** i for i in range(game.k)], dtype=np.int64)
        self.poids[game.k] = VICTOIRE

    def get_lignes(self):
        game = self.game
        lignes = []
        for depart in itertools.product(range(game.length), repeat=game.dim):
            for d in game.directions:
                cases = [tuple(x + i * y for x, y in zip(depart, d)) for i in range(game.k)]
                if all(game.in_bounds(c) for c in cases):
                    lignes.append([np.ravel_multi_index(c, game.shape) for c in cases])
        return np.array(lignes)

    def heuristique(self, state):
        # score du point de vue du joueur 1
        # une ligne ou il n'y a que des pions a moi (au moins un) peut encore etre gagnee par moi
        # une ligne avec des pions des deux joueurs ne sert plus a personne
        valeurs = state.reshape(-1)[self.lignes]
        moi = np.sum(valeurs == 1, axis=1)
        lui = np.sum(valeurs == -1, axis=1)
        score = np.sum(self.poids[moi[lui == 0]]) - np.sum(self.poids[lui[moi == 0]])
        return int(score)

    def minmax(self, state, profondeur, alpha, beta, maximisation):
        # state est toujours vu du joueur 1 (celui qui choisit le coup a la racine)
        # maximisation : c'est au joueur 1 de jouer
        if profondeur == 0:
            return self.heuristique(state)

        valid_moves = np.where(self.game.get_valid_moves(state) == 1)[0]
        if len(valid_moves) == 0:
            return 0

        player = 1 if maximisation else -1
        meilleur = -np.inf if maximisation else np.inf
        for action in valid_moves:
            indices = self.game.to_base(action)
            state[indices] = player
            if self.game.check_win(state, action):
                # + profondeur pour preferer gagner vite (et perdre le plus tard possible)
                score = player * (VICTOIRE * 10 + profondeur)
            else:
                score = self.minmax(state, profondeur - 1, alpha, beta, not maximisation)
            state[indices] = 0
            # on annule le coup au lieu de copier la grille a chaque fois

            if maximisation:
                meilleur = max(meilleur, score)
                alpha = max(alpha, score)
            else:
                meilleur = min(meilleur, score)
                beta = min(beta, score)
            if beta <= alpha:
                break
            # elagage alpha beta : l'adversaire ne laissera jamais arriver ici
        return meilleur

    def scores(self, state):
        # score de chaque coup legal pour le joueur 1
        state = state.copy()
        res = {}
        for action in np.where(self.game.get_valid_moves(state) == 1)[0]:
            indices = self.game.to_base(action)
            state[indices] = 1
            if self.game.check_win(state, action):
                res[int(action)] = VICTOIRE * 10 + self.profondeur
            else:
                # pas d'elagage a la racine pour avoir le vrai score de chaque coup
                # (sinon les egalites sont mal detectees)
                res[int(action)] = self.minmax(state, self.profondeur - 1, -np.inf, np.inf, False)
            state[indices] = 0
        return res

    def get_action(self, state, player):
        neutral_state = self.game.change_perspective(state, player)
        res = self.scores(neutral_state)
        meilleur = max(res.values())
        candidats = [a for a, s in res.items() if s == meilleur]
        if self.aleatoire:
            return int(np.random.choice(candidats))
        return candidats[0]
