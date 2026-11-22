import math
import numpy as np
import itertools


class NDGame:
    def __init__(self, length, dim, k=None):
        self.length = length # length as in the length of the side of the hypercube
        self.dim = dim
        self.k = length if k is None else k
        # k = nb de pions a aligner pour gagner (par defaut toute une ligne)
        # ex : NDGame(4, 3) c'est le qubic, NDGame(5, 2, 4) un puissance 4 sans gravite
        if self.k > self.length:
            raise ValueError("k ne peut pas etre plus grand que le cote")
        self.action_size = self.length ** self.dim
        self.shape = (self.length, ) * self.dim #dim-uplet fait de chaque bord de lhypercube

        # on ne garde qu'une direction sur deux : d et -d donnent la meme ligne
        # (on regarde des deux cotes dans check_win de toute facon)
        self.directions = [
            d for d in itertools.product([-1, 0, 1], repeat=self.dim)
            if any(d) and d[next(i for i, x in enumerate(d) if x != 0)] == 1
        ]
        # pour dim = 3 ca fait 13 directions au lieu de 26

        self.symmetries = self.get_symmetries()

    def __repr__(self):
        return f"{'x'.join([str(self.length)] * self.dim)}_k{self.k}"

    def get_initial_state(self):
        return np.zeros(self.shape)
            #jimplemente un matrice taille^n, l'encodage pour le reseau est fait dans get_encoded_state

            #implementation par hypergraphe etait lautre moyen
            #on aurait fait ca pour garder la pertinence de conv2D
            #la on met la fonction arete de william pour inserer des 2

    def to_base(self, n, base=None):
        # action (entier) -> indices dans l'hypercube
        # meme chose que np.unravel_index(n, self.shape)
        if base is None:
            base = self.length
        decomp = []
        for _ in range(self.dim):
            decomp.append(n % base)
            n //= base
        return tuple(reversed(decomp))

    def get_next_state(self, state, action, player):
        indices = self.to_base(action, self.length)
        state[indices] = player
        return state

    def get_valid_moves(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)
        #transforme mon tableau n dimensionsel en tab 1 D trouve les cases libres (==0)
        #ca renvoie un tableau de 1 et 0 selon si la case est libre

    def get_opponent(self, player):
        return -player

    def get_opponent_value(self, value):
        return -value

    def change_perspective(self, state, player):
        return state * player

    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False

    def get_directions(self):
        return self.directions

    def in_bounds(self, indices):
        return all(0 <= i < self.length for i in indices)

    def count_direction(self, state, indices, direction, player):
        # compte les pions du joueur a la suite dans une direction
        # on s'arrete au premier trou (avant on comptait toute la ligne,
        # ca marchait seulement parce que k = length)
        count = 0
        indices = tuple(x + y for x, y in zip(indices, direction))
        while self.in_bounds(indices) and state[indices] == player:
            count += 1
            indices = tuple(x + y for x, y in zip(indices, direction))
        return count

    def check_win(self, state, action):
        if action is None:
            return False

        indices_action = self.to_base(action, self.length)
        player = state[indices_action]
        if player == 0:
            return False

        for direction in self.directions:
            oppose = tuple(-x for x in direction)
            alignes = 1 + self.count_direction(state, indices_action, direction, player) \
                        + self.count_direction(state, indices_action, oppose, player)
            if alignes >= self.k:
                return True
        return False

    def get_winner(self, state):
        # gagnant sur une position quelconque (sans connaitre le dernier coup)
        # plus lent, sert pour les tests et le mode jeu
        for action in range(self.action_size):
            if state[self.to_base(action)] != 0 and self.check_win(state, action):
                return int(state[self.to_base(action)])
        return 0

    def get_encoded_shape(self):
        # forme de la grille vue par le reseau
        # jusqu'a 3 dimensions on garde la vraie forme (conv1d / conv2d / conv3d)
        # au dela pytorch n'a pas de conv4d donc on aplatit en 2D comme avant
        if self.dim <= 3:
            return self.shape
        return (self.length, self.length ** (self.dim - 1))

    def get_encoded_state(self, state):
        state = state.reshape(self.get_encoded_shape())
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)

        return encoded_state

    def get_symmetries(self):
        # le groupe des symetries de l'hypercube : on permute les axes
        # et on retourne (flip) certains axes -> 2^dim * dim! symetries
        # 8 pour le morpion classique, 48 pour le 3x3x3
        # chaque symetrie envoie une ligne gagnante sur une ligne gagnante
        # donc la valeur d'une position ne change pas
        sym = []
        for perm in itertools.permutations(range(self.dim)):
            for flips in itertools.product([False, True], repeat=self.dim):
                sym.append((perm, tuple(i for i in range(self.dim) if flips[i])))
        return sym

    def apply_symmetry(self, array, symmetry):
        perm, flips = symmetry
        array = np.transpose(array, perm)
        if flips:
            array = np.flip(array, axis=flips)
        return np.ascontiguousarray(array)

    def augment(self, state, action_probs, max_symmetries=None):
        # renvoie (state, policy) pour toutes les symetries (ou un tirage de max_symmetries)
        # la policy est remise sous forme d'hypercube pour subir la meme transformation
        symmetries = self.symmetries
        if max_symmetries is not None and max_symmetries < len(symmetries):
            idx = np.random.choice(len(symmetries), max_symmetries, replace=False)
            symmetries = [symmetries[i] for i in idx]

        probs = action_probs.reshape(self.shape)
        res = []
        for s in symmetries:
            res.append((
                self.apply_symmetry(state, s),
                self.apply_symmetry(probs, s).reshape(-1)
            ))
        return res

    def display(self, state):
        # affichage console, pour dim > 2 on affiche les tranches les unes a cote des autres
        symboles = {1: 'X', -1: 'O', 0: '.'}
        if self.dim == 1:
            return ' '.join(symboles[int(x)] for x in state)
        if self.dim == 2:
            return '\n'.join(' '.join(symboles[int(x)] for x in ligne) for ligne in state)
        tranches = state.reshape((-1, self.length, self.length))
        lignes = []
        for i in range(self.length):
            lignes.append('   '.join(' '.join(symboles[int(x)] for x in t[i]) for t in tranches))
        return '\n'.join(lignes)
