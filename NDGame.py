import math
import numpy as np
import itertools


                

    
    
class NDGame:
    def __init__(self, length, dim):
        self.length = length # elgnth as in the length of the side of the hypercube
        self.dim = dim
        self.action_size = self.length ** self.dim


    def get_initial_state(self):
        dimensions = (self.length, ) * self.dim #dim-uplet fait de chaque bord de lhypercube
        return np.zeros(dimensions) 
            #jimplemente un matrice taille^n mais je la transformerai en 2D plus tard
            
            #implementation par hypergraphe etait lautre moyen 
            #on aurait fait ca pour garder la pertinence de conv2D
            #la on met la fonction arete de william pour inserer des 2
            
    def to_base(self, n, base):
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
        return [d for d in itertools.product([-1, 0, 1], repeat=self.dim) if any(d)]
        #any regarde si il ny a que des zero (cest un OU pour chaque element du tableau en quelque sorte)
    
    def in_bounds(self, indices):
        return all(0 <= i < self.length for i in indices)

    def check_win(self, state, action):
        if action is None:
            return False
        
        indices_action = self.to_base(action, self.length)
        player = state[indices_action]
        for direction in self.get_directions():
            indices_avant = indices_action
            indices_arriere = indices_action
            alignes = 1
            for i in range(self.length):
                
                indices_avant = tuple((x + y)  for x, y in zip(indices_avant, direction))
                if self.in_bounds(indices_avant):
                    alignes += (state[indices_avant] == player)
                    
                indices_arriere = tuple((x - y)  for x, y in zip(indices_arriere, direction))
                if self.in_bounds(indices_arriere):
                    alignes += (state[indices_arriere] == player)
                    
            if alignes >= self.length :
                return True
        return False

    def get_encoded_state(self, state):
        state = state.reshape(self.length,-1)
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)
        
        return encoded_state
    
