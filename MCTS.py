import numpy as np
import math
import torch
import random

import matplotlib.pyplot as plt

import tqdm
from tqdm import trange

import torch.nn as nn
import torch.nn.functional as F


class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior = 0, visit_count = 0):
        
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        #prior : proba d'une action du pov du parent
        self.children = []
        self.visit_count = visit_count
        """
        self.expandable_moves = game.get_valid_moves(state)
        #car on expande tout dun coup mtn
        """
        self.visit_count = visit_count
        self.value_sum = 0
        
    def is_fully_expanded(self):
        return len(self.children) > 0 # and np.sum(self.expandable_moves) == 0 
    
    def select(self):
        best_child = None
        best_ucb = -np.inf
        
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
                
        return best_child
    
    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value  = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        # 1 - ... permet de renverser les roles parents/enfant adversaires 
        #return q_value + self.args['C'] * (math.sqrt(math.log(self.visit_count) / child.visit_count)
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior
    
    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_perspective(child_state, player=-1)

                child = Node(self.game, self.args, child_state, self, action, prob)
                self.children.append(child)
        return child
        """
        action = np.random.choice(np.where(self.expandable_moves == 1)[0])
        # [0] tranforme le n - uplet renvoyé en un tab auquel rand choice est appliquable
        self.expandable_moves[action] = 0
        #indique que la case est desormais plus jouable
        """
        
                    
        
    """
    def simulate(self):
        value, is_terminal = self.game.get_value_and_terminated(self.state, self.action_taken)
        value = self.game.get_opponent_value(value)
        
        if is_terminal:
            return value
        
        rollout_state = self.state.copy()
        rollout_player = 1
        #rollout state signifie etat de reference, courant, initial 
        while True:
            valid_moves = self.game.get_valid_moves(rollout_state)
            action = np.random.choice(np.where(valid_moves == 1)[0])
            rollout_state = self.game.get_next_state(rollout_state, action, rollout_player)
            value, is_terminal = self.game.get_value_and_terminated(rollout_state, action)
            if is_terminal:
                if rollout_player == -1:
                    value = self.game.get_opponent_value(value)
                return value    
            
            rollout_player = self.game.get_opponent(rollout_player)
    """
    
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        
        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)  


class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model
        
    @torch.no_grad() 
    # car on ne souhaite pas stocker le parcours effectué sur le 
    # gradient mais juste lutiliser pour obtenir policy,val
    # enleve lautograd 
    #gain de memoire et de vitesse
    
    def search(self, state):
        root = Node(self.game, self.args, state, visit_count = 1)
        #initialise visit count a 1 car on letend apres
        
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(state), device = self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis = 1).squeeze(0).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) \
            * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)
        # P(s,a) = (1-epsilon) (p\a) + epsilon (eta\a)
        #les lignes precedentes servent a introduire du bruit
        #pour avoir + d'aleatoire et + explorer
        
        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves 
        policy /= np.sum(policy)    
        root.expand(policy)
        
        for search in range(self.args['num_searches']):
            #selection
            node = root
            while node.is_fully_expanded():
                node = node.select()
                
                
            #mesure pour la racine     
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)
            
            if not is_terminal:
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state),device = self.model.device).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                #pour eviter de jouer la ou on a pas le droit
                
                policy *= valid_moves #ca met a 0 les coups illegaux
                policy /= np.sum(policy) # ca remet une proba
                
                value = value.item()
                 
                #expansion
                node = node.expand(policy)
                
                '''
                #simulation
                value = node.simulate()
                '''
                
            #retropropagation
            node.backpropagate(value)    
            
        #on veut maintenant extraire les resulatats : le vecteur de proba
        #on renvoie donc le nb de visite effectuées sur un sommet divisé par le nombre total de visites
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs
