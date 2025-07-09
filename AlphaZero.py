import numpy as np
import math
import torch
import random

import matplotlib.pyplot as plt

import tqdm
from tqdm import trange

import torch.nn as nn
import torch.nn.functional as F

from NDGame import *
from ResNet import *
from MCTS import *
from AlphaZero import *


class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)
        
    def selfPlay(self):
        memory = []
        player = 1
        state = self.game.get_initial_state()
        
        while True:
            neutral_state = self.game.change_perspective(state, player)
            #car on veut tjrs etre le joueur 1 pour mcts
            action_probs = self.mcts.search(neutral_state)
            
            memory.append((neutral_state, action_probs, player))
            
            temperature_action_probs = action_probs ** (1 / self.args['temperature'])
            #la temperature permet de modifier durant le processus les coeff dexploration-optimisation
            #si la temperature est est superieure a 1, on explore plus : car la puissance va exacerber ou au contraire reduire les differences
            #plus la temperature est basse, on optimise
            temperature_action_probs = temperature_action_probs / np.sum(temperature_action_probs)
            #car il faut renormaliser derriere
            
            action = np.random.choice(self.game.action_size, p = temperature_action_probs)
            #choisi une action au hasard selon la distribution de proba
            
            state = self.game.get_next_state(state, action, player)
            
            value, is_terminal = self.game.get_value_and_terminated(state, action)
            
            if is_terminal:
                returnMemory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                    returnMemory.append((
                        self.game.get_encoded_state(hist_neutral_state), #encode letat
                        hist_action_probs, #change pas
                        hist_outcome #indique le gagnant du pov du joueur courant
                    ))
                return returnMemory
            
            player = self.game.get_opponent(player)
                
            
    
    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])]
            # Change to memory[batchIdx:batchIdx+self.args['batch_size']] in case of an error
            # sample correspond a tous les elemnts de memory dindices
            # entre batchIdx et batchIdx + batch size
            
            state, policy_targets, value_targets = zip(*sample)
            # names = ['Alice', 'Bob', 'Charlie']
            # ages = [25, 30, 35]
            # zipped = zip(names, ages)
            # print(list(zipped))
            # [('Alice', 25), ('Bob', 30), ('Charlie', 35)]


            
            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)
            
            state = torch.tensor(state, dtype=torch.float32, device = self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device = self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device = self.model.device)
            
            out_policy, out_value = self.model(state)
            
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss
            
            self.optimizer.zero_grad() # change to self.optimizer
            loss.backward()
            self.optimizer.step() # change to self.optimizer
            
    
    def learn(self):
        for iteration in range(self.args['num_iterations']):
            
            memory = []
            # on stock ici etat, proba, et recompense
            
            self.model.eval()
            # desactive les drop out, ie la mise a zero aleatoires de neurons evitant le surapprentissage
            # contraire de model.train()
            
            # During training, batch norm layers compute the mean 
            # and variance of the current batch to normalize the input.

            # During evaluation, batch norm uses running estimates (computed during 
            # training) instead of the current batch statistics.
            
            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations']):
                memory += self.selfPlay()
                # on stock les donnees de la partie
            # trange donne une visualisation de la progression dans la boucle
            
            self.model.train()
            for epoch in trange(self.args['num_epochs']):
                if len(memory) < self.args['batch_size']:
                    print(f"[WARN] Skipping epoch {epoch} — not enough data in memory ({len(memory)} < {self.args['batch_size']})")
                    continue  # skip this epoch, go to next one
                self.train(memory)

            torch.save(self.model.state_dict(), f"model_{iteration}_{self.args['num_searches']}_{self.args['num_iterations']}_{self.args['num_selfPlay_iterations']}_{self.args['num_epochs']}_{self.args['nom']}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}_{self.args['num_searches']}_{self.args['num_iterations']}_{self.args['num_selfPlay_iterations']}_{self.args['num_epochs']}_{self.args['nom']}.pt")
            # saves the states to a disk file
            # torch.save(obj, f, pickle_module=pickle, pickle_protocol=2, _use_new_zipfile_serialization=True)[source]
