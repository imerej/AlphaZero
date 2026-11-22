import os
import numpy as np
import math
import torch
import random

import tqdm
from tqdm import trange

import torch.nn as nn
import torch.nn.functional as F

from NDGame import *
from ResNet import *
from MCTS import *


class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)
        self.historique = []
        # pertes moyennes a chaque iteration, pour tracer les courbes apres

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
                    #indique le gagnant du pov du joueur courant

                    # chaque position est ajoutee avec ses symetries
                    # meme valeur, policy transformee de la meme facon que la grille
                    if self.args.get('augment', True):
                        positions = self.game.augment(hist_neutral_state, hist_action_probs,
                                                      self.args.get('max_symmetries'))
                    else:
                        positions = [(hist_neutral_state, hist_action_probs)]

                    for s, p in positions:
                        returnMemory.append((
                            self.game.get_encoded_state(s), #encode letat
                            p,
                            hist_outcome
                        ))
                return returnMemory

            player = self.game.get_opponent(player)



    def train(self, memory):
        random.shuffle(memory)
        pertes = []
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:batchIdx + self.args['batch_size']]
            # sample correspond a tous les elemnts de memory dindices
            # entre batchIdx et batchIdx + batch size
            # (avant il y avait un min(len(memory) - 1, ...) qui oubliait le dernier element)
            if len(sample) < 2:
                continue
            # batchnorm plante en mode train avec un batch de taille 1

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

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pertes.append((policy_loss.item(), value_loss.item()))
        return pertes


    def learn(self):
        dossier = self.args.get('dossier', 'checkpoints')
        os.makedirs(dossier, exist_ok=True)

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

            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations'], desc=f"self play {iteration}"):
                memory += self.selfPlay()
                # on stock les donnees de la partie
            # trange donne une visualisation de la progression dans la boucle

            self.model.train()
            pertes = []
            for epoch in trange(self.args['num_epochs'], desc=f"train {iteration}"):
                if len(memory) < self.args['batch_size']:
                    print(f"[WARN] Skipping epoch {epoch} — not enough data in memory ({len(memory)} < {self.args['batch_size']})")
                    continue  # skip this epoch, go to next one
                pertes += self.train(memory)

            if pertes:
                policy_loss, value_loss = np.mean(pertes, axis=0)
                self.historique.append((iteration, policy_loss, value_loss))
                print(f"iteration {iteration} : {len(memory)} positions, policy loss {policy_loss:.3f}, value loss {value_loss:.3f}")

            nom = f"{iteration}_{self.args['num_searches']}_{self.args['num_iterations']}_{self.args['num_selfPlay_iterations']}_{self.args['num_epochs']}_{self.args['nom']}"
            torch.save(self.model.state_dict(), os.path.join(dossier, f"model_{nom}.pt"))
            torch.save(self.optimizer.state_dict(), os.path.join(dossier, f"optimizer_{nom}.pt"))
            # saves the states to a disk file
            # torch.save(obj, f, pickle_module=pickle, pickle_protocol=2, _use_new_zipfile_serialization=True)[source]

        return self.historique
