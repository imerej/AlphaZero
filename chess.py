import numpy as np
import math
import torch
import random

import matplotlib.pyplot as plt

import tqdm
from tqdm import trange

import torch.nn as nn
import torch.nn.functional as F






def est_adversaire(char):
    if (char == 't' or char == 'c' or char ==  'f' or char == 're' or char == 'ro'):
        return True
    return False
class Chess:
    def __init__(self):
        self.row_count = 8
        self.column_count = 8 
        self.action_size = self.row_count * self.column_count
        
    def get_initial_state(self):
        board = np.array([
            ['t', 'c', 'f', 'd', 'r', 'f', 'c', 't'],  # adversaire 
            ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],  
            ['.', '.', '.', '.', '.', '.', '.', '.'],  
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],  # joueur courant 
            ['T', 'C', 'F', 'D', 'R', 'F', 'C', 'T'],  
        ])
        return board
    #blanc commence
    #on sera toujours les majuscules durant la partie
    
    
    def get_next_state(self, state, action, player):
        (dep, arr, changement) = action #case de depart case d'arrivee et la nouvelle piece dans le cas ou un pion se transforme
        state[arr] =  state[dep]
        state[dep] = '.'
        if(changement != ''):
            state[arr] =  changement
        return state
    #action entier entre 0 et 8
    
    def get_valid_moves_pawn(self, state, position):
        x,y = position
        valid_moves = []
        if (x > 2 and state[x-1][y] == '.'):
            valid_moves += (x-1,y,'')
        
        if (x == 1 and state[x-1][y] == '.'):
            valid_moves += (x,y+1,'T')
            valid_moves += (x,y+1,'D')
            valid_moves += (x,y+1,'F')
            valid_moves += (x,y+1,'C')
            
            
        if (x > 1 and y < 7 and est_adversaire(state[x-1][y+1])):
            valid_moves += [(x+1,y+1,'')]
            
        if (x == 1 and y < 7 and  est_adversaire(state[x-1][y+1])):
            valid_moves += (x,y+1,'T')
            valid_moves += (x,y+1,'D')
            valid_moves += (x,y+1,'F')
            valid_moves += (x,y+1,'C')
            
            
        if (x > 1 and y > 0 and est_adversaire(state[x-1][y-1])):
            valid_moves += [(x-1,y+1,'')]    
            
        if (x == 1 and y > 0 and est_adversaire(state[x-1][y-1])):
            valid_moves += (x,y+1,'T')
            valid_moves += (x,y+1,'D')
            valid_moves += (x,y+1,'F')
            valid_moves.append((x,y+1,'C'))
        
        return valid_moves
        
    
    
    
    def get_valid_moves_knight(self, state, position):
        x, y = position
        valid_moves = []
        moves = [
            (x + 1, y + 2), (x + 2, y + 1),
            (x + 2, y - 1), (x + 1, y - 2),
            (x - 1, y - 2), (x - 2, y - 1),
            (x - 2, y + 1), (x - 1, y + 2),
        ]
        for (nx, ny) in moves:
            if 0 <= nx < 8 and 0 <= ny < 8:
                target = state[nx][ny]
                if target == '.' or est_adversaire(target):
                    valid_moves.append((nx, ny, ''))
        return valid_moves

    def get_valid_moves_king(self, state, position):
        x, y = position
        valid_moves = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < 8 and 0 <= ny < 8:
                    target = state[nx][ny]
                    if target == '.' or est_adversaire(target):
                        valid_moves.append((nx, ny, ''))
        return valid_moves
            
            
    def get_valid_moves_bishop(self, state, position):
        x, y = position
        valid_moves = []

        for dx, dy in [(1,1), (-1,1), (1,-1), (-1,-1)]:
            nx, ny = x + dx, y + dy
            while 0 <= nx < 8 and 0 <= ny < 8:
                target = state[nx][ny]
                if target == '.':
                    valid_moves.append((nx, ny, ''))
                elif  est_adversaire(target):
                    valid_moves.append((nx, ny, ''))
                    break
                else:
                    break
                nx += dx
                ny += dy
        return valid_moves
        
            
    def get_valid_moves_rook(self, state, position):
        x, y = position
        valid_moves = []

        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = x + dx, y + dy
            while 0 <= nx < 8 and 0 <= ny < 8:
                target = state[nx][ny]
                if target == '.':
                    valid_moves.append((nx, ny, ''))
                elif  est_adversaire(target):
                    valid_moves.append((nx, ny, ''))
                    break
                else:
                    break
                nx += dx
                ny += dy
        return valid_moves

            
            
    
    def get_valid_moves_queen(self, state, position):
        return self.get_valid_moves_rook(state, position) + self.get_valid_moves_bishop(state, position)

    
    
    def get_valid_moves(self, state, last_moves): #state.reshape(-1)?
        valid_moves = []
        for x in range(8):
            for y in range(8):
                if(state[x][y] == 'R'):
                    for arr in self.get_valid_moves_king(state, (x,y)):
                        (nx,ny,char) = arr
                        valid_moves += [((x,y), (nx,ny), char)]
                if(state[x][y] == 'D'):
                    for arr in self.get_valid_moves_queen(state, (x,y)):
                        (nx,ny,char) = arr
                        valid_moves += [((x,y), (nx,ny), char)]
                if(state[x][y] == 'C'):
                    for arr in self.get_valid_moves_knight(state, (x,y)):
                        (nx,ny,char) = arr
                        valid_moves += [((x,y), (nx,ny), char)]
                if(state[x][y] == 'F'):
                    for arr in self.get_valid_moves_bishop(state, (x,y)):
                        (nx,ny,char) = arr
                        valid_moves += [((x,y), (nx,ny), char)]
                if(state[x][y] == 'T'):
                    for arr in self.get_valid_moves_rook(state, (x,y)):
                        (nx,ny,char) = arr
                        valid_moves += [((x,y), (nx,ny), char)]
                if(state[x][y] == 'P'):
                    for arr in self.get_valid_moves_pawn(state, (x,y)):
                        (nx,ny,char) = arr
                        valid_moves += [((x,y), (nx,ny), char)]
        return valid_moves
    
    def check_win(self, state, action):
        if action == None:
            return False
        
        row = action // self.column_count
        column = action % self.column_count
        player = state[row, column]
        
        return (
            np.sum(state[row, :]) == player * self.column_count
            #state[row, :] renvoie un tab contenant toute la ligne row
            or np.sum(state[:, column]) == player * self.row_count
            or np.sum(np.diag(state)) == player * self.row_count
            or np.sum(np.diag(np.flip(state, axis=0))) == player * self.row_count
        )
    
    
    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False
    
    def get_opponent(self, player):
        return -player
    
    def get_opponent_value(self, value):
        return -value
    
    def change_perspective(self, state, player):
        return state * player
    
    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)
        
        return encoded_state
    #ceci transforme le plateau (tab) en trois tab : j1, j2, vide







    jeu = Chess()
jeu.__init__()
state = jeu.get_initial_state()
print(state)
#coups = jeu.get_valid_moves(state, [])
coups = jeu.get_valid_moves_pawn(state, (1,0))
print(coups)
