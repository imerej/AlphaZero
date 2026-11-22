# AlphaZero sur le morpion en n dimensions

Implementation d'AlphaZero (MCTS + reseau residuel, apprentissage par self play) pour le morpion
sur un hypercube de cote `length` en dimension `dim`. On gagne en alignant `k` pions
(par defaut `k = length`, donc une ligne entiere) dans n'importe quelle direction,
diagonales comprises.

Quelques exemples :

- `NDGame(3, 2)` : le morpion classique
- `NDGame(3, 3)` : morpion 3x3x3
- `NDGame(4, 3)` : le qubic (4x4x4)
- `NDGame(5, 2, 4)` : grille 5x5, il faut en aligner 4

## Fichiers

- `NDGame.py` : regles du jeu, detection de victoire, symetries de l'hypercube
- `ResNet.py` : le reseau (une tete policy et une tete value)
- `MCTS.py` : la recherche guidee par le reseau
- `AlphaZero.py` : boucle self play / entrainement
- `main.py` : entrainement, evaluation contre un joueur aleatoire, partie contre l'ia
- `test_ndgame.py` : tests
- `chess.py` : debut des echecs, pas fini

## Utilisation

```
pip install -r requirements.txt

python main.py train --length 3 --dim 3
python main.py eval --length 3 --dim 3 --model checkpoints/model_2_60_3_200_4_3x3x3_k3.pt
python main.py play --length 3 --dim 3 --model checkpoints/model_2_60_3_200_4_3x3x3_k3.pt --first
```

`python main.py -h` pour voir tous les parametres (nb de recherches, iterations, bruit de dirichlet, etc).
Les modeles sont enregistres dans `checkpoints/` a chaque iteration. On peut reprendre un entrainement
avec `--model` et `--optimizer`.

En mode `play` on donne les coordonnees separees par des espaces. En 3D les tranches sont affichees
cote a cote, la premiere coordonnee est le numero de la tranche.

## Details

**Reseau.** Jusqu'a la dimension 3 le reseau utilise des convolutions de la meme dimension que la
grille (Conv1d, Conv2d, Conv3d). Avant, la grille etait toujours aplatie en `length x length^(dim-1)`
pour passer dans une Conv2d, et en 3D des cases voisines dans le cube ne l'etaient plus pour les filtres.
Au dela de la dimension 3 on garde l'aplatissement en 2D (pytorch n'a pas de Conv4d).

**Symetries.** L'hypercube a `2^dim * dim!` symetries (permutations des axes et retournements) :
8 pour le morpion classique, 48 pour le 3x3x3. Elles envoient une ligne gagnante sur une ligne gagnante,
donc chaque position du self play est ajoutee avec toutes ses images (la policy est transformee de
la meme facon). `--max_symmetries` permet d'en tirer seulement quelques unes si la memoire devient trop grosse.

**Victoire.** On ne teste que les lignes qui passent par le dernier coup, et seulement une direction
sur deux (`d` et `-d` donnent la meme ligne, on compte des deux cotes) : 13 directions au lieu de 26 en 3D.
Le comptage s'arrete au premier trou, ce qui permet d'avoir `k < length`.

Nombre de lignes gagnantes pour `k = length` : `((n+2)^d - n^d) / 2` (8 pour le 3x3, 49 pour le 3x3x3,
76 pour le qubic). Les tests verifient `check_win` en comparant avec toutes les lignes calculees a la main.

## Resultats

Morpion 3x3, 4 iterations de 100 parties de self play, 60 recherches (environ 2 min sur cpu).
Evaluation sur 200 parties contre un joueur aleatoire, le modele joue la moitie en premier :

| | recherches | victoires | nuls | defaites |
|---|---|---|---|---|
| reseau non entraine | 5 | 124 | 24 | 52 |
| reseau entraine | 5 | 187 | 13 | 0 |
| reseau non entraine | 60 | 183 | 11 | 6 |
| reseau entraine | 60 | 183 | 17 | 0 |

Le 3x3x3 est une victoire facile pour le premier joueur (il prend le centre) mais il faut nettement plus
d'iterations pour que le reseau le comprenne.

## Ce qui a ete corrige

- dans le MCTS, `expand` renvoyait le dernier enfant cree et la valeur du reseau etait retropropagee
  depuis cet enfant au lieu du noeud etendu (donc du mauvais point de vue)
- `dirichlet_epsilon` valait 1 : a la racine la policy du reseau etait completement remplacee par le bruit.
  Maintenant 0.25 et le bruit n'est tire que sur les coups legaux
- le decoupage en batch oubliait le dernier element de la memoire
- import circulaire dans `AlphaZero.py`

## A faire

- self play en parallele (plusieurs parties a la fois pour evaluer le reseau en batch)
- comparer chaque nouveau modele a l'ancien avant de le garder
- finir les echecs
