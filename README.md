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
- `minmax.py` : joueur minmax avec heuristique (profondeur 2 par defaut), sert d'adversaire de reference
- `main.py` : entrainement, evaluation (contre l'aleatoire ou le minmax), partie contre l'ia
- `test_ndgame.py` : tests
- `chess.py` : debut des echecs, pas fini

## Utilisation

```
pip install -r requirements.txt

python main.py train --length 3 --dim 3
python main.py eval --length 3 --dim 3 --model checkpoints/model_2_60_3_200_4_3x3x3_k3.pt
python main.py play --length 3 --dim 3 --model checkpoints/model_2_60_3_200_4_3x3x3_k3.pt --first

# alphazero contre le minmax, ou le minmax tout seul
python main.py eval --length 3 --dim 3 --model ... --adversaire minmax
python main.py eval --length 4 --dim 3 --joueur minmax --profondeur 2
python main.py play --length 3 --dim 3 --joueur minmax
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

**Minmax.** On regarde toutes les lignes gagnantes de l'hypercube. Une ligne ou un seul joueur a
des pions peut encore etre gagnee par lui : elle vaut `10^(nb de pions)` pour lui. Une ligne ou les deux
joueurs ont des pions ne compte plus. L'heuristique est la somme pour moi moins la somme pour l'adversaire
(meme idee que le score du minmax en C du puissance 4, en comptant les facons potentielles de gagner).
Avec la profondeur 2 (mon coup puis la reponse de l'adversaire) il voit les victoires immediates et
bloque celles de l'adversaire, mais pas les fourchettes. Elagage alpha beta, et tirage au hasard entre
les coups de meme score pour ne pas toujours jouer la meme partie.

## Resultats

Morpion 3x3, 4 iterations de 100 parties de self play, 60 recherches (environ 2 min sur cpu).
Evaluation sur 200 parties contre un joueur aleatoire, le modele joue la moitie en premier :

| | recherches | victoires | nuls | defaites |
|---|---|---|---|---|
| reseau non entraine | 5 | 124 | 24 | 52 |
| reseau entraine | 5 | 187 | 13 | 0 |
| reseau non entraine | 60 | 183 | 11 | 6 |
| reseau entraine | 60 | 183 | 17 | 0 |

Contre le minmax de profondeur 2 (100 parties, 60 recherches) le reseau entraine fait 100 nuls,
le reseau non entraine fait nul quand il commence et perd toutes les parties ou il joue en second.

Le minmax de profondeur 2 gagne toutes ses parties contre l'aleatoire en 3x3x3 et en 4x4x4.

Morpion 3x3x3, 8 iterations de 100 parties de self play, 60 recherches, 16 symetries par position
(environ 15 min sur cpu). Le 3x3x3 est gagne d'avance par le premier joueur (il prend le centre) et
il n'y a pas de nul possible. La value loss tombe presque a 0 vers l'iteration 4 et les parties de
self play raccourcissent (environ 6 coups) : le reseau a compris que le premier joueur gagne vite.

Evaluation sur 100 parties, la moitie en premier :

| | recherches | adversaire | en premier | en second |
|---|---|---|---|---|
| reseau non entraine | 60 | minmax | 0 / 50 | 0 / 50 |
| reseau non entraine | 5 | minmax | 0 / 50 | 0 / 50 |
| reseau entraine | 5 | minmax | 50 / 50 | 0 / 50 |
| reseau entraine | 60 | minmax | 50 / 50 | 0 / 50 |
| reseau entraine | 60 | aleatoire | 50 / 50 | 50 / 50 |
| minmax | | minmax | 50 / 50 | 0 / 50 |

(nombre de victoires). En second toutes les parties sont perdues contre le minmax, mais le minmax perd
aussi toutes ses parties en second contre lui meme : il trouve la victoire forcee quand il commence.
Pour comparer vraiment les joueurs il faudrait passer au 4x4x4 (qubic).

Attention avec seulement 2 coeurs : lancer deux entrainements ou evaluations en meme temps les rend
enormement plus lents (pytorch utilise plusieurs threads par processus).

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
