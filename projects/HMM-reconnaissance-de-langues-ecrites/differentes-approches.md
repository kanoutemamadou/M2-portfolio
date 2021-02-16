Le but du prjet était d'utiliser les différentes techniques des HMM pour reconnaître des langues à partir des fréquences d'émission des caractèreset des matrices de transition.

Langages utilisés: R

Données:
Base de textes au choix, avec minimum 30 textes
Ici 90 textes avec 45 textes en français et anglais


Les différentes approches:

## Prémière approche

L'algorithme EM multivariée avec une gaussienne mixture         
Evaluer la performance par validation croisée                  
Afficher la matrice de confusion                    

## Deuxième approche: Classifieur markovien

Créer un classifieur basé sur les matrices de transitions des différents symboles(lettres d'alphabets) de chaque langue et les probabilités initiales obtenues des 2 matrices de transitions des symboles.                                        
Evaluer la performance par validation croisée                      
Afficher la matrice de confusion                             

## Decodage par viterbi
IL fallait former une phrase d'au plus 1000 caractères où on alterne des parties en anglais et en français.
Le but était d'utiliser l'algorithme de viterbi pour détecter les passages en français et en anglais en se servant des probabilités d'émissions des symboles dans ces deux langues et une matrice de transition(on savait qu'on avait alterner des textes en français et en anglais).

## Algorithme de Baum-Welch
L'objectif était de partir de déterminer la matrice de transition A et les probabilités de transition en servant de l'algorithme de Baum-Welch:                                     
- En ayant une initialisation avec les paramètres de l'exercice précédent.
- En ayant une initialisation aléatoire

## L'algorithme EM Variationnel: Modèle à blocs stochastique
On devait déterminer les paramètres alpha et beta qui nous permettent d'affirmer que tel texte fait partie du groupe de français ou anglais à travers la matrice d'adjacence.
