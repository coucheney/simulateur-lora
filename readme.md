

## fichier de configuration
### Nodes.txt
Ce fichier permet de configurer le placement et la configuration des noeuds. 

Chaque ligne dans ce fichier donnera une configuration pour un ensemble de noeud.

placement nbNoeud [algo] [sf] [power] [cr] [period] [packetLen] [radius]
* placement : méthode de placement (rand, grid, line)
* nbNoeud : nombre de noeud à placer (si placement = grid, nbNoeud correspond à la largeur de la grille)

Les autre argument sont optionnel (argument:xxxxx)

* algo : nom de l'algorithme d'apprentissage (configuré dans configAlgo.txt) 
* sf : entier compris entre 7 et 12
* power : entier compris entre 0 et 20
* cr : entier compris entre 1 et 4
* period : entier correspondant au temps moyen entre l'envoie de deux messages
* packetLen : entier corespondant à la taille des packet envoyé
* radius : rayon du cercle utilisé pour placer les noeud

Il est également possible de placer un noeud à des coordonnées précise, en remplaçant la méthode de placement par un entier et ainsi utiliser les deux premiers arguments comme coordonnées 

### configAlgo.txt
Ce fichier permet la configuration des algorithmes d'apprentissage (contenu dans le fichier learn.py).
Chaque ligne correspond à un algo 

NomAlgo ConstructeurObjet
* nomAlgo : nom que l'on souhaite utiliser dans le fichier Node.txt
* ConstructeurObjet : constructeur de l'objet lié à l'algorithme d'apprentissage (fragment de code qui serra exécuté)

### sensitivity.txt
Fichier qui permet de configurer la sensibilité de l'antenne  

### TX.txt
Fichier de configuration de l'énergie consommée (en milliampère-Heure) en rapport avec la puissance d'émission 

23 valeurs (couverture des puissances de -2 à 20 dBm) 