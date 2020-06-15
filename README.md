# Image Inpainting via Sparse Representation

## Introduction:
L’*inpainting* en image s’attache à la reconstruction d’images détériorées ou au remplissage de parties manquantes (éliminer une personne ou un objet d’une image par exemple). Cette partie est consacrée à l’implémentation d’une technique d’*inpainting* présentée dans utilisant le Lasso et sa capacité à trouver une solution parcimonieuse en termes de poids. Ces travaux sont inspirés des recherches en Apprentissage de dictionnaire et représentation parcimonieuse pour les signaux. L’idée principale est de considérer qu’un signal complexe peut être décomposé comme une somme pondérée de signaux élémentaires, appelés atomes.

L’analyse en composante principale (ACP) est un exemple de ce type de décomposition, mais du fait des contraintes d’orthogonalité, les atomes (ou la base canonique) ainsi inférés ne sont pas redondants - chacun représente le plus d’information possible et la reconstitution du signal est unique.Dans le cas de l’*inpainting*, l’objectif est au contraire d’obtenir un dictionnaire fortement redondant. La difficulté est donc double : réussir à construire un dictionnaire d’atomes - les signaux élémentaires - et trouver un algorithme de décomposition/recomposition d’un signal à partir des atomes du dictionnaire,i.e. trouver des poids parcimonieux sur chaque atome tels que la combinaison linéaire des atomes permette d’approcher le signal d’origine. Ici, nous étudions essentiellement une solution pour la deuxième problématique - la reconstruction - à l’aide de l’algorithme du LASSO.

## Résultats:
### Image Bruité:
![Image of Lenna](https://github.com/YacineAll/Image_Inpainting_via_Sparse_Representation/raw/master/output/lennaNoised.png)
### Reconstruction d'une région:
![Image of Reconstruct](https://github.com/YacineAll/Image_Inpainting_via_Sparse_Representation/raw/master/output/regionReconstructed.png)

## Discussion:
### Avantages:
* Cela donne de très bons résultats qui sont réalistes pour nous, il suffit de faire un bon réglage des paramètres.
* Modèle simple à comprendre, intuitif et interprétable.
### Inconvénients:
* Il est très lent à exécuter même avec de petites images (300, 300), il a donc une complexité médiocre.
* Plus la résolution et la qualité de l'image sont élevées, plus les résultats sont mauvais.
