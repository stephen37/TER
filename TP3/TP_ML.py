from __future__ import division

import collections
import operator
import numpy as np
from pprint import pprint

# Travail sur les donnees:
# On importe le corpus movie_reviews de nltk
from nltk.corpus import movie_reviews
# On recupere la liste des fichiers: chaque fichier contient une critique.
# Les 1000 premieres sont negatives, les 1000 suivantes sont positives.
Ids = movie_reviews.fileids()

# Creer une fonction count_words qui prend en entree une liste de mots et retourne un dictionnaire de comptes
# On pourra utiliser la fonction Counter du module collections 

def count_words(words):
    return collections.Counter(words)
    
# Creer une fonction combine_counts qui prend en entree une liste de dictionnaires et retourne un dictionnaire qui les combine
# Ses valeurs sont la somme des valeurs des dictionnaires d'entree

def combine_counts(counts):
    dicRes = {}
    for dico in counts :
        for word in dico :
            if word in dicRes :
                dicRes[word] += dico[word]
            else :
                dicRes[word] = dico[word]

    return dicRes



dico_test = [{"a": 1, "b": 3, "c" : 2}, {"a" : 3, "c": 5}]


"""
# Creer une fonction get_n_top_words qui prend en entree un dictionnaire et un entier n 
# et qui retourne la liste des n elements les plus frequents du dictionnaire
# On pourra utiliser la fonction itemgetter du module operator       
"""
def get_n_top_words(count, n):

    return collections.Counter(count).most_common(n)


# Creer une fonction get_top_values qui prend en entree un dictionnaire et une liste d'elements                                                                                            
# et renvoie la liste des valeurs associees a ces elements (dans le meme ordre)
       
def get_top_values(count, top_keys):
    listeRes = []

    for word in top_keys :
        if word in count :
            listRes.append(count[word])
        else :
            listRes.append(0)
    return listRes

    

    
# Creer une fonction normalize_counts qui prend en entree un dictionnaire de comptes
# et qui renvoie ce dictionnaire avec des valeurs normalisees 
def normalize_counts(count):
    total = 0.0
    
    for word in count :
        total += count[word]
    for word in count :
        count [word] /= total

    return count

# Combiner toutes ces fonctions dans une fonction get_counts_matrix qui prend en entree la liste Ids des noms de fichiers et un entier n
# et renvoie pour chacun d'entre eux la liste des n comptes normalises associes aux n mots les plus frequents de l'*ensemble* du corpus
# On peut recuperer les mots associes a un fichier fid sous forme de liste en utilisant la fonction
# movie_reviews.words(fileids=fid)
# On transformera la liste (iteree sur les fichiers) de liste de comptes en matrice a l'aide de la fonction
# numpy.array()
def get_counts_matrix(Ids, n):
    resultat = []
    dics = {}
    tmp = []
    for fid in Ids:
        liste_temp = movie_reviews.words(fileids=fid)
        count = count_words(liste_temp)
        dics[fid]  = count
        tmp.append(count)
    for e in dics:
         dics[e]  = normalize_counts(dics[e])
         
    dico_finale = combine_counts(tmp)
    dico_normalized = normalize_counts(dico_finale)
    top_words = get_n_top_words(dico_finale,n)
    for fich in dics:
        tab = []
        for mot in top_words:
            if dics[fich].has_key(mot[0]):
                tab.append(dics[fich][mot[0]])
            else:
                tab.append(0)
        resultat.append(tab)
    return resultat  

# Choisir n, et obtenir la matrice demandee pour le corpus movie_reviews.
# Verifier que la matrice a la taille attendu avec la methode shape
n = 10
test = np.matrix([[1, 2], [3, 4]])
M = np.array(get_counts_matrix(Ids, n))

# PCA
# On utilise la classe PCA du module mlab de matplotlib
# Les donnees projetees selon les nouvelles coordonnees sont contenues dans l'attribut PCA.Y

# La fonction plot_PCA vise a afficher les deux premiere dimensions des donnees dans les nouvelles coordonnees
# Completer la fonction plot_PCA en donnant en abcisse les points correspondant a la premiere dimension
# et en ordonnee les points correspondant a la deuxieme, separement pour les critiques positives et negatives
# Ensuite, l'utiliser sur les donnes. 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA

#PCA = Analyse en Composante Principale

#MR_PCA = PCA(M)
#Pour obtenir les donnees qui nous interesse, il faut utiliser MR_PCA.Y
"""
Ca nous met les valeurs dans un nouveau repere (n dimensions)
1ere dimension : nb de critiques (2000 points)
"""

"""
def plot_PCA(PCA_projection):
    #x_neg -> Prendre les 1000 premieres dimensions X  pour les commentaires negatifs
    #x_pos -> Prendre les 1000 premieres dimensions X  pour les commentaires positifs
    
    x_neg = MR_PCA.X[1000:]
    y_neg = #
    x_pos = #
    y_pos = #

    plt.figure()
    plt.xlabel('First component')
    plt.xlabel('Second component')
    plt.title("PCA applied to texts of the Movie_reviews corpus")
    plt.plot(x_neg,y_neg,'ro')
    plt.plot(x_pos,y_pos,'bo')
    plt.savefig("PCA.eps", format='eps', dpi=1000)

"""

"""

# Perceptron
# On utilise la classe Perceptron: elle comprend un constructeur lui fournit donnees et parametres
# Et deux methodes, pour l'entrainement, et pour le test
# Comprendre le code et completer la ligne manquante a l'aide de la procedure decrite dans les slides
# On evitera une boucle for en vectorialisant, c'est a dire en executant l'operation pour toutes les donnes
# a la fois a l'aide d'une operation matricielle.

class perceptron(object):
    def __init__(self, data, iterations, weights = None, learning_rate = 0.1):
        self.data = data
        self.it = iterations
        # Si on ne precise pas comment initialiser les parametres, le modele les met a zero
        if weights == None:
            self.weights = np.zeros(np.shape(data)[1])
        self.l_r = learning_rate

    def train(self, labels):
        counter = 0
        # On creer des variables pour retenir les meilleurs parametres
        best_weights = self.weights
        best_error_rate = len(labels)
        while (counter < self.it):
            # On utilise les parametres pour estimer les labels des donnes
            estimate_labels = -np.ones(len(labels))
            estimate_labels[np.dot(self.data, self.weights) > 0] = 1
            # On cree un tableau qui nous indiquera pour quels exemples notre modele a fait une erreur
            errors = np.arange(0,len(labels))[ labels != estimate_labels ]
            # Si il n'y a pas d'erreur, on arrete
            if not errors.size:
                break
            counter += 1
            # Completer le code pour mettre a jour l'attribut self.weights 
            self.weights += self.l_r * #
            # Si notre modele s'est ameliore, on garde en memoire ses parametres
            if (errors.size < best_error_rate):
                best_weights = self.weights
        # Une fois le nombre maximal d'iterations termine, on garde les meilleurs parametres
        self.weights = best_weights

    def test(self, labels):
        # On calcule la proportion d'erreurs sur les donnes
        estimate_labels = -np.ones(np.shape(labels))
        estimate_labels[np.dot(self.data, self.weights) > 0] = 1
        errors = np.ones(np.shape(labels))[ labels != estimate_labels ]
        return sum(errors)/len(labels)


# Une fois la classe completee, creer un vecteur contenant les labels, creer la classe associee a nos donnes,
# et entrainer et tester le perceptrons sur les labels.
"""


"""
y = 
model =
"""

"""
 
# Regression logistique
# On utilise la classe logistic_regression: elle est presque identique a la classe perceptron.
# La difference dans la procedure se trouve au niveau de la mise a jour: on doit calculer le gradient 
# et les utiliser pour mettre a jour les parametres.
# Completer le code, ici aussi en vectorialisant les calculs.

class logistic_regression(object):
    def __init__(self, data, iterations, weights = None ):
        self.data = data
        self.it = iterations
        if weights == None:
            self.weights = np.zeros(np.shape(data)[1])
        self.it_counter = 0

    def train(self, labels, learning_rate = 1.0):
        counter = 0
        while (counter < self.it):
            # Ici, calculer les log probabilite en utilisant les donnees et parametres
            log_probabilities = #
            self.it_counter += 1
            counter +=1
            # Ici, calculer le gradient a l'aide de la formule des slides
            gradient = #
            # Puis l'utiliser pour mettre a jour l'attribut self.weights
            self.weights += #

    def test(self, labels):
        log_probabilities = 1 / (1 + np.exp(-np.dot(self.data, self.weights)))
        estimate_labels = np.round(log_probabilities)
        errors = np.ones(np.shape(labels))[ labels != estimate_labels ]
        return sum(errors)/len(labels)

# Une fois la classe completee, creer une vecteur contenant les labels (qui sont differents des precedents! Pourquoi ?)
# creer la classe associee a nos donnes, et tester
# On fera cette fois plus attention a la procedure: il faudra separer les donnes d'entrainement et de test,
# et verifier la progression du modele periodiquement
"""
"""
y = 
model =


"""
# Comment evoluent les resultats si l'on utilise d'autres features ? On peut essayer d'utiliser des bigrams, des n-grams
# ou d'autres features plus specifique.
# Comment le PCA pourrait-il etre utile ici ? 



"""
"""
