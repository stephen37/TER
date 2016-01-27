import string
import codecs
from pprint import pprint
from nltk.util import ngrams
import numpy as np

#Returns a dictionnary, the key is a word and the value is the probability to have that word in the corpus.
def proba1Gram(file_name) :
    dico1Gram = {}
    
    with codecs.open(file_name, "r", encoding="utf8") as myfile:
        countTotal = 0
        
        for line in myfile:
            line = line.strip() #Remove the \n
            mots = line.split()
            for mot in mots :
                countTotal +=1
                if mot not in dico1Gram :
                    dico1Gram[mot] = 1
                else :
                    dico1Gram[mot] += 1



    
    for mot in dico1Gram :
        dico1Gram[mot] /= 1.0 * countTotal
    return dico1Gram
   
result1 = proba1Gram("./newsco.en")

#pprint(result1)
#result2 = probaNGram("./count_2w.txt")


# Returns a dictionnary, the first element is a word, the key associated with this element is the following word and then the value is the probability that the second word follows the first one.

def proba2gram(file_name) :
    dico2Gram = {}
    
    with codecs.open(file_name, "r", encoding="utf8") as myfile:
        countTotal = 0
        
        for line in myfile:
            line = line.strip() #Remove the \n
            mots = line.split()

            # We create bigrams using the nltk library
            bigram = list(ngrams(mots, 2))

            for mot in bigram :
                
                mot1 = mot[0]
                mot2 = mot[1]
                count = 1
                if mot1 not in dico2Gram :
                    dico2Gram[mot1] = {mot2 : count}
                else :
                    count +=1
                    dico2Gram[mot1][mot2] = count

            mot1 = mot[0]
            mot2 = mot[1]
            count = 1
            if mot1 not in dico2Gram :
                dico2Gram[mot1] = {mot2 : count}
            else :
                count +=1
                dico2Gram[mot1][mot2] = count

        for mot in dico2Gram :
            sumN = sum(dico2Gram[mot].values())
            for mot2 in dico2Gram[mot] :
                dico2Gram[mot][mot2] /= sumN * 1.0

        return dico2Gram


result2 = proba2gram("./newsco.en")
#pprint(result2)


'''
Question 2 : Il faut faire le produit des probabilites
'''


#Returns the probability of have the given sentence is the corpus.
def proba3_1gram(phrase):

    #motsReplace = phrase.replace(",", " ,").replace(".", " .").replace("!", " !").replace("?", " ?").lower().split()
    motsReplace = phrase.lower().split()
    res = 1.0

    for word in motsReplace :
#        print("result pour le mot ", word , " = ", result [word])
        res *= float(result1[word])

    return res


text1 = "i want that house"
#print("1ere phrase", proba3_1gram(text1))


# Returns the probability of having the given sentence in the corpus but in 2-gram.
def proba3_2gram(phrase) :
    motsReplace = phrase.split()

    #Mot precedent
    historique = motsReplace[0]
    # On calcul la proba du mot 
    res = result1[historique]
    for word in motsReplace[1:] :
        if historique in result2 :
            if word in result2[historique] :
                #On multiplie par la proba du mot suivi par le mot precdent
                res *= result2[historique][word]
            else :
                result2[historique][word] = 1.0 / len(result2[historique])
                res *= result2[historique][word]
        else :
            result2[historique] = {}
            result2[historique][result1[word]] = 1.0
            res *= result2[historique][result1[word]]
        historique = word

    return res
        

text2 = "there is a fucking house"
#print("2-gram", proba3_2gram(text2))



def sample_from_discrete_distrib(distrib):
    words, probas = list(zip(*distrib.items()))
    return np.random.choice(words, p=probas)

def generation(corpus) :
    with codecs.open(corpus, "r", encoding="utf-8") as myfile :
        for line in myfile:
            line = line.strip() #Remove the \n
            mots = line.split()
            # We create bigrams using the nltk library
            bigram = list(ngrams(mots, 2))
            h = "I"
            word = ""
            res = "I"

            for i in range(100) :
                if h not in result2.keys() :
                    break
                else :
                    word = sample_from_discrete_distrib(result2[h])
                    h = word
                    res += " " + word
                    
        return res




print(generation("./newstest2009.en"))

'''TODO :
0) Generer des phrases (Done)

1) Il faut lisser les valeurs. Pour lisser les valeurs,on attribue aux mots et n-grams jamais vus un
peu de la densite de probabilite. (Done)

2) Prendre la phrase la plus probable avec random.permutation -> renvoyer la meilleure permutation

'''
