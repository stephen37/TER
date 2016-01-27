from pprint import pprint
import codecs
import operator
import matplotlib.pyplot as plt
import nltk

def countsFile(file) :

    count = 0
    # Codecs ouvre et ferme le fichier par lui meme, il gere aussi l'encoding tout seul, C'est magique 
    with codecs.open(file, "r", encoding="utf8") as myfile:
        countWords = 0
        dictWords = {}
        
        for line in myfile :
            line = line.strip().split(" ") #remove \n
        
            for word in line :
                countWords += 1
                if word not in dictWords :
                    dictWords[word] = 1
                else :
                    dictWords[word] += 1

                
        print("nbWords", len(dictWords))
        #        pprint(dictWords)

        for w in dictWords :
            dictWords[w] = 1.0 * dictWords[w] / float(countWords)
            
        resDictWords = sorted(dictWords.items(), key = operator.itemgetter(1), reverse = True)
#        pprint(resDictWords)
        return resDictWords
        

dictFrequency = countsFile("./newsco.en")


def displayZipfLine(liste) :
    x = []
    y = []
    count = 1
    for i in liste :
        y.append(i[1])
        x.append(count)
        count += 1
    plt.loglog(x,y,"-")
    plt.savefig("./plot.png")
    plt.show()

displayZipfLine(dictFrequency)


def testNLTK() :
#    nltk.download('punkt')
    sentence = """At eight o'clock on Thursday morning
... Arthur didn't feel very good."""
    tokens = nltk.word_tokenize(sentence)

    print(tokens)

#testNLTK()
