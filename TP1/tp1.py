import codecs
from string import ascii_letters
import operator


def mean(list):
    sum = 0
    for element in list:
        sum += element

    return sum/len(list)

listNumber = [1,5,8,7,2]

print(mean(listNumber))

def variance(list):
    meanL = mean(list)
    sum = 0
    for element in list:
        sum += (element - meanL)**2

    return sum/len(list)

print(variance(listNumber))


authors = ["Sartre", "Camus", "Bourdieu", "Sartre", "Sartre"]

def nbBooks(list) :
    dicoAuth = {}
    for auth in list :
        if auth not in dicoAuth :
            dicoAuth[auth] = 1
        else :
            dicoAuth[auth] +=1
    for el in dicoAuth :
        if dicoAuth[el] == 1:
            # On peut aussi ecrire print " {0} wrote 1 book".format(el)"
            print el, "wrote", dicoAuth[el], "book"
        else :
            print el, "wrote", dicoAuth[el], "books"
        
nbBooks(authors)

def countsFile() :

    count = 0
    # Codecs ouvre et ferme le fichier par lui meme, il gere aussi l'encoding tout seul, C'est magique 
    with codecs.open("./english.txt", "rt", encoding="utf8") as myfile:
        countWords = 0
        countDifferentWords = 0
        dictWords = {}
        listWords = []
        longestWordPerLine = []
        maxSizeWord = 0
        countLine = 0
        dictChars = {}
        countChars = 0
        for line in myfile :
            line = line.strip().split(" ") #remove \n
            longestWordPerLine.append("")

            for word in line :
                for char in word :
                    countChars += 1
                    if char not in dictChars :
                        dictChars[char] = 1
                    else :
                        dictChars[char] +=1
                        
                    #                if word in ascii_letters:
                countWords +=1
                if word not in listWords :
                    listWords.append(word)
                    countDifferentWords += 1
                if len(word) > maxSizeWord :
                    maxSizeWord = len(word)
                    longestWordPerLine[countLine] = word
            maxSizeWord = 0
            countLine +=1
        
        print "nbWords", countWords
        print "nbDifferentWords", countDifferentWords
        print "longest word", longestWordPerLine

        for items in dictChars :
            dictChars[items] = 1.0 * dictChars[items] / countChars

            
    print "frequency", sorted(dictChars.items(), key = operator.itemgetter(1))

countsFile()
