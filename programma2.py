#-*- coding: utf-8 -*-
import sys
import nltk
import math
from nltk import FreqDist
from nltk import bigrams, trigrams
from nltk.chunk import ne_chunk

#La funzione estrae i tokens, il PoS_tag e la lunghezza totale del corpus.
def TokensPoSInfo(frasi):
    tokens = []
    tokensPoSTot = []
    for frase in frasi: #ciclo for per tokenizzare le frasi, analizzare le Part of Speech e creare array di tokens
        fraseTokenizzata = nltk.word_tokenize(frase)
        tokens += fraseTokenizzata
        tokensPoS = nltk.pos_tag(fraseTokenizzata)
        tokensPoSTot += tokensPoS
    return tokens, tokensPoSTot

#La funzione estrae, in ordine di frequenza decrescente, le 10 PoS più frequenti.
def DieciPoSFreq(tokensPoSTot):
    PoS = []
    for tokens in tokensPoSTot:
        PoS.append(tokens[1]) #estraggo dall'array tokensPoSTot (token, tag) solo il token
    distrPoS = nltk.FreqDist(PoS) #FreqDist crea un oggetto contenente le info sulla frequenza dei tokens nel corpus
    freqPoS = distrPoS.most_common(10) #most_common crea un array con (in questo caso) i 10 PoS più frequenti
    return freqPoS

#La funzione estrae, in ordine di frequenza decrescente, le 10 PoS di bigrammi più frequenti.
def DieciBigrammiFreq(tokensPoSTot):
    bigrammiPoS = bigrams(tokensPoSTot) #bigrams restituisce un iteratore di bigrammi
    distrBigrammi = FreqDist(bigrammiPoS)
    freqBigrammi = distrBigrammi.most_common(10)
    return freqBigrammi

#La funzione estrae, in ordine di frequenza decrescenre, le 10 PoS di trigrammi più frequenti
def DieciTrigrammiFreq(tokensPoSTot):
    trigrammiPoS = trigrams(tokensPoSTot) #trigrams restituisce un iteratore di trigrammi
    distrTrigrammi = FreqDist(trigrammiPoS)
    freqTrigrammi = distrTrigrammi.most_common(10)
    return freqTrigrammi

#La funzione calcola e restituisce in ordine di frequenza decrescente i 20 aggettivi più frequenti
def VentiAggettivi(tokensPoSTot):
    aggettiviPoS = []
    for tokens in tokensPoSTot:
        tagPoS = tokens[1]
        if ((tagPoS == "JJ") or (tagPoS == "JJR") or (tagPoS == "JJS")): #controllo che la PoS sia un aggettivo
            aggettiviPoS += tokens #aggiungo il token all'array che contiene gli aggettivi
    distrAggettivi = FreqDist(aggettiviPoS)
    freqAggettivi = distrAggettivi.most_common(20)
    return freqAggettivi

#La funzione calcola e restituisce in ordine di frequenza decrescente i 20 avverbi più frequenti
def VentiAvverbi(tokensPoSTot):
    avverbiPoS = []
    for tokens in tokensPoSTot:
        tagPoS = tokens[1]
        if ((tagPoS == "RB") or (tagPoS == "RBR") or (tagPoS == "RBS") or (tagPoS == "WRB")): #controllo che la PoS sia un avverbio
            avverbiPoS += tokens #aggiungo il token all'array che contiene gli avverbi
    distrAvverbi = FreqDist(avverbiPoS)
    freqAvverbi = distrAvverbi.most_common(20)
    return freqAvverbi

#La funzione, dato un dizionario, restituisce una lista di coppie (chiave, valore), ordinate per valore 
def Ordina(dict):
    return sorted(dict.items(), key = lambda x: x[1], reverse = True)

#La funzione estrae i 20 bigrammi Aggettivo+Sostantivo con frequenza massima dove ogni token ricorre più di 3 volte nel corpus 
def FreqMassima(tokensPoSTot):
    bigrammiDict = {} #creo un dizionario 
    bigrammi = list(bigrams(tokensPoSTot)) #list crea un array di bigrammi
    bigrammiSet = list(set(bigrammi)) #creo un array dal set di bigrammi (univoci)
    for bigramma in bigrammiSet:
        token1 = bigramma[0] #estaggo i token dal bigramma
        freqToken1 = tokensPoSTot.count(token1) 
        token2 = bigramma[1]
        freqToken2 = tokensPoSTot.count(token2)
        if ((token1[1] in ["JJ", "JJS", "JJR"]) and (token2[1] in ["NN", "NNP", "NNS", "NNPS"])): #controllo che il 1° token sia un aggettivo seguito da un sostantivo
            if (freqToken1 > 3 and freqToken2 > 3): #controllo che entrambi i tokens occorrano nel corpus più di 3 volte
                freqBigramma = bigrammi.count(bigramma) #calcolo la frequenza del bigramma
                bigrammiDict[bigramma] = freqBigramma

    bigrammiOrdDict = Ordina(bigrammiDict) #ordino in maniera decrescente il dizionario
    numeroCiclo = 0
    for elem in bigrammiOrdDict:
        freqToken1 = tokensPoSTot.count(elem[0][0]) 
        freqToken2 = tokensPoSTot.count(elem[0][1])
        print(numeroCiclo+1, "° bigramma:", elem[0], ", la sua frequenza massima è ", elem[1])
        print("- La frequenza relativa di", elem[0][0], "è", freqToken1)
        print("- La frequenza relativa di", elem[0][1], "è", freqToken2)
        numeroCiclo += 1
        if numeroCiclo >= 20: 
            break #stampo solo i primi 20 elementi

#La funzione estrae i 20 bigrammi Aggettivo+Sostantivo con probabilità condizionata massima dove ogni token ricorre più di 3 volte nel corpus 
def ProbCondizionata(tokensPoSTot):
    bigrammiDict = {}
    bigrammi = list(bigrams(tokensPoSTot))
    bigrammiSet = list(set(bigrammi))
    for bigramma in bigrammiSet:
        token1 = bigramma[0]
        freqToken1 = tokensPoSTot.count(token1)
        token2 = bigramma[1]
        freqToken2 = tokensPoSTot.count(token2)
        if ((token1[1] in ["JJ", "JJS", "JJR"]) and (token2[1] in ["NN", "NNP", "NNS", "NNPS"])): 
            if (freqToken1 > 3 and freqToken2 > 3): 
                freqBigramma = bigrammi.count(bigramma)
                probCond = freqBigramma / freqToken1 #calcolo la probabilità condizionata
                bigrammiDict[bigramma] = probCond

    bigrammiOrdDict = Ordina(bigrammiDict)
    numeroCiclo = 0
    for elem in bigrammiOrdDict:
        print(numeroCiclo+1, "° bigramma:", elem[0], ", la sua probabilità condizionata è ", elem[1])
        numeroCiclo += 1
        if numeroCiclo >= 20:
            break

#La funzione estrae i 20 bigrammi Aggettivo+Sostantivo con forza associativa massima dove ogni token ricorre più di 3 volte nel corpus 
def LocalMutualInformation(tokensPoSTot):
    bigrammiDict = {}
    bigrammi = list(bigrams(tokensPoSTot))
    bigrammiSet = list(set(bigrammi))
    for bigramma in bigrammiSet:
        token1 = bigramma[0]
        freqToken1 = tokensPoSTot.count(token1)
        token2 = bigramma[1]
        freqToken2 = tokensPoSTot.count(token2)
        if ((token1[1] in ["JJ", "JJS", "JJR"]) and (token2[1] in ["NN", "NNP", "NNS", "NNPS"])):
            if (freqToken1 > 3 and freqToken2 > 3):
                freqBigramma = bigrammi.count(bigramma)
                probToken1 = freqToken1 / len(tokensPoSTot)
                probToken2 = freqToken2 / len(tokensPoSTot)
                probBigramma = freqBigramma / len(tokensPoSTot)
                LMI = freqBigramma * math.log(probBigramma / (probToken1 * probToken2)) #calcolo la Local Mutual Information
                bigrammiDict[bigramma] = LMI

    bigrammiOrdDict = Ordina(bigrammiDict)
    numeroCiclo = 0
    for elem in bigrammiOrdDict:
        print(numeroCiclo+1, "° bigramma:", elem[0], ", la sua LMI è ", elem[1])
        numeroCiclo += 1
        if numeroCiclo >= 20:
            break
        
#La funzione estrae le frasi con probabilità più alta e più bassa
#ogni frase deve avere almeno 6 tokens e non più di 25
#ogni token deve essere presente nel corpus almeno 2 volte
def FrasiMaxVenticinque(frasi, tokens):
    massimoFreq = 0
    minimoFreq = 500
    fraseMassima = ""
    fraseMinima = ""
    for frase in frasi:
        fraseTokenizzata = nltk.word_tokenize(frase)
        tokenFraseVerifica = False #inizializzo la var a false
        sommaOccorrenze = 0
        for token in fraseTokenizzata:
            freqToken = tokens.count(token)
            sommaOccorrenze += freqToken
            if (freqToken < 2):
                tokenFraseVerifica = True #la var prende true se trova tokens con frequenza < 2
                break 
        if (tokenFraseVerifica): #se la var è true salta la frase e passa alla prossima
            continue
        if (len(fraseTokenizzata) >= 6 and len(fraseTokenizzata) < 25): #controllo il num di tokens presenti nella frase
            mediaFreq = sommaOccorrenze / len(fraseTokenizzata)
            if (mediaFreq > massimoFreq):
                massimoFreq = mediaFreq
                fraseMassima = fraseTokenizzata
            if (mediaFreq < minimoFreq):
                minimoFreq = mediaFreq
                fraseMinima = fraseTokenizzata
    return fraseMassima, massimoFreq, fraseMinima, minimoFreq

#La funzione calcola attraverso un modello markoviano di II ordine la frase con probabilità più alta  
def FraseProbabileMarkovDue(frasi, tokens):
    probMassima = 0
    fraseMassima = ""
    vocabolario = list(set(tokens))
    for frase in frasi:
        fraseTokenizzata = nltk.word_tokenize(frase)
        tokenFraseVerifica = False
        for token in fraseTokenizzata:
            freqToken = tokens.count(token)
            if (freqToken < 2):
                tokenFraseVerifica = True
                break
        if (tokenFraseVerifica):
            continue
        if (len(fraseTokenizzata) >= 6 and len(fraseTokenizzata) < 25):
            trigrammi = list(trigrams(tokens))
            bigrammi = list(bigrams(tokens))
            probTot = 0
            for trigramma in trigrammi:
                freqTrigramma = trigrammi.count(trigramma)
                bigramma = (trigramma[0], trigramma[1])
                freqBigramma = bigrammi.count(bigramma)
                probTot += (freqTrigramma) / (freqBigramma + len(vocabolario)) #formula per il calcolo del mod di Markow di ordine 2
            probMedia = probTot / len(trigrammi) 
            if (probMedia > probMassima): #trovo la frase con prob più alta
                probMassima = probMedia
                fraseMassima = frase
    return fraseMassima, probMassima
                
#La funzione calcola e resituisce i 15 nomi propri di persona più frequenti in ordine di frequenza decrescente
def QuindiciNomiP(NamedEntity):
    nomiPropriPoS = []
    for nodo in NamedEntity:
        NE = ''
        if hasattr(nodo, 'label'): #controlla se il nodo è intermedio o foglia
            if nodo.label() in ["PERSON"]: #estrae l'etichetta del nodo
                for partNE in nodo.leaves(): #ottengo la lista delle foglie del nodo intermedio
                    NE += partNE[0]
                nomiPropriPoS.append(NE)
    distrNomiP = FreqDist(nomiPropriPoS)
    freqNomiP = distrNomiP.most_common(15)
    return freqNomiP
    

def main(file1, file2):
    fileInput1 = open(file1, mode = "r", encoding = "utf-8")
    fileInput2 = open(file2, mode = "r", encoding = "utf-8")

    raw1 = fileInput1.read()
    raw2 = fileInput2.read()

    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    frasi1 = sent_tokenizer.tokenize(raw1)
    frasi2 = sent_tokenizer.tokenize(raw2)

    tokens1, analisiTestoPoS1 = TokensPoSInfo(frasi1)
    tokens2, analisiTestoPoS2 = TokensPoSInfo(frasi2)

    PoSFrequenti1 = DieciPoSFreq(analisiTestoPoS1)
    PoSFrequenti2 = DieciPoSFreq(analisiTestoPoS2)

    print("PROGRAMMA2 2")
    print()
    print()

    print("1.a) Estrazione 10 PoS più frequenti in ordine decrescente")

    
    print("- 10 PoS più frequenti nel file", file1)
    for i in range(0, 10):
        print(i+1, "° PoS:", PoSFrequenti1[i][0], "con frequenza", PoSFrequenti1[i][1])
    print()
    print("- 10 PoS più frequenti nel file", file2)
    for i in range(0, 10):
        print(i+1, "° PoS:", PoSFrequenti2[i][0], "con frequenza", PoSFrequenti2[i][1])

    print()
    print()

    BigrammiFreq1 = DieciBigrammiFreq(analisiTestoPoS1)
    BigrammiFreq2 = DieciBigrammiFreq(analisiTestoPoS2)

    print("1.b) Estrazione 10 bigrammi di PoS più frequenti in ordine decrescente")

    print("- 10 bigrammi PoS più frequenti nel file", file1)
    for i in range(0, 10):
        print(i+1, "° bigramma di PoS", BigrammiFreq1[i][0], "con frequenza", BigrammiFreq1[i][1])
    print()
    print("- 10 bigrammi PoS più frequenti nel file", file2)
    for i in range(0, 10):
        print(i+1, "° bigramma di PoS", BigrammiFreq2[i][0], "con frequenza", BigrammiFreq2[i][1])

    print()
    print()

    TrigrammiFreq1 = DieciTrigrammiFreq(analisiTestoPoS1)
    TrigrammiFreq2 = DieciTrigrammiFreq(analisiTestoPoS2)

    print("1.c) Estrazione 10 trigrammi di PoS più frequenti in ordine decrescente")

    print("- 10 trigrammi PoS più frequenti nel file", file1)
    for i in range(0, 10):
        print(i+1, "° trigramma di PoS", TrigrammiFreq1[i][0], "con frequenza", TrigrammiFreq1[i][1])
    print()
    print("- 10 trigrammi PoS più frequenti nel file", file2)
    for i in range(0, 10):
        print(i+1, "° trigramma di PoS", TrigrammiFreq2[i][0], "con frequenza", TrigrammiFreq2[i][1])

    print()
    print()

    AggettiviFreq1 = VentiAggettivi(analisiTestoPoS1)
    AggettiviFreq2 = VentiAggettivi(analisiTestoPoS2)

    print("1.d) Estrazione 20 aggettivi e 20 avverbi PoS più frequenti in ordine decrescente")

    print("- 20 aggettivi PoS più frequenti nel file", file1)
    for i in range(0, 20):
        print(i+1, "° aggettivo più frequente è:", AggettiviFreq1[i][0], "con frequenza", AggettiviFreq1[i][1])
    print()
    print("- 20 aggettivi PoS più frequenti nel file", file2)
    for i in range(0, 20):
        print(i+1, "° aggettivo più frequente è:", AggettiviFreq2[i][0], "con frequenza", AggettiviFreq2[i][1])

    print()
    print()

    AvverbiFreq1 = VentiAvverbi(analisiTestoPoS1)
    AvverbiFreq2 = VentiAvverbi(analisiTestoPoS2)
    
    print("- 20 avverbi PoS più frequenti nel file", file1)
    for i in range(0, 20):
        print(i+1, "° avverbio più frequente è", AvverbiFreq1[i][0], "con frequenza", AvverbiFreq1[i][1])
    print()
    print("- 20 avverbi PoS più frequenti nel file", file2)
    for i in range(0, 20):
        print(i+1, "° avverbio più frequente è", AvverbiFreq2[i][0], "con frequenza", AvverbiFreq2[i][1])

    print()
    print()

    print("2) Estrazione 20 bigrammi composti da aggettivo e sostantivo dove ogni token ha frequenza > 3 con:")

    print("2.a) frequenza massima")
    print("- File", file1, ":")
    FreqMassima(analisiTestoPoS1)
    print()
    print("- File", file2, ":")
    FreqMassima(analisiTestoPoS2)

    print()

    print("2.b) probabilità condizionata massima")
    print("- File", file1, ":")
    ProbCondizionata(analisiTestoPoS1)
    print()
    print("- File", file2, ":")
    ProbCondizionata(analisiTestoPoS2)

    print()

    print("2.c) forza associativa massima")
    print("- File", file1, ":")
    LocalMutualInformation(analisiTestoPoS1)
    print()
    print("- File", file2, ":")
    LocalMutualInformation(analisiTestoPoS2)
    

    print()
    print()

    FraseMax1, FreqMax1, FraseMin1, FreqMin1 = FrasiMaxVenticinque(frasi1, tokens1)
    FraseMax2, FreqMax2, FraseMin2, FreqMin2 = FrasiMaxVenticinque(frasi2, tokens2)

    print("3) Estrazione frasi con un num di token tra 6 e 25, in cui ogni token ricorre almeno 2 volte nel corpus con:")
    print("3.a) media di distribuzione dei token più alta e più bassa")
    print("Nel file", file1, "la frase con distribuzione dei token")
    print("- più alta è:", FraseMax1, "con frequenza media dei token di", FreqMax1)
    print("- più bassa è:", FraseMin1, "con frequenza media dei token di", FreqMin1)
    print()
    print("Nel file", file2, "la frase con distribuzione dei token")
    print("- più alta è:", FraseMax2, "con frequenza media dei token di", FreqMax2)
    print("- più bassa è:", FraseMin2, "con frequenza media dei token di", FreqMin2)    
    
    print()

    FraseProb1, Prob1 = FraseProbabileMarkovDue(frasi1, tokens1)
    FraseProb2, Prob2 = FraseProbabileMarkovDue(frasi2, tokens2)

    print("3.b) probabilità più alta.")
    print("Nel file", file1, "la frase con probabilità più alta è", FraseProb1, "con probabilità", Prob1)
    print()
    print("Nel file", file2, "la frase con probabilità più alta è", FraseProb2, "con probabilità", Prob2)
    
    print()
    print()

    NamedEntity1 = ne_chunk(analisiTestoPoS1)
    NamedEntity2 = ne_chunk(analisiTestoPoS2)
    
    NomiPFreq1 = QuindiciNomiP(NamedEntity1)
    NomiPFreq2 = QuindiciNomiP(NamedEntity2)

    print("4) Estrazione dei 15 nomi propri di persona più frequenti.")

    print("- 15 nomi propri di persona più frequenti nel file", file1)
    for i in range(0, 15):
        print(i+1, "° nome più frequente è", NomiPFreq1[i][0], ", con frequenza", NomiPFreq1[i][1])
    print()
    print("- 15 nomi proprio di persona più frequenti nel file", file2)
    for i in range(0, 15):
        print(i+1, "° nome più frequente è", NomiPFreq2[i][0], ", con frequenza", NomiPFreq2[i][1])
    
main(sys.argv[1], sys.argv[2])
