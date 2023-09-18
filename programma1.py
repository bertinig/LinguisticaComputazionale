# -*- coding: utf-8 -*-
import sys
import nltk

#La funzione eatrae i tokens, il PoS_tag e la lunghezza totale del corpus.
def TokensInfo(frasi):
    tokensTot = []
    tokensPoSTot = []
    lunghezzaTot = 0.0
    for frase in frasi: #ciclo for per tokenizzare le frasi, analizzare le Part of Speech e creare array di tokens
        tokens = nltk.word_tokenize(frase)
        tokensTot += tokens
        tokensPoS = nltk.pos_tag(tokens)
        tokensPoSTot += tokensPoS
        lunghezzaTot += len(tokens)
    return tokensTot, tokensPoSTot, lunghezzaTot

#La funzione calcola la lunghezza media delle frasi in termini di tokens.
def MediaFrasi(frasi):
    numFrasi = 0.0
    numTokens = 0.0
    for frase in frasi:
        numFrasi += 1
        tokens = nltk.word_tokenize(frase)
        numTokens += len(tokens)
    LunghezzaMediaFrase = numTokens / numFrasi
    return LunghezzaMediaFrase

#La funzione calcola la lunghezza media dei tokens in termini di caratteri.
def MediaCaratteri(frasi):
    numeroTokens = 0.0
    lunghezzaCar = 0.0
    for frase in frasi:
        tokens = nltk.word_tokenize(frase)
        for tok in tokens:
            numeroTokens += 1
            lunghezzaCar += len(tok)
    lungMediaCar = lunghezzaCar / numeroTokens
    return lungMediaCar

#La funzione estrae il numero di hapax presenti nei primi 1000 tokens.
def NumeroHapax(tokens):
    vocabolario = list(set(tokens)) #list crea un array dal set che restituisce le parole tipo (tokens senza ripetizioni)
    hapax = []
    for token in vocabolario:
        freqTok = tokens.count(token) #count restituisce il numero di occorrenze del token nell'array 
        if freqTok == 1:
            hapax.append(token)
    return len(hapax)

#La funzione calcola la grandezza del vocabolario e la ricchezza lessicale all'aumentare proporzionale del corpus di 500 tokens.
def VocabolarioTTR(tokens):
    for i in range(500, len(tokens), 500): #l'indice parte da 500 e per tutta la lunghezza del corpus aumenta di 500 tokens alla volta
        partTokens = tokens[0:i] #divido il corpus in parti, da 0 a i
        vocabolario = list(set(partTokens))
        TTR = len(vocabolario) / len(partTokens) #Type Token Ratio (indice di ricchezza lessicale)
        print("Nei primi", i, "tokens:")
        print("- la grandezza del vocabolario è di", len(vocabolario), "parole tipo,")
        print("- la ricchezza lessicale è di", TTR, ".")

#La funzione calcola la percentuale di distribuzione delle parole piane e delle funzionali presenti nei due file.
def DistribuzioneTermini(tokensPoSTot):
    totPiene = 0.0 
    totFunzionali = 0.0
    for tokensPoS in tokensPoSTot:
        if tokensPoS[1] in {"NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "RB", "RBR", "RBS", "WRB"}: #controllo che la PoS sia un sostantivo, verbo, avverbio e aggettivo 
            totPiene += 1 #sommo 1 per ogni PoS che è una parola piena
        if tokensPoS[1] in {"PRP", "PRP$", "WP", "WP$", "PDT", "DT", "WDT", "IN", "CC"}: #controllo che la PoS sia un pronome, articolo, preposizione e congiunzione
            totFunzionali += 1 #sommo 1 per ogni PoS che è una parola funzionale
    percentualePiene = float(totPiene/100) 
    percentualeFunzionali = float(totFunzionali/100) 
    return percentualePiene, percentualeFunzionali

def main(file1, file2):
    fileInput1 = open(file1, mode = "r", encoding = "utf-8")
    fileInput2 = open(file2, mode = "r", encoding = "utf-8")

    raw1 = fileInput1.read()
    raw2 = fileInput2.read()

    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    frasi1 = sent_tokenizer.tokenize(raw1)
    frasi2 = sent_tokenizer.tokenize(raw2)

    tokens1, tokensPoS1, lunghezzaTok1 = TokensInfo(frasi1)
    tokens2, tokensPoS2, lunghezzaTok2 = TokensInfo(frasi2)

    lunghezzaFrasi1 = len(frasi1)
    lunghezzaFrasi2 = len(frasi2)

    print("PROGRAMMA 1")
    print()
    print()
    print("a) Calcolo numero token e numero frasi:")

    print("Il file", file1, "è lungo", lunghezzaTok1, "tokens.")
    print("Il file", file2, "è lungo", lunghezzaTok2, "tokens.")
    
    if (lunghezzaTok1 > lunghezzaTok2):
        print ("Dunque,", file1, "è più lungo di", file2, ".")
    elif (lunghezzaTok1 < lunghezzaTok2):
        print ("Dunque,", file2, "è più lungo di", file1, ".")
    else:
        print("Dunque, i due file hanno la stessa lunghezza.")

    print()

    print("Il file", file1, "contiene", lunghezzaFrasi1, "frasi.")
    print("Il file", file2, "contiene", lunghezzaFrasi2, "frasi.")

    if (lunghezzaFrasi1 > lunghezzaFrasi2):
        print ("Dunque,", file1, "ha un numero maggiore di frasi.")
    elif (lunghezzaFrasi1 < lunghezzaFrasi2):
        print ("Dunque,", file2, "ha un numero maggiore di frasi.")
    else:
        print ("Dunque, i due file hanno lo stesso numero di frasi.")

    print()
    print()

    print("b) Calcolo della lunghezza media delle frasi e dei caratteri:")
    lunghezzaMedia1 = MediaFrasi(frasi1)
    lunghezzaMedia2 = MediaFrasi(frasi2)

    print("Il file", file1, "ha le frasi di lunghezza media di", lunghezzaMedia1, "tokens.")
    print("Il file", file2, "ha le frasi di lunghezza media di", lunghezzaMedia2, "tokens.")

    if (lunghezzaMedia1 > lunghezzaMedia2):
        print("Quindi, il file", file1, "ha una lunghezza media delle frasi maggiore.")
    elif (lunghezzaMedia1 < lunghezzaMedia2):
        print("Quindi, il file", file2, "ha una lunghezza media delle frasi maggiore.")
    else:
        print("Quindi, i file hanno la stessa lunghezza media delle frasi.")

    print()

    lungMedia1 = MediaCaratteri(frasi1)
    lungMedia2 = MediaCaratteri(frasi2)

    print("Il file", file1, "ha una lunghezza media delle parole di", lungMedia1, "caratteri.")
    print("Il file", file2, "ha una lunghezzza media delle parole di", lungMedia2, "caratteri.")

    if (lungMedia1 > lungMedia2):
        print("Quindi, la lunghezza media dei caratteri nel file", file1, "è maggiore.")
    elif (lungMedia1 < lungMedia2):
        print("Quindi, la lunghezza media dei caratteri nel file", file2, "è magggiore.")
    else:
        print("Quindi, la lunghezza media dei caratteri è la stessa in entrambi i file.")

    print()
    print()

    print("c) Calcolo hapax nei primi 1000 token:")

    tokensMille1 = tokens1[0:1000]
    tokensMille2 = tokens2[0:1000]
    lungHapaxMille1 = NumeroHapax(tokensMille1)
    lungHapaxMille2 = NumeroHapax(tokensMille2)

    print("Il file", file1, "contiene", lungHapaxMille1, "hapax nei primi 1000 tokens.")
    print("Il file", file2, "contiene", lungHapaxMille2, "hapax nei primi 1000 tokens.")

    if (lungHapaxMille1 > lungHapaxMille2):
        print("Perciò, il numero di hapax nei primi 1000 tokens del file", file1, "è maggiore.")
    elif (lungHapaxMille1 < lungHapaxMille2):
        print("Perciò, il numero di hapax nei primi 1000 tokens del file", file2, "è maggiore.")
    else:
        print("Perciò, il numero di hapax nei primi 1000 tokens è lo stesso in entrambi i file.")
    
    
    print()
    print()
    
    print("d) Calcolo vocabolario e ricchezza lessicale all'aumentare di 500 tokens:")

    print("Vocabolario del", file1)
    VocabolarioTTR(tokens1)
    print()
    print("Vocabolario del", file2)
    VocabolarioTTR(tokens2)

    print()
    print()
    
    print("e) Calcolo distribuzione in % delle parole:")

    percParolePiene1, percParoleFunz1 = DistribuzioneTermini(tokensPoS1)
    percParolePiene2, percParoleFunz2 = DistribuzioneTermini(tokensPoS2)

    print("Il file", file1, "ha il", percParolePiene1, "% di parole piene e il", percParoleFunz1, "% di parole funzionali.")
    print("Il file", file2, "ha il", percParolePiene2, "% di parole piene e il", percParoleFunz2, "% di parole funzionali.")
    print("Quindi,")
    if (percParolePiene1 > percParolePiene2):
        print("il file", file1, "ha una percentuale maggiore di parole piene,")
    elif (percParolePiene1 < percParolePiene2):
        print("il file", file2, "ha una percentuale maggiore di parole piene,")
    else:
        print("i file hanno la stessa percentuale di parole piene.")

    if (percParoleFunz1 > percParoleFunz2):
        print("il file", file1, "ha una percentuale maggiore di parole funzionali.")
    elif (percParoleFunz1 < percParoleFunz2):
        print("il file", file2, "ha una percentuale maggiore di parole funzionali.")
    else:
        print("i file hanno la stessa percentuale di parole funzionali.")

main(sys.argv[1], sys.argv[2])
