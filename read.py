import pandas as pd
import string
from sklearn.model_selection import train_test_split
import re


def clean(line):
    #trans = str.maketrans('','', '.,;:!?()[]{}<>')
    trans  = str.maketrans('','', string.punctuation)
    words = [word.translate(trans).upper() for word in line.split()]
    #print(words)

    return words


def read_file():
    lines = []

    with open('entrenamiento.txt', 'r') as f:
        for line in f:
            lines.append(line)


    # divide entre training, validation y test
    # 80% training, 10% validation, 10% test

    train, test = train_test_split(lines, test_size=0.2, random_state=42)
    train, val  = train_test_split(train, test_size=0.125, random_state=42)

    train = [re.sub(r'[^\w\s]', '', line) for line in train]
    val   = [re.sub(r'\d+', '', line) for line in train]


    # for line in lines: print(line)

    spam_list = [[]]
    ham_list  = [[]]

    s  = []
    h  = []

    # separar spam de ham

    for line in train:
        if line[0] == 's':
            s.append(line)
        else:
            h.append(line)

    number_of_spam = len(s)
    number_of_ham  = len(h)
    number_of_messages = number_of_spam + number_of_ham
    #print(number_of_spam, number_of_ham, number_of_messages)


    p_spam = number_of_spam / number_of_messages
    p_ham  = number_of_ham  / number_of_messages

    #print(p_spam, p_ham)

    # tamaño de vocabulario

    #vocab = set()
    vocab = []

    for line in train:
        for word in clean(line):
            if word not in vocab:
                #vocab.add(word)
                vocab.append(word)



    # limpiar spam
    for line in s:
        temp = clean(line)
        #print(temp)
        #temp.pop(0)
        temp = temp[1:]
        #print(temp)
        spam_list.append(temp)

    # limpiar ham
    for line in h:
        temp = clean(line)
        #print(temp)
        #temp.pop(0)
        temp = temp[1:]
        #sprint(temp)
        ham_list.append(temp)

    # diccionario de palabras con su frecuencia
    spam_dict = {}
    ham_dict  = {}

    # frecuencia de palabras en spam
    for line in spam_list:
        for word in line:
            if word in spam_dict:
                spam_dict[word] += 1
            else:
                spam_dict[word] = 1

    # frecuencia de palabras en ham
    for line in ham_list:
        for word in line:
            if word in ham_dict:
                ham_dict[word] += 1
            else:
                ham_dict[word] = 1
    
    #for e in spam_dict:
    #    #spam_dict[e] = spam_dict[e] / number_of_spam
    #    if e == '':
    #        print('WTF')
#
    #for e in ham_dict:
    #    #ham_dict[e] = ham_dict[e] / number_of_ham
    #    if e == '':
    #        print('WTF')
#
    #for e in vocab:
    #    if e == '':
    #        print('WTF')

    #return spam_dict, ham_dict
    return spam_dict, ham_dict, p_spam, p_ham, vocab, test, val


spam_dict, ham_dict, p_spam, p_ham, vocab, test, val = read_file()


#for e in spam:
#    print(e)
#
#print('\n\n\n--------------------------------------------------------------------------------------\n\n\n')
#
#for e in ham:
#    print(e)
#
#

