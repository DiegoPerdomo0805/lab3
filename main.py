# Diego Perdomo, 20204
# Jose Gonzalez, 20335
# Universidad del Valle de Guatemala
# Inteligencia Artificial


from read import read_file
from model import model_bayes
import pandas as pd
# modeolo de bayes de sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


def main():
    spam_dict, ham_dict, p_spam, p_ham, vocab, test, val = read_file()
    accuracy_test , accuracy_val = model_bayes(spam_dict, ham_dict, p_spam, p_ham, vocab, test, val)

    file = pd.read_csv('entrenamiento.txt', sep='\t', header=None, names=['label', 'message'])
    file['message'] = file['message'].str.replace('[^\w\s]', '')
    file['message'] = file['message'].str.replace('\d+', '')
    file['message'] = file['message'].str.lower()
    
    x_train, x_test, y_train, y_test = train_test_split(file['message'], file['label'], random_state=1)

    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(x_train)

    clf = MultinomialNB().fit(X_train_vectorized, y_train)

    X_train_vectorized = vectorizer.transform(x_train)
    print('Accuracy ScikitLearn, training : ', clf.score(X_train_vectorized, y_train))

    X_test_vectorized = vectorizer.transform(x_test)
    print('Accuracy ScikitLearn, test : ', clf.score(X_test_vectorized, y_test))

    ###
    # nuestra exactitud en test es de 0.9532794249775381
    # nuestra exactitud en val es de 0.9560975609756097
    #Accuracy ScikitLearn, training :  0.9896931927133269
    #Accuracy ScikitLearn, test :  0.9841840402588066
    ###
    # La diferencia entre la exactitud de nuestro modelo y el de sklearn es de 0.0309053847187315
    # La diferencia entre la exactitud de nuestro modelo y el de sklearn es de 0.0280864792831969
    ###
    # SciKitLearn es un poco mejor que nuestro modelo, pero la diferencia es muy pequeña
    # por lo que podemos decir que nuestro modelo es bastante bueno
    # puede que sea por el hecho de que nuestro modelo no tiene en cuenta combinaciones de palabras que 
    # pueden ser importantes para la clasificación, pero en general es bastante bueno




main()