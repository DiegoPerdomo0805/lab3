from read import clean
import math

def model_bayes(spam_dict, ham_dict, p_spam, p_ham, vocab, test, val):



    false_positive_spam = 0
    false_positive_ham = 0
    true_positive_spam = 0
    true_positive_ham = 0

    # bayes y laplace smoothing

    # probabilidad de spam dado una palabra
    # Laplace smoothing
    # p(w|spam) = (count(w, spam) + 1) / (count(spam) + |V|)
    # donde |V| es el tamaño del vocabulario
    p_spam_word = {}
    for word in vocab:
        p_spam_word[word] = 0
        if word in spam_dict:
            p_spam_word[word] = (spam_dict[word] + 1) / (len(spam_dict) + len(vocab))
        else:
            p_spam_word[word] = 1 / (len(spam_dict) + len(vocab))

    #print("------------------")
    #for e in p_spam_word:
    #    print(e, p_spam_word[e])
    #print("------------------")


    # probabilidad de ham dado una palabra
    # Laplace smoothing
    # p(w|ham) = (count(w, ham) + 1) / (count(ham) + |V|)
    # donde |V| es el tamaño del vocabulario
    p_ham_word = {}
    for word in vocab:
        p_ham_word[word] = 0
        if word in ham_dict:
            p_ham_word[word] = (ham_dict[word] + 1) / (len(ham_dict) + len(vocab))
        else:
            p_ham_word[word] = 1 / (len(ham_dict) + len(vocab))

    #print("------------------")
    #for e in p_ham_word:
    #    print(e, p_ham_word[e])
    #print("------------------")

    # probabilidad de spam o ham dado un mensaje
    # p(spam|message) = p(spam) * p(w1|spam) * p(w2|spam) * ... * p(wn|spam)
    # si una palabra no está en el vocabulario, se asume que su probabilidad es 1 / (count(spam) + |V|)
    # p(ham|message) = p(ham) * p(w1|ham) * p(w2|ham) * ... * p(wn|ham)
    # si una palabra no está en el vocabulario, se asume que su probabilidad es 1 / (count(ham) + |V|)
    # si p(spam|message) > p(ham|message), el mensaje es spam
    # si p(spam|message) < p(ham|message), el mensaje es ham
    # si p(spam|message) = p(ham|message), el mensaje es spam

    # si se clasifica un mensaje como spam y es spam, se actualiza la probabilidad de spam
    # si se clasifica un mensaje como ham y es ham, se actualiza la probabilidad de ham
    # si se clasifica un mensaje como spam y es ham, se actualiza la probabilidad de ham
    # si se clasifica un mensaje como ham y es spam, se actualiza la probabilidad de spam

    for line in test:
        temp = clean(line)
        answer = temp[0]
        temp = temp[1:]
        
        #print(answer)
        p_s = 1
        p_h = 1
        for word in temp:
            if word not in vocab:
                p_spam_word[word] = 1 / (len(spam_dict) + len(vocab))
                p_ham_word[word]  = 1 / (len(ham_dict)  + len(vocab))
                vocab.append(word)
                #print(word, p_spam_word[word], p_ham_word[word])

            p_s *= p_spam_word[word]
            p_h *= p_ham_word[word]
        p_s *= p_spam
        p_h *= p_ham

        

        if p_s > p_h or p_s == p_h:
            if answer == "SPAM":
                true_positive_spam += 1
            else:
                false_positive_spam += 1
        else:
            if answer == "HAM":
                true_positive_ham += 1
            else:
                false_positive_ham += 1

    test = {"true_positive_spam": true_positive_spam, "false_positive_spam": false_positive_spam, "true_positive_ham": true_positive_ham, "false_positive_ham": false_positive_ham}
    
    # NOW VALIDATION
    false_positive_spam = 0
    false_positive_ham = 0
    true_positive_spam = 0
    true_positive_ham = 0


    for line in val:
        temp = clean(line)
        answer = temp[0]
        temp = temp[1:]
        
        #print(answer)
        p_s = 1
        p_h = 1
        for word in temp:
            if word not in vocab:
                p_spam_word[word] = 1 / (len(spam_dict) + len(vocab))
                p_ham_word[word]  = 1 / (len(ham_dict)  + len(vocab))
                vocab.append(word)
                #print(word, p_spam_word[word], p_ham_word[word])

            p_s *= p_spam_word[word]
            p_h *= p_ham_word[word]
        p_s *= p_spam
        p_h *= p_ham

        

        if p_s > p_h or p_s == p_h:
            if answer == "SPAM":
                true_positive_spam += 1
            else:
                false_positive_spam += 1
        else:
            if answer == "HAM":
                true_positive_ham += 1
            else:
                false_positive_ham += 1

    val = {"true_positive_spam": true_positive_spam, "false_positive_spam": false_positive_spam, "true_positive_ham": true_positive_ham, "false_positive_ham": false_positive_ham}

    accuracy_test = (test["true_positive_spam"] + test["true_positive_ham"]) / (test["true_positive_spam"] + test["true_positive_ham"] + test["false_positive_spam"] + test["false_positive_ham"])
    accuracy_val = (val["true_positive_spam"] + val["true_positive_ham"]) / (val["true_positive_spam"] + val["true_positive_ham"] + val["false_positive_spam"] + val["false_positive_ham"])

    print("Accuracy test: ", accuracy_test)
    print("Accuracy validation: ", accuracy_val)

    user_input = input("Enter a message: ")
    user_input = clean(user_input)

    p_s = 1
    p_h = 1
    for word in user_input:
        if word not in vocab:
            p_spam_word[word] = 1 / (len(spam_dict) + len(vocab))
            p_ham_word[word]  = 1 / (len(ham_dict)  + len(vocab))
            vocab.append(word)
            #print(word, p_spam_word[word], p_ham_word[word])

        p_s *= p_spam_word[word]
        p_h *= p_ham_word[word]
    p_s *= p_spam
    p_h *= p_ham

    # devolver la probabilidad de que sea spam y ham,así mismo como cuál es la decisión de clasificación de su modelo

    print("Probabilidad de que sea spam: ", p_s)
    print("Probabilidad de que sea ham: ", p_h)

    if p_s > p_h or p_s == p_h:
        print("Es spam")
    else:
        print("Es ham")

    return accuracy_test , accuracy_val





    
                

        
    #print(model_spam_prob, model_ham_prob)
                


        

    
#prueba()
    



