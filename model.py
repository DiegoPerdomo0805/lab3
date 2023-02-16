from read import clean
import math

def model_bayes(spam_dict, ham_dict, p_spam, p_ham, vocab, test, val):

    model_spam_prob = 1
    model_ham_prob = 1
    
    # bayes y laplace smoothing

    # probabilidad de spam dado una palabra
    p_spam_word = {}
    for word in vocab:
        p_spam_word[word] = 0
        if word in spam_dict:
            p_spam_word[word] = (spam_dict[word] + 1) / (len(spam_dict) + len(vocab))
        else:
            p_spam_word[word] = 1 / (len(spam_dict) + len(vocab))

    # probabilidad de ham dado una palabra
    p_ham_word = {}
    for word in vocab:
        p_ham_word[word] = 0
        if word in ham_dict:
            p_ham_word[word] = (ham_dict[word] + 1) / (len(ham_dict) + len(vocab))
        else:
            p_ham_word[word] = 1 / (len(ham_dict) + len(vocab))

    # probabilidad de spam o ham dado un mensaje
    for line in test:
        temp = clean(line)
        answer = temp[0]
        temp = temp[1:]
        p_s = math.log(p_spam)
        p_h = math.log(p_ham)
        for word in temp:
            if word not in vocab:
                p_spam_word[word] = math.log(1 / (len(spam_dict) + len(vocab)))
                p_ham_word[word]  = math.log(1 / (len(ham_dict)  + len(vocab)))

            p_s += math.log(p_spam_word[word])
            p_h += math.log(p_ham_word[word])
        p_s += math.log(p_spam)
        p_h += math.log(p_ham)

        if p_s > p_h:
            if answer == 'SPAM':
                if p_s < model_spam_prob:
                    model_spam_prob = p_s
        else:
            if answer == 'HAM':
                if p_h < model_ham_prob:
                    model_ham_prob = p_h
        
    print(model_spam_prob, model_ham_prob)
                


            


        

    

    



