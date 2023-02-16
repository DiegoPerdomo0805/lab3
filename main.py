from read import read_file
from model import model_bayes

spam_dict, ham_dict, p_spam, p_ham, vocab, test, val = read_file()

model_bayes(spam_dict, ham_dict, p_spam, p_ham, vocab, test, val)

#def main():
