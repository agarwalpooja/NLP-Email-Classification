import argparse
import sys
import os
import nltk
import email
import random
import nltk.classify.util
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.classify import NaiveBayesClassifier
from email.parser import Parser


rootdir = "enron1"

def create_word_features(words):
    my_dict = dict([(word, True) for word in words])
    return my_dict

#List for containing files
business_list=[]
spam_list=[]
private_list=[]

for directories, subdirs, files in os.walk(rootdir):
    
    if(os.path.split(directories)[1] == 'business'):
        for filename in files:
            f = open(os.path.join(directories, filename), encoding="ascii", errors="surrogateescape")
            data = f.read()

            words = word_tokenize(data)
            business_list.append((create_word_features(words),"business"))      
     
    if(os.path.split(directories)[1] == 'spam'):
        for filename in files:
            f = open(os.path.join(directories, filename), encoding="ascii", errors="surrogateescape")
            data = f.read()
            words = word_tokenize(data)
            spam_list.append((create_word_features(words), "spam"))
    
    if(os.path.split(directories)[1] == 'private'):
        for filename in files:
            f = open(os.path.join(directories, filename), encoding="ascii", errors="surrogateescape")
            data = f.read()
            words = word_tokenize(data)
            private_list.append((create_word_features(words), "private"))
    
print("Number of private dataset : ", len(private_list))

combined_list = business_list + spam_list

random.shuffle(combined_list)

divide = int(len(combined_list)* .7)

training_set = combined_list[:divide] + private_list
test_set = combined_list[divide:]

print("Number of training dataset : " ,len(training_set))
print("Number of testing dataset : ", len(test_set))

classifier = NaiveBayesClassifier.train(training_set)

accuracy = nltk.classify.util.accuracy(classifier, test_set)

print("Accuracy is: ", accuracy * 100)

classifier.show_most_informative_features(10)

parser = argparse.ArgumentParser(description='Parse input string')
parser.add_argument('string', help='Input String')

args = parser.parse_args()
arg_str = args.string

with open("enron1/"+arg_str, encoding="ascii") as f:
    data = f.read()

words = word_tokenize(data)
features = create_word_features(words)
print("This email is :", classifier.classify(features))
