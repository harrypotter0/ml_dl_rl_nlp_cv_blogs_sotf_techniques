import nltk
from nltk.corpus import wordnet as wn
car_synsets = wn.synsets("car")
print(car_synsets)
for sense in car_synsets:
    lemmas = [l.name() for l in sense.lemmas()]
    print("Lemmas for sense : " + sense.name() + "(" +sense.definition() + ") - " + str(lemmas))

