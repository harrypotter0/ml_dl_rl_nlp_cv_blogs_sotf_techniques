from nltk.corpus import wordnet as wn 
car_synsets = wn.synsets('car')
print(car_synsets)

for car in car_synsets:
    print(car.lemmas())
    printimport nltk
from nltk.corpus import wordnet as wn
car_synsets = wn.synsets("car")
print(cake_synsets)
for sense in car_synsets:
    lemmas = [l.name() for l in sense.lemmas()]
    print("Lemmas for sense : " + sense.name() + "(" +sense.definition() + ") - " + str(lemmas))(car.definition())
    print(car.hypernyms())
    print(car.hyponyms())

fight_all = wn.synsets('fight')
print(fight_all)

fight_verb = wn.synsets('fight','v')
print(fight_verb)

fight_noun = wn.synsets('fight', 'n')
print(fight_noun)

print(fight_noun[0].pos())

walk = wn.synset('walk.v.01')
run = wn.synset('run.v.01')
stand = wn.synset('stand.v.01')

print(run.path_similarity(walk))

from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()
print(wnl.lemmatize('running'))
print(wnl.lemmatize('oxen', wn.NOUN))
print(wnl.lemmatize('geese', wn.NOUN))

