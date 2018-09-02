from nltk import word_tokenize, pos_tag
print(pos_tag(word_tokenize('I am masturbating with NLP')))

import nltk
tagged_sentences = nltk.corpus.treebank.tagged_sents()
print(tagged_sentences[0])
print(len(tagged_sentences))
print(len(nltk.corpus.treebank.tagged_words()))

## Training our own pos tagger using sklearn

def features(sentence, index):
    return {
        'word': sentence[index],
        'is_first': index==0,
        'is_last': index == len(sentence)-1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index-1],
        'next_word': '' if index == len(sentence)-1 else sentence[index+1],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]        
    }

import pprint 
pprint.pprint(features(['This', 'is', 'a', 'sentence'], 2))

def untag(tagged_sentence):
    return [w for w,t in tagged_sentence]

# Split the dataset for training and testing
cutoff = int(.75 * len(tagged_sentences))
training_sentences = tagged_sentences[:cutoff]
test_sentences = tagged_sentences[cutoff:]

print(len(training_sentences))
print(len(test_sentences))

def transform_to_dataest(tagged_sentences):
    X,y = [],[]

    for tagged in tagged_sentences:
        for index in range(len(tagged)):
            X.append(features(untag(tagged), index))
            y.append(tagged[index][1])
    return X,y

X,y = transform_to_dataest(training_sentences)

## Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ('vectorizer', DictVectorizer(sparse=False)),
    ('classifier', DecisionTreeClassifier(criterion='entropy'))
])

clf.fit(X[:1000], y[:1000])
print("training completed")
X_test, y_test = transform_to_dataest(test_sentences)

print(clf.score(X_test, y_test))

def pos_tag(sentence):
    tagged_sentence = []
    tags = clf.predict([features(sentence, index ) for index in range(len(sentence))])
    return zip(sentence, tags)

print(pos_tag(word_tokenize('This is my yard!!')))
