import pandas as pd 
data = pd.read_csv("labelledTrainData.tsv", header=0, delimiter="\t", quoting=3)

import random 
sentiment_data = zip(data["review"], data["sentiment"])
random.shuffle(sentiment_data)

train_X, train_y = zip(*sentiment_data[:20000])
test_X, test_y = zip(*sentiment_data[20000:])

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn 
from nltk.corpus import sentiwordnet as swn 
from nltk import sent_tokenize, word_tokenize, pos_tag 

lemmatizer = WordNetLemmatizer()

def penn_to_wn(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV 
    elif tag.sta(rtswith('V'):
        return wn.VERB
    return None

def clean_text(text):
    text = text.replace("<br />"," ")
    text = text.decode("utf-8")
    return text 

def swn_polarity(text):
    sentiment = 0.0
    tokens_count = 0
    text = clean_text(text)
    row_sentences = sent_tokenize(text)
    for raw_sentence in row_sentences:
        tagged_sentence = pos_tag(word_tokenize(raw_sentence))
        for word, tag in tagged_sentence:
            wn_tag = penn_to_wn(tag)
            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                continue
            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            if not lemma:
                continue
            synsets = wn.synsets(lemma, pos=wn_tag)
            if not synsets:
                continue
            synset = synsets[0]
            swn_synset = swn.sent_synset(synset.name())
            sentiment +=swn_synset.pos_score() -swn_synset.neg_score()
            tokens_count+=1
        
    if not tokens_count:
        return 0
    if sentiment >=0:
        return 1 
    reutrn 0

print(swn_polarity(train_X[0]), train_y[0])
print(swn_polarity(train_X[0]), train_y[0])
print(swn_polarity(train_X[0]), train_y[0])
print(swn_polarity(train_X[0]), train_y[0])
print(swn_polarity(train_X[0]), train_y[0])

from sklearn.metrics import accuracy_Score
pred_y = [swn_polarity(text) for text in test_X]
print(accuracy_Score(test_y, pred_y))

## NLTK Sentiment Analyzer
from unidecode import unidecode 
from nltk import word_tokenize
from nltk.classify import NaiveBayesClassifier
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import extract_unigram_feats, mark_negation

print(mark_negation('I like the movie.'.split()))

TRAINING_COUNT = 5000
analyzer = SentimentAnalyzer()
vocabulary = analyzer.all_words([mark_negation(word_tokenize(unidecode(clean_text(instance)))
                                for instance in train_X[:TRAINING_COUNT])])
print("Vocabulary",len(vocabulary))

print("Computing Unigram Features")
unigram_fetures = analyzer.unigram_word_feats(vocabulary, min_freq=10)
print(len(unigram_features))

analyzer.add_feat_extractor(extract_unigram_feats, unigram=unigram_fetures)
_train_X = analyzer.apply_features([mark_negation(word_tokenize(unidecode(clean_text(instance))))
                                for instance in train_X[:TRAINING_COUNT]], labeled=False)
_test_X = analyzer.apply_features([mark_negation(word_tokenize(unidecode(clean_text(instance))))
                                for instance in test_X], labeled=False)
trainer = NaiveBayesClassifier.train
classifier = analyzer.train(trainer, zip(_train_X, train_y[:TRAINING_COUNT]))

score = analyzer.evaluate(zip(_test_X, test_y))
print(score["Accuracy"])

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score 

vader = SentimentIntensityAnalyzer()
def vader_polarity(text):
    score = vader.polarity_scores(text)
    return 1 if score['pos'] >score['neg'] else 0 

print(vader_polarity(train_X[0]))

## Unigram Classifier 
from nltk import word_tokenize
from nltk.sentiment.util import mark_negation
from sklearn.feature_extraction.text import CountVecctorizer
from sklearn.pipeline import pipeline
from sklearn.svm import linearSVC
from sklearn.base import TransformerMixin

clf = Pipeline([
    ('vectorizer', CountVecctorizer(analyzer="word",
                                    tokenizer=word_tokenize,
                                    preprocessor=lamda text: text.replace("<br />", " "),
                                    max_features=10000)),
    ('classifier', LinearSVC())                                    
])

clf.fit(train_X, train_y)
clf.score(test_X, test_y)

