# from sklearn.datasets import fetch_20newsgroups
# news = fetch_20newsgroups(subset='all')

# print(len(news.data))
# print(len(news.target_names))
# print(news.target_names)
# for text, num_label in zip(news.data[:10], news.target[:10]):
#     print('[%s]:\t\t %s' %(news.target_names[num_label], text[:100].split('\n')[0]))

from sklearn.cross_validation import train_test_split

def train(classifier, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
    classifier.fit(X_train, y_train)
    print("Accuracy : %s"%classifier.score(X_test, y_test))
    return classifier

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

trial1 = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', MultinomialNB()),
])
train(trial1, news.data, news.target)

from nltk.corpus import stopwords

trial2 = Pipeline([
    ('vectorizer',TfidfVectorizer(stop_words=stopwords.words('engish'))),
    ('classifier',MultinomialNB())
])
train(trail2, news.data, news.target)

trial3 = Pipeline([
    ('vectorizer',TfidfVectorizer(stop_words=stopwords.words('engish'))),
    ('classifier',MultinomialNB(alpha=0.05))
])

train(trial3, news.data, news.target)


trial3 = Pipeline([
    ('vectorizer',TfidfVectorizer(stop_words=stopwords.words('engish'),
    min_df=5)),
    ('classifier',MultinomialNB(alpha=0.05))
])

train(trial3, news.data, news.target)

import string
from nlltk.stem import PorterStemmer
from nltk import word_tokenize
def stemming_tokenizer(text):
    stemmer = PorterStemmer()
    return [stemmer.stem(w) for w in word_tokenize(text)]

trial5 = Pipeline([
    ('vectorizer', TfidfVectorizer(tokenizer=stemming_tokenizer,
                             stop_words=stopwords.words('english') + list(string.punctuation))),
    ('classifier', MultinomialNB(alpha=0.05)),
])
 
train(trial5, news.data, news.target)
