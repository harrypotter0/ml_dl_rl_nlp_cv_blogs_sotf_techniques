import re
import pandas as pd 
def clean_review(text):
    text = re.sub('<[^<]+?',' ',text)
    text = text.replace("\\",'')
    text = text.replace('"', '')
    return text

df = pd.read_csv('labelledTrainData.csv', sep='\t', quoting=3)
df['cleaned_review'] = df['review'].apply(clean_review)

print(df['review'][0])
print('\n\n')
print(df['cleaned_review'][0])

VOCABULARY = ['dog','cheese','cat','mouse']
TEXT = 'the mouse ate the cheese'

def to_bow(text):
    words = text.split(" ")
    return [1 if w in words else 0 for w in VOCABULARY]

print(to_bow(TEXT))

import spacy 
import numpy as np 
lnp = spacy.load('en_core_web_md')

EMBEDDINGS_LEN = len(nlp.vocab['apple'].vector)
print('EMBEDDINGS_LEN=',EMBEDDINGS_LEN)

embeddings_index = np.zeros((len(vectorizer.get_feature_names())+1,EMBEDDINGS_LEN))
for word, idx in word2idx.items():
    try:
        embedding = nlp.vocab[word].vector
        embeddings_index[idx] = embedding
    except:
        pass


from keras.models import Sequential 
from keras.layers import Dense, LSTM, Embedding 

model = Sequential()
model.add(Embedding(len(vectorizer.get_feature_names())+1,
            ))
