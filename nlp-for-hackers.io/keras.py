import re 
import pandas as pd 
from sklearn.model_selection import train_test_split
def clean_review(text):
    text = re.sub('<[^<]+?',' ',text)
    text = text.replace('\\"', '')
    text = text.replace('"','')
    return text 

df = pd.read_csv('labelled.csv', sep='\t', quoting=3)
df['cleaned_review']=df['review'].apply(clean_review)
X_train,X_test,y_train,y_test = train_test_split(df['cleaned_review'], df['sentiment'], test_size=0.2)

from sklearn.features_extraction.text import CountVectorizer
from nltk.corpus import stopwords 
vectorizer = CountVectorizer(binary=True, stop_words=stopwords.words('english'),
                            lowercase=True, min_df=3, max_df=0.9, max_features=5000) 
X_train_onehot = vectorizer.fit_transform(X_train)

from keras.models import Sequential 
from keras.layers import Dense 

model = Sequential()
model.add(Dense(units=500, activation='relu', input_dim=len(vectorizer.get_feature_names())))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X_train_onehot[:-100], y_train[:-100],
        epochs=2, batch_size=128, verbose=1,
        validation_data=(X_train_onehot[-100:], y_train[-100:]))

scores = model.evaluate(vectorizer.transfer(X_test), y_test, verbose=1)
print(scores[1])

## Convolutional Network
word2idx = {word: idx for idx, word in enumerate(vectorizer.get_feature_names())}
tokenize = vectorizer.build_tokenizer()
preprocess = vectorizer.build_preprocessor()

def to_sequence(tokenizer, preprocessor, index, text):
    words = tokenizer(preprocessor(text))
    indexes = [index[word] for word in words if word in index]
    return indexes 

print(to_sequence(tokenize, preprocess, word2idx, "This is an important test!"))
X_train_sequences = [to_sequence(tokenize, preprocess, word2idx, x)for x in X_train]
print(X_train_sequences[0])

MAX_SEQ_LENGTH = len(max(X_train(X_train_sequences, key=len)))
print(MAX_SEQ_LENGTH)
from keras.preprocessing.sequence import pad_sequences
N_FEATURES = len(vectorizer.get_feature_names())
X_train_sequences = pad_sequences(X_train_sequences, maxlen=MAX_SEQ_LENGTH, value=N_FEATURES)
print(X_train_sequences[0])

