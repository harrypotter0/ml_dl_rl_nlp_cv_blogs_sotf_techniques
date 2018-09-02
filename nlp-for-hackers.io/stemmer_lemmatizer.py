from nltk.stem import SnowballStemmer

snow = SnowballStemmer('english')
print(snow.stem('getting'))
print(snow.stem('rabbits'))

from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
print(wnl.lemmatize('getting','v'))

