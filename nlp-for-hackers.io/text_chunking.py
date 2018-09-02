from nltk.corpus import conll2000
chunked_sentence = conll2000.chunked_sents()[0]
# print(chunked_sentence)

from nltk.chunk import conlltags2tree, tree2conlltags
iob_tagged = tree2conlltags(chunked_sentence)
# print(iob_tagged)

chunk_tree = conlltags2tree(iob_tagged)
# print(chunk_tree)

print(len(conll2000.chunked_sents()))
print(len(conll2000.chunked_words()))

## Prepare train and test datasets
import random 
shuffled_conll_sents = list(conll2000.chunked_sents())
random.shuffle(shuffled_conll_sents)
train_sents = shuffled_conll_sents[:int(len(shuffled_conll_sents)*0.9)]
test_sents = shuffled_conll_sents[int(len(shuffled_conll_sents)*0.9+1):]

## NLTK Trigram tagger as a chunker
from nltk import ChunkParserI, TrigramTagger
class TrigramChunkParser(ChunkParserI):
    def __init__(self, train_sents):
        # Extract only the(POS-TAG, IOB-CHUNK-TAG) pairs
        train_data = [[(pos_tag,chunk_tag)for word,pos_tag,chunk_tag in tree2conlltags(sent)]
        for sent in train_sents]
        self.tagger = TrigramTagger(train_data)

    def parse(self, sentence):
        pos_tags =[pos for word, pos in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        conlltags = [(word, pos_tag, chunk_tag)
                    for((word, pos_tag),(pos_tag, chunk_tag)) in zip(sentence, tagged_pos_tags)]
        return conlltags2tree(conlltags)
    
trigram_chunker = TrigramChunkParser(train_sents)
print(trigram_chunker.evaluate(test_sents))

## Classfier Based Tagger
import pickle 
from collections import Iterable 
from nltk import ChunkParserI, ClassifierBasedTagger
from nltk.stem.snowball import SnowballStemmer

def features(tokens, index, history):
    stemmer = SnowballStemmer('english')
    # Pad the sequence with placeholders
    tokens = [('__START2__', '__START2__'), ('__START1__', '__START1__')] + list(tokens) + [('__END1__', '__END1__'), ('__END2__', '__END2__')]
    history = ['__START2__', '__START1__'] + list(history)
    index+=2
    word,pos = tokens[index]
    prevword, prevpos = tokens[index-1]
    prevprevword,prevprevpos = tokens[index-2]
    nextword, nextpos = tokens[index+1]
    nextnextword, nextnextpos = tokens[index+2]

    return{
        'word':word,
        'lemma':stemmer.stem(word),
        'pos':pos,

        'next-word': nextword,
        'next-pos': nextpos,

        'next-next-word': nextnextword,
        'nextnextpos': nextnextpos,

        'prev-word': prevword,
        'prev-pos': prevpos,

        'prev-prev-word': prevprevword,
        'prev-prev-pos': prevprevpos,
    }

class ClassifierChunkParser(ChunkParserI):
    def __init__(self, chunked_sents, **kwargs):
        assert isinstance(chunked_sents, Iterable)
        chunked_sents = [tree2conlltags(sent)for sent in chunked_sents]

        def triplets2tagged_pairs(iob_sent):
            return [((word, pos), chunk) for word, pos, chunk in iob_sent]
        chunked_sents = [triplets2tagged_pairs(sent) for sent in chunked_sents]

        self.feature_detector = features 

        self.tagger = ClassifierBasedTagger(
            train=chunked_sents,
            feature_detector=features,
            **kwargs
        )
    
    def parse(self, tagged_sent):
        chunks = self.tagger.tag(tagged_sent)
        iob_triplets = [(w,t,c) for ((w,t),c) in chunks]
        return conlltags2tree(iob_triplets)

classifier_chunker = ClassifierChunkParser(train_sents)
print(classifier_chunker.evaluate(test_sents))

from nltk import word_tokenize, pos_tag 
print(classifier_chunker.parse(pos_tag(word_tokenize("das efaf eesrg er gerg sr gsg ef  es fbsdf bsdf bsd fb esb"))))




