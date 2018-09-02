from nltk import word_tokenize, pos_tag, ne_chunk
sentence = "MArk and John are working at Byju."
print(ne_chunk(pos_tag(word_tokenize(sentence))))

from nltk.chunk import conlltags2tree, tree2colltags

ne_tree = ne_chunk(pos_tag(word_tokenize(sentence)))

iob_tagged = tree2colltags(ne_tree)
print(iob_tagged)

ne_tree = conlltags2tree(iob_tagged)

print(ne_tree)

## GMB NER 
import os 
import collections

ner_tags = collections.Counter()

corpus_root = "/home/harrypotter0/Downloads/gmb-2.2.0.zip"
for root,dirs,files in os.walk(corpus_root):
    for filename in files:
        if filename.endswith(".tags"):
            with open(os.path.join(root, filename), 'rb') as file_handle:
                file_content = file_handle.read().decode('utf-8').strip()
                annotated_sentences = file_content.split('\n\n')
                for annotated_sentence in annotated_sentences:
                    annotated_tokens = [seq for seq in annotated_sentence.split('\n') if seq] 
                    standard_form_tokens = []
                    for idx, annotated_token in enumerate(annotated_tokens):
                        annotations = annotated_token.split('\t')
                        word, tag, ner = annotations[0], annotations[1], annotations[3]
                        if(ner)!='0':
                            ner = ner.split('-')[0]
                        ner_tags[ner]+=1

print(ner_tags)
print("Words=",sum(ner_tags.values()))

## Training our NE Chunker

import string 
from nltk.stem.snowball import SnowballStemmer

def features(tokens, index, history):
    stemmer = SnowballStemmer('english')
    # Pad the sequence with placeholders
    tokens = [('[START2]', '[START2]'), ('[START1]', '[START1]')] + list(tokens) + [('[END1]', '[END1]'), ('[END2]', '[END2]')]
    history = ['[START2]', '[START1]'] + list(history)

    index +=2
    word, pos = tokens[index]
    prevword, prevpos = tokens[index - 1]
    prevprevword, prevprevpos = tokens[index - 2]
    nextword, nextpos = tokens[index + 1]
    nextnextword, nextnextpos = tokens[index + 2]
    previob = history[index - 1]
    contains_dash = '-' in word
    contains_dot = '.' in word
    allascii = all([True for c in word if c in string.ascii_lowercase])
    
    allcaps = word==word.capitalize()
    capitalized = word[0] in string.ascii_uppercase

    prevallcaps = prevword ==prevword.capitalize()
    prevcapitalized = prevword[0] in string.ascii_uppercase

    return {
        'word': word,
        'lemma': stemmer.stem(word),
        'pos': pos,
        'all-ascii': allascii,
 
        'next-word': nextword,
        'next-lemma': stemmer.stem(nextword),
        'next-pos': nextpos,
 
        'next-next-word': nextnextword,
        'nextnextpos': nextnextpos,
 
        'prev-word': prevword,
        'prev-lemma': stemmer.stem(prevword),
        'prev-pos': prevpos,
 
        'prev-prev-word': prevprevword,
        'prev-prev-pos': prevprevpos,
 
        'prev-iob': previob,
 
        'contains-dash': contains_dash,
        'contains-dot': contains_dot,
 
        'all-caps': allcaps,
        'capitalized': capitalized,
 
        'prev-all-caps': prevallcaps,
        'prev-capitalized': prevcapitalized,
 
        'next-all-caps': nextallcaps,
        'next-capitalized': nextcapitalized,
    }


