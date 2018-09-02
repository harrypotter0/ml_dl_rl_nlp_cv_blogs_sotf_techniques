from nltk.corpus import wordnet as wn
 
# Just to make it a bit more readable
WN_NOUN = 'n'
WN_VERB = 'v'
WN_ADJECTIVE = 'a'
WN_ADJECTIVE_SATELLITE = 's'
WN_ADVERB = 'r'
 
def convert(word, from_pos, to_pos):    
    """ Transform words given from/to POS tags """
 
    synsets = wn.synsets(word, pos=from_pos)
 
    # Word not found
    if not synsets:
        return []
 
    # Get all lemmas of the word (consider 'a'and 's' equivalent)
    lemmas = [l for s in synsets
                for l in s.lemmas() 
                if s.name().split('.')[1] == from_pos
                    or from_pos in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE)
                        and s.name().split('.')[1] in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE)]
 
    # Get related forms
    derivationally_related_forms = [(l, l.derivationally_related_forms()) for l in lemmas]
 
    # filter only the desired pos (consider 'a' and 's' equivalent)
    related_noun_lemmas = [l for drf in derivationally_related_forms
                             for l in drf[1] 
                             if l.synset().name().split('.')[1] == to_pos
                                or to_pos in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE)
                                    and l.synset().name().split('.')[1] in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE)]
 
    # Extract the words from the lemmas
    words = [l.name for l in related_noun_lemmas]
    len_words = len(words)
 
    # Build the result in the form of a list containing tuples (word, probability)
    result = [(w, float(words.count(w))/len_words) for w in set(words)]
    result.sort(key=lambda w: -w[1])
 
    # return all the possibilities sorted by probability
    return result
 
print(convert("death", WN_NOUN, WN_VERB))
# [('die', 0.75), ('end', 0.2), ('decease', 0.05)]
 
print(convert("story", WN_NOUN, WN_VERB))
# [('report', 0.2222222222222222), ('tell', 0.2222222222222222), ('narrate', 0.2222222222222222),...
 
print(convert("boring", WN_ADJECTIVE, WN_NOUN))
# [('tedium', 0.3333333333333333), ('dullness', 0.16666666666666666),...
 
print(convert("trouble", WN_NOUN, WN_ADJECTIVE))
# [('troublous', 0.6666666666666666), ('problematical', 0.3333333333333333)]
 
print(convert("solve", WN_VERB, WN_ADJECTIVE_SATELLITE))
# [('solvent', 0.5), ('workable', 0.5)]
 
print(convert("think", WN_VERB, WN_ADJECTIVE))
# [('cogitative', 0.6666666666666666), ('recollective', 0.3333333333333333)]  
 
 