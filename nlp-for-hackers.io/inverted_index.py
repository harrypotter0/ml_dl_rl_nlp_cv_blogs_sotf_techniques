from collections import defaultdict

def create_index(data):
    index = defaultdict(list)
    for i,tokens in enumerate(data):
        print(tokens)
        for token in tokens:
            index[token].append(i)
    return index

arr = [['a','b'],['a','c']]
print(create_index(arr))
