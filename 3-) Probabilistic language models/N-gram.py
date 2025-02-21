import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from collections import Counter

# veri seti oluşturma
corpus = [
    "I love apple",
    "I love him",
    "I love NLP",
    "you loves me",
    "he loves apple",
    "they love apple",
    "I love you and you love me"]


# verileri tokınlaştırma işlemi 

tokens = [word_tokenize(sentence.lower()) for sentence in corpus]

# bigram 
bigrams = []
for token_list in tokens:
    bigrams.extend(list(ngrams(token_list, 2)))

bigrams_freq = Counter(bigrams)

# trigram
trigrams = []
for token_list2 in tokens:
    trigrams.extend(list(ngrams(token_list2, 3)))

trigrams_freq = Counter(trigrams)    


# model testing


bigram = ("i","love") # hedef bi gram

prop_you = trigrams_freq[("i","love","you")]/bigrams_freq[bigram]

print("you kelimesinin olma olasılığı ", prop_you)

prop_apple = trigrams_freq[("i","love","apple")]/bigrams_freq[bigram]

print("apple olma olasılığı ", prop_apple)




