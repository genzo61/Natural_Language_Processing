import nltk
import numpy as np
nltk.download("punkt")

text = "hello, i .will be: computer engineer in 2026"

# word_tokanize splits the text into words.
word_tokens = nltk.word_tokenize(text)
print("word tokens are : ", word_tokens)

#sentence tokanize : sent_tokanize splits the text into sentences.
sentences_tokens = nltk.sent_tokenize(text)
print("sentences tokens are : ",sentences_tokens)