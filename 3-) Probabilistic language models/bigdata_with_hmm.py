import nltk
from nltk.tag import hmm
from nltk.corpus import conll2000

# veri setini içeriye aktarma
nltk.download("conll2000")

train_data = conll2000.tagged_sents("train.txt")
test_data = conll2000.tagged_sents("test.txt")

# train hmm

trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)


# modeli test etme işlemi

text_sentences = "I like going to school".split()
tags = hmm_tagger.tag(text_sentences)
print("tags : ", tags)


"""
Part of speech çalışması POS.

"""