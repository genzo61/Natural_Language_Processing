import nltk
from nltk.tag import hmm

# örnek training data
train_data = [
    [("I","PRP"),("am","VBP"),("a","DT"),("teacher","NN")],
    [("You","PRP"),("are","VBP"),("a","DT"),("student","NN")]
    ]

# train HMM
trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)



# yeni bir cümle oluştur ve cümlenin içerisinde bulunan her bir kelimenin türünü belirle

test_sentences = "I am a student".split()

tags = hmm_tagger.tag(test_sentences)

print("tags", tags)

#%%
test_sentences2 = "He is a pilot".split()

tags = hmm_tagger.tag(test_sentences2)

print("tags", tags)