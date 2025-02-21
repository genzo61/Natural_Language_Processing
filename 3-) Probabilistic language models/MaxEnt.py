"""
Sentiment Analysis and Classification  (olumlu ve olumsuz sınıflandırma işlemi)
"""

from nltk.classify import MaxentClassifier

# eğitim veri seti tanımlama
train_data = [
    ({"love":True, "amazing":True, "happy":True, "terrible":False}, "positive"),
    ({"hate":True, "terrible":True}, "negative"),
    ({"joy":True, "happy":True, "hate": False}, "positive"),
    ({"sad":True, "depressed":True, "love":False}, "negative")
    ]

# train MAXENT
classifier = MaxentClassifier.train(train_data, max_iter = 10)

# test etme işlemi

test_sentences = "I do not like this movie"
features = {word : (word in test_sentences.lower().split()) for word in ["love","amazing","happy","terrible","hate","joy","depressed","sad"]}
label = classifier.classify(features)
print("labels :", label)

