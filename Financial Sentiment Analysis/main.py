#%% import libraries
import pandas as pd
import numpy as np
import nltk 
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

#%% veri setini içeri aktarma ve EDA
df = pd.read_csv("data.csv", encoding="latin-1")
print(df.head())
print(df.isna().sum())
#%% text cleaning and processing, özel karakterler lowercase,stopwords, lemmatization

nltk.download("stopwords")
nltk.download("wordnet") # for lemmatization
nltk.download("omw-1.4") # for differant languages

df.columns = ["text", "label"]
text = list(df["text"])
lemmatizer = WordNetLemmatizer()

corpus = [] # temizlenmiş veri setini içine alacak

for i in range(len(text)):
    r = re.sub("[^a-zA-Z]", " ", text[i]) # özel karakterleri temizle
    r = r.lower() # küçük harfe çevir
    r = r.split() # kelimeleri ayır
    r = [word for word in r if word not in stopwords.words("english")]
    r = [lemmatizer.lemmatize(word) for word in r]
    r = " ".join(r)
    corpus.append(r)
df["text2"] = corpus
print(df.columns)

#%% model training and evaluation
X = df["text2"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# modele text verilerini vermemiz için sayısallaştırmamız gerek
# feature extraction : BOW bag of words
cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)

# classify training : model training and evaluation
model = LogisticRegression()
model.fit(X_train_cv, y_train) # eğitim

X_test_cv = cv.transform(X_test)

# prediction
prediction = model.predict(X_test_cv)

# doğruluk karşılaştırılması için confusion matrix kullanımı

c_matrix = confusion_matrix(y_test, prediction)
print(c_matrix)

accuracy = 100*(sum(sum(c_matrix)) - c_matrix[1,0] - c_matrix[0,1]) / sum(sum(c_matrix))

print("model başarisi : ", accuracy)


#%% testing by user 

text = "Cryptocurrency market sees unprecedented growth"

text_cv = cv.transform([text])
prediction = model.predict(text_cv)
print(prediction)

#### if we wanto these operation by dinamically, we can create a function
