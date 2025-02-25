# importing libraries
import pandas as pd
import numpy as np 


# veri setini içeri aktarma
df = pd.read_csv("spam.csv", encoding="latin1")
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace = True)
df.columns = ["label","text"]
# EDA: keşifsel veri analizi

print(df.isna().sum())

#%% text celaning and preprossing, özel karakterler, lowercase ,tokenization,stopwords,lemmatization
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download("stopwords")
nltk.download("wordnet") # lemma bulmak için 
nltk.download("omw-1.4") # faklı diller için 

text = list(df.text)
lemmatizer = WordNetLemmatizer()

corpus = [] # temizlenmiş veri setini içinde bulundaracak

for i in range(len(text)):
    
    r = re.sub("[^a-zA-Z]"," ",text[i]) # metin içerisinde harf olmayan tüm karakterlerden kurtul
    r = r.lower()
    r = r.split() # kelimeleri ayır
    r = [word for word in r if word not in stopwords.words("english")] # stopwordslerden kurtulma
    r = [lemmatizer.lemmatize(word) for word in r]
    r = " ".join(r)
    corpus.append(r)
df["text2"] = corpus      


#%% model training and evaluation

X = df["text2"]
y = df["label"]

from sklearn.model_selection import train_test_split

x_train,x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# modele text verilerini vermemiz için sayısallaştırmamız gerek
# feature extraction : BOW bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train_cv = cv.fit_transform(x_train)

# classify training : model training and evaluation
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train_cv, y_train) # eğitim

x_test_cv = cv.transform(x_test)

# prediction
prediction = dt.predict(x_test_cv)

# doğruluk karşılaştırılması için confusion matrix kullanımı

from sklearn.metrics import confusion_matrix
c_matrix = confusion_matrix(y_test, prediction)


accuracy = 100*(sum(sum(c_matrix)) - c_matrix[1,0] - c_matrix[0,1]) / sum(sum(c_matrix))

print("model başarısı : ", accuracy)
