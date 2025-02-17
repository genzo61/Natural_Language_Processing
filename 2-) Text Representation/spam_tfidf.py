# impor libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import re
# veri setini yükle
df = pd.read_csv("spam.csv", encoding="ISO-8859-1")

# Veri setini temizleme HW
documents = df["v2"]
def clean_text(text):
    # küçük harfe çevirme
    text = text.lower()
    # stopwordslerden kurtulma
    nltk.download("stopwords")
    stop_words_en = set(stopwords.words("english"))
    text_list = text.split()
    text = [w for w in text_list if w.lower() not in stop_words_en]
    text = " ".join(text) # tekrardan string'e çevirme
    # rakamları temizleme
    text = re.sub(r"\d+","",text)
    # özel karakterlerin temizlenmesi
    text = re.sub(r"[^\w\s]","",text)
    # kısa kelimelerin temizlenmesi
    text = " ".join([word for word in text.split() if len(word) > 2])
    return text 
# fonksiyonun uygulanması...
cleaned_document = [clean_text(r) for r in documents]    

# tfidf
tfidf_vektorizer = TfidfVectorizer()
X = tfidf_vektorizer.fit_transform(cleaned_document)

# kelime kümesini inceleme
feature_names = tfidf_vektorizer.get_feature_names_out()
tfidf_score = X.mean(axis=0).A1

# tf-idf scor larını içeren df oluşturma
df_tfidf = pd.DataFrame({"word": feature_names, "tfidf_score": tfidf_score})

# scorları sırala ve incele
df_tfidf_sorted = df_tfidf.sort_values(by="tfidf_score", ascending=False)
print(df_tfidf_sorted.head(10))