# %% importing libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
#  veri setini içeriye aktarma
df = pd.read_csv("IMDB Dataset.csv")
# metin verilerini alalım
documents = df["review"]
labels = df["sentiment"]
#  metin ön işleme adımları
def clean_text(text):
    # büyük harften küçük harf dönüşümü
    text = text.lower()
    # stopwords lerden kurtulma işlemi
    nltk.download("stopwords")
    stop_words_en = set(stopwords.words("english"))
    text_list = text.split()
    text = [w for w in text_list if w.lower() not in stop_words_en]
    text = " ".join(text)  # tekrardan stringe çevirme !!!
    # rakamları temizleme
    text = re.sub(r"\d+", "", text)
    # özel karakterlerin temizlenmesi
    text = re.sub(r"[^\w\s]", "", text)
    # kısa kelimelerin temizlenmesi
    text = " ".join([word for word in text.split() if len(word) > 2])
    return text
# metinleri temizle
cleaned_doc = [clean_text(row) for row in documents]
# %% bow
#  vektörizer tanımla 
vektorizer = CountVectorizer()
#  metini sayısal hale getir
X = vektorizer.fit_transform(cleaned_doc[:75])
#  kelime kümesi göster
feature_names = vektorizer.get_feature_names_out()
#  vektör temsili göster  
vektor_temsili2 = X.toarray()
print("vektor temsili : ", vektor_temsili2)
# kelime frekanslarını gösterme
word_counts = X.sum(axis=0).A1
word_freq = dict(zip(feature_names,word_counts))

# most common 5 words

most_comon_words = Counter(word_freq).most_common(5)
print("most common words : ",most_comon_words)





