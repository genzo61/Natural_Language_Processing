# libraries
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
nltk.download("stopwords")
# veri setini yükleme
df = pd.read_csv("IMDB Dataset.csv")
documents = df["review"]
# metin temizleme
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+","",text) # sayıları kaldırma
    text = re.sub(r"[^\w\s]","",text) # özel karakterler
    text = " ".join([word for word in text.split() if len(word) > 2])
    # text = simple_preprocess(text) # tokenizasyon işlemi !!!
    # stopwordsler
    stopwords_eng = set(stopwords.words("english"))
    text_list = text.split()
    text = [w for w in text_list if w.lower() not in stopwords_eng]
    text = " ".join(text)
    return text

cleaned_document = [clean_text(doc) for doc in documents]
# metin tokenization
tokinezid_document = [simple_preprocess(doc) for doc in cleaned_document]

#%% word2vec (metini sayısallaştırma işlemi)

word2vec_model = Word2Vec(sentences=tokinezid_document, vector_size=50,window=5,min_count=1,sg=0)

word_vectors = word2vec_model.wv

words = list(word_vectors.index_to_key)[:500]
vectors = [word_vectors[word] for word in words]

# clustring Kmeans

kmeans = KMeans(n_clusters=2)
kmeans.fit(vectors)
clusters = kmeans.labels_

# PCA
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(vectors)

# 2d görselleştirme
plt.figure()
plt.scatter(reduced_vectors[:,0],reduced_vectors[:,1], c = clusters, cmap="viridis")

centers = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers[:,0],centers[:,1],c="red",marker="x",s=130,label="center")
plt.legend()

# figure üzerine kelimelerin eklenmesi
for i, word in enumerate(words):
    plt.text(reduced_vectors[i,0],reduced_vectors[i,1], word, fontsize=7)
plt.title("word2vec")    




