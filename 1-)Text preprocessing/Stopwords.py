import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
stop_words_en = set(stopwords.words("english"))
text = "there are some examples of stopwords in english language."
text_list = text.split()
filtered_words = [w for w in text_list if w.lower() not in stop_words_en]
print("original english text : ", text)
print("filtered english text : ", filtered_words)