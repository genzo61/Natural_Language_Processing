# stemming  (Kök Bulma İşlemi)
import nltk
from nltk.stem import PorterStemmer # stemming needs
nltk.download("wordnet")
stemmer = PorterStemmer()
words = ["running", "affordable", "better", "went", "fired", "marker", "bordership"]
# stems (kökler)
stems = [stemmer.stem(w) for w in words] # list comp. yapısı var
print("stems (kelimelerin kökleri) : ", stems)
print("*******************************************************")
# Lemmatization (Çekim eklerini çıkarma işlemidir) yapım eki ve kök kalır.
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
words2 = ["running", "affordable", "better", "went", "fired", "marker", "bordership"]
lemmas = [lemmatizer.lemmatize(x,pos='v') for x in words2]
print("lemmas : ", lemmas)