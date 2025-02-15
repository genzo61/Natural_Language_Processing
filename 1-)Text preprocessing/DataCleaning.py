# Eliminate excess white space in texts (Fazla boşlukları ortadan kaldırma)
text = "Hello,        i will     be     computer      engineer  in   2026"
text.split()
cleaned_text = " ".join(text.split())
print("original text : ", text)
print("cleaned text : ", cleaned_text)
print("***************************************************")
# uppercase to lowercase conversion
text2 = "HELLO, I WILL BE COMPUTER ENGİNEER İN 2026"
cleaned_text2 = text2.lower()
print("original upper text : ", text2)
print("cleaned upper text : ", cleaned_text2)
print("***************************************************")
# remove punctuation
import string
text4 = "hello, i .will be: computer engineer in 2026"
cleaned_text4 = text4.translate(str.maketrans("","",string.punctuation))
print("original text : ", text4)
print("cleaned text : ", cleaned_text4)
print("***************************************************")
# remove special characters
import re
text3 = "hello{{, i will@@ be computer engi̇neer] i̇n 2026@"
cleaned_text3 = re.sub(r"[^a-zA-Z0-9\s]","",text3)
print("original special character text : ", text3)
print("cleaned special character text : ", cleaned_text3)
print("***************************************************")
# correcting typos (yazim hatalarini duzelme)
import numpy as np
from textblob import TextBlob
np.int = int
np.float = float
text5 = "helli  I wönt be "
cleaned_text5 = TextBlob(text5).correct()
print("original text : ", text5)
print("cleaned text : ", cleaned_text5)
print("***************************************************")
# Removing html and url tags
from bs4 import BeautifulSoup
html_text = "<html><h2>hello, world! 2035</h2></html>"
cleaned_text6 = BeautifulSoup(html_text,"html.parser").getText()
print("original html text : ", html_text)
print("cleaned html text : ", cleaned_text6)
