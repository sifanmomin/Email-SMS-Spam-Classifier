import streamlit as st

import pickle
import string

import nltk
import string # Import the string module

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download necessary NLTK resources
nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('punkt_tab')

ps = PorterStemmer()


def transfrom_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)

  y = []
  for i in text:
    if i.isalnum():
      y.append(i)

  text = y[:]
  y.clear()
  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:  # Now string.punctuation is accessible
      y.append(i)

  text = y[:]
  y.clear()
  for i in text:
    y.append(ps.stem(i))

  return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button("Predict"):


  # 1.preprocess
  transformed_sms =transfrom_text(input_sms)

  # 2 vectorize
  vector_input = tfidf.transform([transformed_sms])
  # 3 predict

  result = model.predict(vector_input)[0]
  # 4 display
  if result == 1:
      st.header("Spam")
  else:
      st.header("Not Spam")
