import streamlit as st
import pickle
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string
ps = PorterStemmer()
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)

st.title('SMS Spam Classifier')

message = st.text_area('Enter your message')
button = st.button('Predict')
if button:
    transformed_sms = transform_text(message)
    vector_input = tfidf.transform([transformed_sms]).toarray()
    # 3. predict
    result = model.predict(vector_input)[0]
    print('************************')
    if result == 0:
        print('The message is a ham')
        st.write('The message is a ham')
    else:
        print('The the message is spam')
        st.write('The message is a Spam')
    print('************************')
    
