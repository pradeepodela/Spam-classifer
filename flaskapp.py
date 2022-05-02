from email import message
import pickle
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string
from flask import Flask,redirect,url_for,render_template,request

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


app=Flask(__name__)
@app.route('/')
def welcome():
    return render_template('/index.html')
### Result checker submit html page
@app.route('/sudmit',methods=['POST','GET'])
def sudmit():
    total_score=0
    if request.method=='POST':
        message=request.form['message']
        transformed_sms = transform_text(message)
        vector_input = tfidf.transform([transformed_sms]).toarray()
        # 3. predict
        result = model.predict(vector_input)[0]
        print('************************')
        if result == 0:
            print('The message is a ham')
            
        else:
            print('The the message is spam')
            
        print('************************')
        
    return render_template('/index.html',result=result)


if __name__=='__main__':
    app.run(debug=True)