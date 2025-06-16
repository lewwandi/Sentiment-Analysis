from flask import Flask, render_template,request, redirect
import numpy as np
import string
import re
import pandas as pd
import pickle
from nltk.stem import PorterStemmer

app = Flask(__name__)

ps=PorterStemmer()

#load the model
with open('static/model/sentiment_model.pkl','rb') as file:
    model=pickle.load(file)

#load stopwords
with open('static/model/corpora/stopwords/english','r') as file:
    stopwords=file.read().splitlines()

#load tokens
vocab=pd.read_csv('static/model/vocabulary.txt',header=None)
tokens=vocab[0].tolist()

data = dict()
reviews = []
positive = 0
negative = 0

@app.route("/")
def index():
    data['reviews'] = reviews
    data['positive'] = positive
    data['negative'] = negative
    return render_template('index.html', data=data)

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text=text.replace(punctuation,'')
    return text

#data preprocessing
def data_preprocessing(text):
 data=pd.DataFrame([text],columns=['tweet'])
 data["tweet"]=data["tweet"].apply(lambda x:" ".join(x.lower() for x in x.split()))
 data["tweet"]=data["tweet"].apply(lambda x: " ".join(re.sub(r'^https?:\/\/.*[\r\n]*', '', x,flags=re.MULTILINE) for x in x.split()))
 data["tweet"]=data["tweet"].apply(remove_punctuations)
 data["tweet"]=data["tweet"].str.replace('\d+', '', regex=True)  
 data["tweet"]=data["tweet"].apply(lambda x: " ".join(x for x in x.split() if x not in stopwords))
 data["tweet"]=data["tweet"].apply(lambda x: " ".join(ps.stem(x) for x in x.split()))
 return data["tweet"]


def vectorizer(ds):
    vectorized_list=[]

    for sentence in ds:
        sentence_lst=np.zeros(len(tokens))

        for i in range(len(tokens)):
            if tokens[i] in sentence.split():
                sentence_lst[i]=1

        vectorized_list.append(sentence_lst)

    vectorized_list_new=np.asarray(vectorized_list,dtype=np.float32)
    return vectorized_list_new


def get_prediction(vectorized_text):
   prediction=model.predict(vectorized_text)
   if prediction==1:
      return 'Negative'
   else:
      return 'Positive'





@app.route("/", methods = ['post'])
def my_post():
    text = request.form['text']
    preprocessed_txt = data_preprocessing(text)
    vectorized_txt = vectorizer(preprocessed_txt)
    prediction = get_prediction(vectorized_txt)
  

    if prediction == 'negative':
        global negative
        negative += 1
    else:
        global positive
        positive += 1
    
    reviews.insert(0, text)
    return redirect(request.url)

if __name__ == "__main__":
    app.run()