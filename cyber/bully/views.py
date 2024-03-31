from django.shortcuts import render,redirect
from . models import user
from django.contrib import messages
from django.contrib.auth.models import User,auth
import numpy as np
import pandas as pd
import re
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from django.conf import settings
from django.contrib import messages
import pickle

# Create your views here.
def index(request):
    return render(request,'index.html')

def about(request):
    return render(request,'about.html')

def contact(request):
    return render(request,'contact.html')


def pred(request):
    return render(request,'pred.html')    

def login(request):
    if request.method=="POST":
       
        try:
            Userdetails=user.objects.get(email=request.POST['email'],password=request.POST['password'])
            print("Username=",Userdetails)
            request.session['email']=Userdetails.email
            return render(request,'prediction.html')
        except user.DoesNotExist as e:
            messages.success(request,'Username/Password Invalid...!')
    return render (request,'login.html')  
    

def register(request):
    if request.method=='POST':
        name=request.POST['name']   
        email=request.POST['email']
        password=request.POST['password']
        rpwd=request.POST['repeatpassword']
        user(name=name,email=email,password=password,rpwd = rpwd).save()
        messages.success(request,'The New User '+request.POST['name']+" is saved Successfully...!")
        return render(request,'register.html')
    
    else:
        return render(request,'register.html') 

def logout(request):
    try:
        del request.session['email'] 
    except: 
        return render(request,'index.html')
    return render(request,'index.html')

def result(request):
  return render(request, "result.html")

def prediction(request):
    df = pd.read_csv('static/public_data_labeled.csv')

    value = ''
    #if request.method == 'POST':


    def preprocess_tweet(tweet):
    # Remove words other than alphabets.
        row = re.sub("[^A-Za-z ]", "", tweet).lower()
    
    # Tokenize words.
        words = word_tokenize(row)

    # Remove stop words.
        english_stops = set(stopwords.words('english'))

    # Remove un-necessary words.
        characters_to_remove = ["''",'``',"rt","https","’","“","”","\u200b","--","n't","'s","...","//t.c" ]
        clean_words = [word for word in words if word not in english_stops and word not in characters_to_remove]

    # Lematise words.
        wordnet_lemmatizer = WordNetLemmatizer()
        lemma_list = [wordnet_lemmatizer.lemmatize(word) for word in clean_words]

        return " ".join(lemma_list)
        
    df['Processed_Tweet'] = df['full_text'].map(preprocess_tweet)

    textmessage = request.POST['textmessage']
    cv = CountVectorizer(max_features=1500)

    X = cv.fit_transform(df['Processed_Tweet']).toarray()

    # Label encode.
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(df['label'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Logistic Regression.
    lr = LogisticRegression(random_state=0)

    # Train classifier.
    lr.fit(X_train, y_train)

    # Predict on train set.
    y_train_pred = lr.predict(X_train)

    # Predict on test set.
    y_test_pred = lr.predict(X_test)

    # save the model to disk

    # f = 'static/cybermodel.sav'
    # pickle.dump(lr, open(f, 'wb'))
    # print("Model saved")

    
    # Creating our training model.
    test1 = [preprocess_tweet(textmessage)]
    test2 = cv.transform(test1)
    model = pickle.load(open("static/cybermodel.sav", "rb"))
    prediction = model.predict(test2)

        

    if int(prediction[0]) == 1:
        value = 'Offensive'
    elif int(prediction[0]) == 0:
        value = "Non-offensive"
    
    return render(request,
                  'result.html',
                  {
                      'context': value,
                      'title': 'Prediction',
                      'active': 'btn btn-success peach-gradient text-white',
                      'background': 'bg-primary text-white'
                  }) 


def pred(request):
    if request.method == "POST":
            return render(request,'chart.html')
    return render(request,'pred.html')




