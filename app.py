
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
import json
import random
from flask import Flask, render_template, request,jsonify
import re
import string

app = Flask(__name__,static_url_path='/assets',
            static_folder='./assets', 
            template_folder='./templates')


h5_ar=load_model("arabic_chatbot_model.h5")
h5_fr=load_model("chatbot_francais_model.h5")
h5_ang=load_model("chatbot_englais_model.h5")

def supprime_accent_mot(L):
        out = ""
        for mot in L:
            for c in mot:
                if c == 'é' or c == 'è' or c == 'ê' :
                    c = 'e'
                if c == 'à':
                    c = 'a'
                if c == 'ù' or c == 'û':
                    c = 'u'
                if c == 'î' or c=='ï':
                    c = 'i'
                if c == 'ç':
                    c = 'c'
                if c=='ô':
                  c='o'
                  
                out +=c.lower()   
        return out
def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


    # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))
#chatbot_francais-------------------------------------------------------------------------------
intents_fr = json.loads(open('data_francais.json',encoding='utf-8').read())
words_fr = pickle.load(open('words_fr.pkl','rb'))
classes_fr = pickle.load(open('classes_fr.pkl','rb'))

def predict_class_fr(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words_fr,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes_fr[r[0]], "probability": str(r[1])})
    return return_list

def getResponse_fr(ints, intents_json):
  if(ints==[]):
        result ="Reformulez votre question autrement"
        
  else:
     tag = ints[0]['intent']
     list_of_intents = intents_json['intents']
     for i in list_of_intents:
         if(i['tag']== tag):
             result = random.choice(i['responses'])
             break   
  return result

def chatbot_response_fr(msg):
    msg=supprime_accent_mot(msg)
    ints = predict_class_fr(msg,h5_fr)
    res = getResponse_fr(ints, intents_fr)
    if msg=="":
        res="tu n'as rien écrit"
    return res

#chatbot_anglais---------------------------------------------------------------------------
intents_eng = json.loads(open('data_anglais.json',encoding='utf-8').read())
words_eng = pickle.load(open('words_eng.pkl','rb'))
classes_eng = pickle.load(open('classes_eng.pkl','rb'))

def predict_class_eng(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words_eng,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes_eng[r[0]], "probability": str(r[1])})
    return return_list


def getResponse_eng(ints, intents_json):
    if(ints==[]):
          result ="Rephrase your question differently"
    else:
       tag = ints[0]['intent']
       list_of_intents = intents_json['intents']
       for i in list_of_intents:
           if(i['tag']== tag):
               result = random.choice(i['responses'])
               break
    return result

def chatbot_response_eng(msg):
    msg=supprime_accent_mot(msg)
    ints = predict_class_eng(msg,h5_ang)
    res = getResponse_eng(ints, intents_eng)
    if msg=="":
        res="You didn't write anything"
    return res

#arabic_chatbot------------------------------------------------------------------
arabic_diacritics = re.compile("""
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                        """, re.VERBOSE)

def remove_diacritics(text):
    text = re.sub(arabic_diacritics, '', text)
    return text

intents_ar = json.loads(open('data_arab.json',encoding='utf-8').read())
words_ar = pickle.load(open('words_ar.pkl','rb'))
classes_ar = pickle.load(open('classes_ar.pkl','rb'))

def predict_class_ar(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words_ar,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes_ar[r[0]], "probability": str(r[1])})
    return return_list


def getResponse_ar(ints, intents_json):
  if(ints==[]):
        result ="أعد صياغة سؤالك بشكل آخر"
        
  else:
     tag = ints[0]['intent']
     list_of_intents = intents_json['intents']
     for i in list_of_intents:
         if(i['tag']== tag):
             result = random.choice(i['responses'])
             break   
  return result

def chatbot_response_ar(msg):
    msg=remove_diacritics(msg)
    ints = predict_class_ar(msg,h5_ar)
    res = getResponse_ar(ints, intents_ar)
    if msg=="":
      res="إنك لم تكتب أي شيء"
    return res




#------------------------------------------------------------------


@app.route("/")
def home():
    return render_template("index.html") 

@app.route("/index.html")
def home_():
    return render_template("index.html")


@app.route("/agadir.html")
def home1():
    return render_template("agadir.html")

@app.route("/casa.html")
def home2():
    return render_template("casa.html")


@app.route("/chefchaouen.html")
def home3():
    return render_template("chefchaouen.html")

@app.route("/fes.html")
def home4():
    return render_template("fes.html")

@app.route("/hociema.html")
def home5():
    return render_template("hociema.html")

@app.route("/hotel -chefchaoun.html")
def home6():
    return render_template("hotel -chefchaoun.html")

@app.route("/hotel -ifrane.html")
def home7():
    return render_template("hotel -ifrane.html")

@app.route("/hotel -meknes.html")
def home8():
    return render_template("hotel -meknes.html")

@app.route("/hotel -tetouan.html")
def home9():
    return render_template("hotel -tetouan.html")

@app.route("/hotel-agadir.html")
def home10():
    return render_template("hotel-agadir.html")


@app.route("/hotel-casablanca.html")
def home11():
    return render_template("hotel-casablanca.html")


@app.route("/hotel-fes.html")
def home12():
    return render_template("hotel-fes.html")

@app.route("/hotel-hociema.html")
def home13():
    return render_template("hotel-hociema.html")

@app.route("/hotel-marrakesh.html")
def home14():
    return render_template("hotel-marrakesh.html")

@app.route("/hotel-rabat.html")
def home15():
    return render_template("hotel-rabat.html")

@app.route("/hotel-tangier.html")
def home16():
    return render_template("hotel-tangier.html")

@app.route("/ifran.html")
def home17():
    return render_template("ifran.html")

@app.route("/info_deguster.html")
def home18():
    return render_template("info_deguster.html")

@app.route("/info_deplace.html")
def home19():
    return render_template("info_deplace.html")

@app.route("/info_hebergement.html")
def home20():
    return render_template("info_hebergement.html")

@app.route("/meknes.html")
def home21():
    return render_template("meknes.html")

@app.route("/Kech.html")
def home22():
    return render_template("Kech.html")

@app.route("/rabat.html")
def home23():
    return render_template("rabat.html")


@app.route("/tanger.html")
def home24():
    return render_template("tanger.html")

@app.route("/tetouan.html")
def home25():
    return render_template("tetouan.html")


@app.route("/chatbot_app_francais.html",methods=['POST','GET'])
def chatbot_francais():
    if request.method == 'POST':
       Q_fr = request.form['Q_fr'] 
       if Q_fr:
           rep_fr=chatbot_response_fr(Q_fr)
           return jsonify({'response_fr' : rep_fr,'question_fr':Q_fr})
    return render_template("chatbot_app_francais.html")

@app.route("/chatbot_app_anglais.html",methods=['POST','GET'])
def chatbot_anglais():
    if request.method == 'POST':
       Q_ang = request.form['Q_ang'] 
       if Q_ang:
           rep_ang=chatbot_response_eng(Q_ang)
           return jsonify({'response_ang' : rep_ang,'question_ang':Q_ang})
    return render_template("chatbot_app_anglais.html")



@app.route("/chatbot_app_arab.html",methods=['POST','GET'])
def chatbot_arab():
    if request.method == 'POST':
       Q_ar = request.form['Q_ar'] 
       if Q_ar:
           rep_ar=chatbot_response_ar(Q_ar)
           return jsonify({'response_ar' : rep_ar,'question_ar':Q_ar})
    return render_template("chatbot_app_arab.html")


""" @app.route("/get")
def get_bot_reponse_eng():
    rep_eng=chatbot_response_eng(Q_ang)
    return str(chatbot_response_eng(Q))


@app.route("/get")
def get_bot_reponse_ar():
    print(Q)
    print(str(chatbot_response_ar(Q)))
    return str(chatbot_response_ar(Q))



@app.route("/get")
def get_bot_reponse_fr():
    print(Q)
    print(str(chatbot_response_fr(Q)))
    return str(chatbot_response_fr(Q)) """

if __name__ == '__main__':
    app.run()