import pickle
from main import *
from keras_preprocessing.sequence import pad_sequences
from keras.models import model_from_json
from sklearn.ensemble import RandomForestClassifier

from comparison import accuracy_rf,accuracy_nb,accuracy_lstm,accuracy_cnn
from flask import Flask,render_template,session,request
from DBConnection import Db
import keras.backend
from textblob.classifiers import NaiveBayesClassifier
from sklearn.model_selection import train_test_split
from clean import cleaning
import pandas as pd
import hashlib
import pandas as pd

from datetime import timedelta

app = Flask(__name__)
app.secret_key = "12345"


@app.route("/")
def load():
    return render_template('/chat_app/login.html')


@app.route("/register")
def register():
    return render_template('/chat_app/register.html')


@app.route("/signup", methods=['POST'])
def signup():
    name = request.form['txt_name']
    uname = request.form['txt_uname']
    password = hashlib.sha256(request.form['txt_password'].encode()).hexdigest()

    dbcon = Db()

    qry = "INSERT INTO login(name,username,password,usertype) VALUES('"+name+"','"+uname+"','"+str(password)+"','user')"

    dbcon.insert(qry)

    return render_template('/chat_app/login.html')

x = ""
usrname = ""


@app.route("/login", methods=['GET','POST'])
def login():
    session.permanent = True
    app.permanent_session_lifetime = timedelta(minutes=25)

    uname = request.form['txt_username']
    password = hashlib.sha256(request.form['txt_password'].encode()).hexdigest()

    print(uname)
    print(str(password))
    dbcon = Db()

    global res
    qry = "SELECT * FROM login WHERE username='" + uname + "' AND password='" + str(password) + "'"
    count = "SELECT COUNT(*) FROM login WHERE username='"+uname+"' AND password='"+str(password)+ "'"

    res = dbcon.selectOne(qry)
    cntr = dbcon.selectOne(count)

    global x,usrname,name

    print(cntr)
    if cntr['COUNT(*)'] > 0:
        if res['usertype'] == 'user':
            session['lid'] = res['loginid']

            x = res['name']
            usrname = res['username']
            name = res['name']

            return render_template('/chat_app/compose.html', data=name, name=name,username=usrname)

        else:
            session['lid'] = res['loginid']

            x = res['name']
            usrname = res['username']

    else:
        return "<script> alert('Invalid credentials'); window.location.href='/';</script>"


@app.route("/compose", methods=['GET', 'POST'])
def compose():
    print(x)
    return render_template('/chat_app/compose.html', id=res,username=usrname)


@app.route("/messages", methods=['GET', 'POST'])
def msgs():
    dbcon = Db()

    global usrname

    qry = "SELECT * FROM messages WHERE tousr='"+usrname+"' AND type<>'spam'"

    res = dbcon.select(qry)

    print(res)

    return render_template('/chat_app/inbox.html', id=x, data=res,length=len(res))


@app.route("/spams")
def spams():
    dbcon = Db()

    qry = "SELECT * FROM messages WHERE tousr='" + usrname + "' AND type='spam'"

    res = dbcon.select(qry)

    print(res)

    return render_template('/chat_app/spam.html', id=x, data=res,length=len(res))


@app.route("/sendmsg", methods=['POST'])
def sendmsg():
    fromkey = request.form['from_user']
    uname = request.form['txt_username']
    msg = request.form['txt_password']

    global usrname

    # pre processing
    var = "D:\\Research\\spam_email_detection\\static\\spamham.csv"

    lst = cleaning(msg)

    string = ''
    for i in lst:
        string += i + " "

    dt = pd.read_csv(var)

    accuracies = dict()

    accuracies['RF'] = accuracy_rf()
    accuracies['NB'] = accuracy_nb()
    accuracies['LSTM'] = accuracy_lstm()
    accuracies['CNN'] = accuracy_cnn()

    print(accuracies)
    for key, value in accuracies.items():
        if value == max(accuracies.values()):
            print(key, value)

            if key == 'RF':
                msgs = dt.values[:, 1]
                labels = dt.values[:, 0]

                xtrain, xtest, ytrain, ytest = train_test_split(msgs, labels, test_size=0.2, random_state=0)

                with open("vector.pkl", "rb") as handle:
                    vector = pickle.load(handle)

                xtest_vector = vector.transform(xtest)
                xtrain_vector = vector.transform(xtrain)

                rf = RandomForestClassifier()

                rf.fit(xtrain_vector, ytrain)

                msgvector = vector.transform(msg)

                predicted = rf.predict(msgvector)

                print(predicted[0])

                dbcon = Db()

                qry = "INSERT INTO messages VALUES('" + uname + "','" + msg + "','" + fromkey + "','" + predicted[0] + "')"

                dbcon.insert(qry)

                alert = "MESSAGE SENT"

                name = dbcon.selectOne("SELECT name from LOGIN WHERE username='" + fromkey + "'")

                return render_template('/chat_app/compose.html', name=name, alert=alert, username=usrname)

            elif key == 'NB':
                messages = dt.values[:1000, 1]
                labels = dt.values[:1000, 0]

                train = []

                for i in range(len(messages)):
                    train.append((messages[i], labels[i]))

                a = NaiveBayesClassifier(train)

                # classification
                s = a.classify(string)

                print(s)

                dbcon = Db()

                qry = "INSERT INTO messages VALUES('" + uname + "','" + msg + "','" + fromkey + "','" + s + "')"

                dbcon.insert(qry)

                alert = "MESSAGE SENT"

                name = dbcon.selectOne("SELECT name from LOGIN WHERE username='" + fromkey + "'")

                return render_template('/chat_app/compose.html', name=name, alert=alert, username=usrname)

            elif key == 'LSTM':
                keras.backend.clear_session()

                path1 = "D:\\Research\\spam_email_detection\\lstm_model.h5"
                path2 = "D:\\Research\\spam_email_detection\\lstm_model.json"
                path3 = "D:\\Research\\spam_email_detection\\lstm_tokenizer.pickle"

                with open(path3, "rb") as h:
                    tokenizer = pickle.load(h)

                jhandle = open(path2, 'r')

                jsoncontent = jhandle.read()

                jhandle.close()

                loadedmodel = model_from_json(jsoncontent)

                loadedmodel.load_weights(path1)

                lst = [msg]

                f = tokenizer.texts_to_sequences(lst)

                trainFeatures = pad_sequences(f, maxlen=20, padding='post')

                loadedmodel.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

                p = loadedmodel.predict(trainFeatures)

                if p[0][0] > 0.5:
                    res = 'ham'

                else:
                    res = 'spam'


                print(res)
                dbcon = Db()

                qry = "INSERT INTO messages VALUES('" + uname + "','" + msg + "','" + fromkey + "','" + res + "')"

                dbcon.insert(qry)

                alert = "MESSAGE SENT"

                name = dbcon.selectOne("SELECT name from LOGIN WHERE username='" + fromkey + "'")

                return render_template('/chat_app/compose.html', name=name, alert=alert, username=usrname)

            else:
                keras.backend.clear_session()

                path1 = "D:\\Research\\spam_email_detection\\model.h5"
                path2 = "D:\\Research\\spam_email_detection\\model.json"
                path3 = "D:\\Research\\spam_email_detection\\tokenizer.pickle"

                with open(path3, "rb") as h:
                    tokenizer = pickle.load(h)

                jhandle = open(path2, 'r')

                jsoncontent = jhandle.read()

                jhandle.close()

                loadedmodel = model_from_json(jsoncontent)

                loadedmodel.load_weights(path1)

                lst = [msg]

                f = tokenizer.texts_to_sequences(lst)

                trainFeatures = pad_sequences(f, 100, padding='post')

                loadedmodel.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

                p = loadedmodel.predict(trainFeatures)

                if p[0][0] > 0.5:
                    res = 'ham'

                else:
                    res = 'spam'

                print(res)

                dbcon = Db()

                qry = "INSERT INTO messages VALUES('" + uname + "','" + msg + "','" + fromkey + "','" + res + "')"

                dbcon.insert(qry)

                alert = "MESSAGE SENT"

                name = dbcon.selectOne("SELECT name from LOGIN WHERE username='" + fromkey + "'")

                return render_template('/chat_app/compose.html', name=list(name.values())[0],alert=alert,username=usrname)


if __name__ == "__main__":
    app.run()