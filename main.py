import os
import keras.backend

from sa import HybridMlp

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from flask import Flask,render_template,request,session
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob.classifiers import NaiveBayesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix, precision_score, f1_score
from clean import cleaning
from mlxtend.plotting import plot_confusion_matrix

import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

import keras
from keras import *
from keras import layers, optimizers
from keras.layers import Embedding, Conv1D, LSTM, GlobalMaxPooling1D, Dense, Dropout, SpatialDropout1D
from keras.models import Model, model_from_json
from keras.preprocessing import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
app.secret_key = "12345"


@app.route("/")
def load():
    return render_template('adminindex.html')


@app.route("/view")
def view():
    dt = pd.read_csv("D:\\Research\\spam_email_detection\\static\\spamham.csv")

    sns.countplot(dt.Category)
    plt.xlabel('Label')
    plt.title('Number of ham and spam messages')
    plt.show()

    x = dt.values[:, :]

    return render_template('dataset.html', data=x)


@app.route("/RF")
def rf():
    dt = pd.read_csv("D:\\Research\\spam_email_detection\\static\\spamham.csv")

    msgs = dt.values[:, 1]
    labels = dt.values[:, 0]

    for message_length in range(len(list(msgs))):
        msgs[message_length] = " ".join(cleaning(msgs[message_length]))

    xtrain, xtest, ytrain, ytest = train_test_split(msgs, labels, test_size=0.3, random_state=0)

    vector = TfidfVectorizer(stop_words='english')

    xtrain_vector = vector.fit_transform(xtrain)

    with open("vector.pkl", "wb") as handle:
        pickle.dump(vector, handle, protocol=pickle.HIGHEST_PROTOCOL)

    xtest_vector = vector.transform(xtest)

    Rf = RandomForestClassifier()

    Rf.fit(xtrain_vector, ytrain)

    predicted = Rf.predict(xtest_vector)

    c = confusion_matrix(predicted, ytest)

    plot_confusion_matrix(c, figsize=(6, 4), hide_ticks=True, cmap=plt.cm.Blues)
    plt.xticks(range(2), ['Spam', 'Ham'], fontsize=16)
    plt.yticks(range(2), ['Spam', 'Ham'], fontsize=16)
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")

    plt.show()

    score = accuracy_score(predicted, ytest)

    return render_template('random_forest.html', totalsize=len(msgs), totaltrainsize=len(xtrain),totaltestsize=len(xtest), c=c, xtest=xtest, ytest=ytest, predicted=predicted, score=score)


@app.route("/NaiveBayes")
def naive():
    dt = pd.read_csv("D:\\Research\\spam_email_detection\\static\\spamham.csv")

    x = dt.values[:, 1]
    y = dt.values[:, 0]

    for message_length in range(len(list(x))):
        x[message_length] = " ".join(cleaning(x[message_length]))

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=0)

    train = []

    for i in range(len(xtrain)):
        train.append((xtrain[i], ytrain[i]))

    a = NaiveBayesClassifier(train)

    predict = []
    for i in xtest:
        s = a.classify(i)
        predict.append(s)

    acc = accuracy_score(predict, ytest)

    c = confusion_matrix(predict, ytest)

    plot_confusion_matrix(c, figsize=(6, 4), hide_ticks=True, cmap=plt.cm.Blues)
    plt.xticks(range(2), ['Spam', 'Ham'], fontsize=16)
    plt.yticks(range(2), ['Spam', 'Ham'], fontsize=16)
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")

    plt.show()

    return render_template('naive_bayes.html', totalsize=len(x), totaltrainsize=len(xtrain),totaltestsize=len(xtest), c=c, xtest=xtest, ytest=ytest, predicted=predict, score=acc)


@app.route("/LSTM")
def lstm():
    keras.backend.clear_session()
    df = pd.read_csv('D:\\Research\\spam_email_detection\\static\\spamham.csv')

    df['Category'] = df['Category'].replace("ham", 1)
    df['Category'] = df['Category'].replace("spam", 0)

    msgs = df.values[:, 1]
    labels = df.values[:, 0]

    for message_length in range(len(list(msgs))):
        msgs[message_length] = " ".join(cleaning(msgs[message_length]))

    X_train, X_test, y_train, y_test = train_test_split(msgs, labels, test_size=0.3, random_state=0)

    max_words = 1000
    max_len = 150

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=20)
    X_test_pad = pad_sequences(X_test_seq, maxlen=20)

    lstm_model = Sequential()
    lstm_model.add(Embedding(7982+1, 20, input_length=20))
    lstm_model.add(LSTM(64))
    lstm_model.add(Dense(256, name='FC1', activation='relu'))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(Dense(8, activation='relu'))
    lstm_model.add(Dense(1, activation='sigmoid'))

    lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    lstm_model.summary()

    history = lstm_model.fit(X_train_pad, y_train, validation_split=0.3, batch_size=128, epochs=20)

    history_dict = history.history

    plt.subplot(2, 1, 1)
    plt.plot(history_dict['accuracy'], color='red', label='train')
    plt.plot(history_dict['val_accuracy'], color='blue', label='test')

    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')

    plt.legend(['Training accuracy','Validation accuracy'])

    plt.subplot(2, 1, 2)
    plt.plot(history_dict['loss'], color='red', label='train')
    plt.plot(history_dict['val_loss'], color='blue', label='test')

    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.legend(['Training loss', 'Validation loss'])

    plt.show()

    with open('D:\\Research\\spam_email_detection\\lstm_tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    model_json = lstm_model.to_json()
    with open("D:\\Research\\spam_email_detection\\lstm_model.json", "w") as json_file:
        json_file.write(model_json)
    lstm_model.save_weights("D:\\Research\\spam_email_detection\\lstm_model.h5")

    x = lstm_model.predict(X_test_pad)

    predicted = []

    for i in x:
        predicted.append(round(i[0]))

    lst = []

    for i in y_test:
        lst.append(i)

    pred = []

    for i in predicted:
        if round(i) == 0:
            pred.append("spam")

        else:
            pred.append("ham")

    lst2 = []

    for i in lst:
        if round(i)==0:
            lst2.append("spam")

        else:
            lst2.append("ham")

    c = confusion_matrix(pred, lst2)

    plot_confusion_matrix(c, figsize=(6, 4), hide_ticks=True, cmap=plt.cm.Blues)
    plt.xticks(range(2), ['Spam', 'Ham'], fontsize=16)
    plt.yticks(range(2), ['Spam', 'Ham'], fontsize=16)
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    plt.show()

    score = accuracy_score(lst, predicted)

    return render_template('lstm.html', totalsize=len(df['Category']), totaltrainsize=len(X_train),totaltestsize=len(X_test), c=c, test_texts=X_test, test_labels=y_test,predicted=predicted, score=score)


@app.route("/CNN")
def cnn():
    keras.backend.clear_session()
    sms_df = pd.read_csv('D:\\Research\\spam_email_detection\\static\\spamham.csv')

    sms_df['Category'] = sms_df['Category'].replace("ham", 1)
    sms_df['Category'] = sms_df['Category'].replace("spam", 0)

    labels = sms_df.values[:, 0]
    msgs = sms_df.values[:, 1]

    train_texts, test_texts, train_labels, test_labels = train_test_split(msgs, labels, test_size=0.3, random_state=0)

    VOCABULARY_SIZE = 5000
    tokenizer = Tokenizer(num_words=VOCABULARY_SIZE)
    tokenizer.fit_on_texts(train_texts)

    print("Vocabulary created")

    meanLength = np.mean([len(item.split(" ")) for item in train_texts])
    # MAX_SENTENCE_LENGTH = int(meanLength + 5)
    MAX_SENTENCE_LENGTH = 100
    trainFeatures = tokenizer.texts_to_sequences(train_texts)
    trainFeatures = pad_sequences(trainFeatures, MAX_SENTENCE_LENGTH, padding='post')
    # trainLabels = train_labels.values

    testFeatures = tokenizer.texts_to_sequences(test_texts)
    testFeatures = pad_sequences(testFeatures, MAX_SENTENCE_LENGTH, padding='post')
    # testLabels = test_labels.values

    print("Tokenizing completed")

    FILTERS_SIZE = 16
    KERNEL_SIZE = 5

    EMBEDDINGS_DIM = 10
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    EPOCHS = 20

    maxlen = 0

    for i in trainFeatures:
        if len(i) > maxlen:
            maxlen = len(i)

    model = Sequential()
    model.add(Embedding(input_dim=VOCABULARY_SIZE + 1, output_dim=EMBEDDINGS_DIM, input_length=maxlen))
    model.add(Conv1D(FILTERS_SIZE, KERNEL_SIZE, activation='relu'))
    model.add(Dropout(0.5))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    #optimizer = optimizers.Adam(lr=LEARNING_RATE)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(trainFeatures, train_labels, validation_split=0.3, batch_size=BATCH_SIZE, epochs=EPOCHS)

    history_dict = history.history

    plt.subplot(2, 1, 1)
    plt.plot(history_dict['accuracy'], color='red', label='train')
    plt.plot(history_dict['val_accuracy'], color='blue', label='test')

    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')

    plt.legend(['Training accuracy', 'Validation accuracy'])

    plt.subplot(2, 1, 2)
    plt.plot(history_dict['loss'], color='red', label='train')
    plt.plot(history_dict['val_loss'], color='blue', label='test')

    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.legend(['Training loss', 'Validation loss'])

    plt.show()

    with open('D:\\Research\\spam_email_detection\\tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    model_json = model.to_json()
    with open("D:\\Research\\spam_email_detection\\model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("D:\\Research\\spam_email_detection\\model.h5")

    #GWO implementation
    # n_hidden_nodes = [10, 5]
    # epoch = 10
    # pop_size = 100
    #
    # vector = TfidfVectorizer(stop_words='english')
    # a = vector.fit_transform(train_texts)
    #
    # b = vector.transform(test_texts)
    #
    # dataset = [a, train_labels, b, test_labels]
    # model1 = HybridMlp(dataset, n_hidden_nodes, epoch, pop_size)
    #
    # model1.training()
    #
    # yhat = []
    #
    # for messages in list(test_texts):
    #     yhat.append(model1.prediction(solution=model1.solution, x_data=messages))

    #implementation ends

    x = model.predict(testFeatures)

    predicted = []

    for i in x:
        predicted.append(round(i[0]))

    lst = []

    for i in test_labels:
        lst.append(i)

    pred = []

    for i in predicted:
        if round(i) == 0:
            pred.append("spam")

        else:
            pred.append("ham")

    lst2 = []

    for i in lst:
        if round(i) == 0:
            lst2.append("spam")

        else:
            lst2.append("ham")

    c = confusion_matrix(pred, lst2)

    plot_confusion_matrix(c, figsize=(6, 4), hide_ticks=True, cmap=plt.cm.Blues)
    plt.xticks(range(2), ['Spam', 'Ham'], fontsize=16)
    plt.yticks(range(2), ['Spam', 'Ham'], fontsize=16)
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")

    plt.show()

    score = accuracy_score(lst, predicted)

    return render_template('cnn.html', totalsize=len(msgs), totaltrainsize=len(train_texts),totaltestsize=len(testFeatures), c=c, test_texts=test_texts, test_labels=test_labels,predicted=predicted, score=score)


@app.route("/bioinsp")
def bioinsp():
    return render_template('bioinspired.html')


@app.route("/predict_bioinsp", methods=['POST'])
def predict_bioinsp():
    frm=request.form["txt_msg"]

    n_hidden_nodes = [10, 5]
    epoch = 50
    pop_size = 100
    dt = pd.read_csv("D:\\Research\\spam_email_detection\\static\\spamham.csv")

    dt['Category'] = dt['Category'].replace("ham", 1)
    dt['Category'] = dt['Category'].replace("spam", 0)

    msgs = dt.values[:, 1]
    labels = dt.values[:, 0]

    xtrain, xtest, ytrain, ytest = train_test_split(msgs, labels, test_size=0.3, random_state=0)

    vector = TfidfVectorizer(stop_words='english')
    a = vector.fit_transform(xtrain)

    b = vector.transform(xtest)

    dataset = [a, ytrain, b, ytest]
    model = HybridMlp(dataset, n_hidden_nodes, epoch, pop_size)

    model.training()

    yhat = model.prediction(solution=model.solution, x_data=frm)

    if round(yhat)>0.5:
        res = 'ham'

    else:
        res = 'spam'

    return render_template("bioinspired.html",f=res)


@app.route("/prediction_RF")
def prediction_rf():
    return render_template('prediction_RF.html')


@app.route("/final_prediction_rf", methods=['POST'])
def final_prediction_rf():
    msg = [request.form['txt_msg']]

    dt = pd.read_csv("D:\\Research\\spam_email_detection\\static\\spamham.csv")

    msgs = dt.values[:, 1]
    labels = dt.values[:, 0]

    xtrain, xtest, ytrain, ytest = train_test_split(msgs, labels, test_size=0.3, random_state=0)

    with open("vector.pkl", "rb") as handle:
        vector = pickle.load(handle)

    xtest_vector = vector.transform(xtest)
    xtrain_vector = vector.transform(xtrain)

    rf = RandomForestClassifier()

    rf.fit(xtrain_vector, ytrain)

    msgvector = vector.transform(msg)

    predicted = rf.predict(msgvector)

    return render_template('prediction_RF.html', data=predicted[0])


@app.route("/prediction_NB")
def prediction_nb():
    return render_template('prediction_NB.html')


@app.route("/final_prediction_nb", methods=['POST'])
def final_prediction_nb():
    msg = request.form['txt_msg']

    dt = cleaning(msg)

    string = ''
    for i in dt:
        string += i + " "

    dt = pd.read_csv("D:\\Research\\spam_email_detection\\static\\spamham.csv")

    x = dt.values[:, 1]
    y = dt.values[:, 0]

    train = []

    for i in range(len(x)):
        train.append((x[i], y[i]))

    # Naive Bayes
    a = NaiveBayesClassifier(train)

    s = a.classify(string)

    return render_template('prediction_NB.html', data=s)


@app.route("/prediction_LSTM")
def prediction_lstm():
    return render_template('prediction_LSTM.html')


@app.route("/final_prediction_lstm",methods=['POST'])
def final_prediction_lstm():
    keras.backend.clear_session()
    msg = request.form['txt_msg']

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

    trainFeatures = pad_sequences(f, 20, padding='post')

    loadedmodel.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

    p = loadedmodel.predict(trainFeatures)

    if p[0][0] > 0.5:
        res = 'ham'

    else:
        res = 'spam'

    return render_template('prediction_LSTM.html', data=res)


@app.route("/prediction_CNN")
def prediction_cnn():
    return render_template('prediction_CNN.html')


@app.route("/final_prediction_cnn", methods=['POST'])
def final_prediction_cnn():
    keras.backend.clear_session()
    msg = request.form['txt_msg']

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

    return render_template('prediction_CNN.html', data=res)


@app.route("/results", methods=['GET','POST'])
def results():
    from comparison import find_accurate
    find_accurate()

    dt = pd.read_csv("D:\\Research\\spam_email_detection\\static\\spam_ham_results.csv")

    category = dt.values[:, 2]
    message = dt.values[:, 1]
    algorithm = dt.values[:, 0]

    optimized = max(list(algorithm), key=list(algorithm).count)

    return render_template('accurate_results.html', algorithm=optimized, message=message, category=category, totalSize=len(category))


if __name__ == "__main__":
    app.run(threaded=False)