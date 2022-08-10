import os
import keras.backend
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential,model_from_json
from keras.layers import Dense, LSTM, Embedding, SpatialDropout1D, Dropout
import pickle

df = pd.read_csv('D:\\Research\\spam_email_detection\\static\\spamham.csv')
df.head()

df['Category'] = df['Category'].replace("ham",1)
df['Category'] = df['Category'].replace("spam",0)

df_train = df.sample(frac=1.0)
df_test = df.drop(df_train.index)
print(df_train.shape, df_test.shape)

X_train,X_test,y_train,y_test=train_test_split(df_train['Message'].values,df_train['Category'].values,test_size=0.2,random_state=0)
# y_train = df_train['Category'].values
# y_test = df_test['Category'].values
# y_test.shape
#
# X_train = df_train['Message'].values
# X_test = df_test['Message'].values

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
word_dict = tokenizer.index_word

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
print(X_train_seq[:5])
print(df_train.iloc[0,:])
for el in X_train_seq[0]:
    print(word_dict[el], end=' ')

X_train_pad = pad_sequences(X_train_seq, maxlen=20, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=20, padding='post')
X_train_pad[:5]
X_train_pad.shape

laenge_pads = 20
anz_woerter = 7982

lstm_model = Sequential()
lstm_model.add(Embedding(input_dim=anz_woerter+1, output_dim=20, input_length=laenge_pads))
lstm_model.add(SpatialDropout1D(0.2))
lstm_model.add(LSTM(32, activation='relu'))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(8, activation='relu'))
lstm_model.add(Dense(1, activation='sigmoid'))

lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm_model.summary()

history = lstm_model.fit(X_train_pad, y_train, epochs=20, batch_size=32)

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

c = confusion_matrix(lst, predicted)
score = accuracy_score(lst, predicted)

print(c)
print(score)

#prediction

keras.backend.clear_session()
msg = ['K fyi x has a ride early tomorrow morning but he''s crashing at our place tonight']

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

lst = msg

f = tokenizer.texts_to_sequences(lst)

trainFeatures = pad_sequences(f, maxlen=20, padding='post')

loadedmodel.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

p = loadedmodel.predict(trainFeatures)

print(p)
if p[0][0] > 0.5:
    res = 'ham'

else:
    res = 'spam'

print(res)

# sms_test = ['Hi Paul, would you come around tonight']
# sms_seq = tokenizer.texts_to_sequences(sms_test)
#
# sms_pad = pad_sequences(sms_seq, maxlen=20, padding='post')
# tokenizer.index_word
# #sms_pad
# predictions = (lstm_model.predict(sms_pad) > 0.5).astype("int32")
# predictions
#
# sms_test = ['Free SMS service for anyone']
# sms_seq = tokenizer.texts_to_sequences(sms_test)
#
# sms_pad = pad_sequences(sms_seq, maxlen=20, padding='post')
# tokenizer.index_word
# #sms_pad
# predictions = (lstm_model.predict(sms_pad) > 0.5).astype("int32")
# predictions
# #lstm_model.predict(sms_pad)