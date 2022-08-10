import imaplib
import email

import pickle

import keras
from keras import *
from keras import layers, optimizers
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from keras.models import Model, model_from_json
from keras.preprocessing import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

text = []
category = []


def read_email_from_gmail():
        mail = imaplib.IMAP4_SSL('imap.gmail.com')
        mail.login('akhilpurushothaman1996@gmail.com','akhil@1996')
        mail.select('inbox')

        result, data = mail.search(None, 'ALL')
        mail_ids = data[0]

        id_list = mail_ids.split()
        first_email_id = int(id_list[0])
        latest_email_id = int(id_list[-1])

        for i in range(latest_email_id, latest_email_id - 5, -1):
            result, data = mail.fetch(str(i), '(RFC822)' )

            for response_part in data:
                if isinstance(response_part, tuple):
                    # from_bytes, not from_string
                    msg = email.message_from_bytes(response_part[1])
                    k=msg

                    mystring=''

                    for a in [k.get_payload() for k in msg.walk() if k.get_content_type() == 'text/plain']:
                        mystring+=a+' '

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

                    lst = [mystring]

                    f = tokenizer.texts_to_sequences(lst)

                    trainFeatures = pad_sequences(f, maxlen=20, padding='post')

                    loadedmodel.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

                    p = loadedmodel.predict(trainFeatures)

                    if p[0][0] > 0.5:
                        res = 'ham'

                    else:
                        res = 'spam'

                email_subject = msg['subject']
                email_from = msg['from']
                print('From : ' + email_from + '\n')
                print('Subject : ' + email_subject + '\n')
                print('Message : ' + mystring + '\n')
                print("The message is : "+res)

                text.append(mystring)
                category.append(res)

                break
            break

        return [text, category]


#read_email_from_gmail()