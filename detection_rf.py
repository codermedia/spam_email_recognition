# no need to import smtplib for this code
# no need to import time for this code
import imaplib
import email
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from clean import cleaning

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

        for i in range(latest_email_id,latest_email_id - 5,-1):
            result, data = mail.fetch(str(i), '(RFC822)' )

            for response_part in data:
                if isinstance(response_part, tuple):
                    # from_bytes, not from_string
                    msg = email.message_from_bytes(response_part[1])
                    k=msg

                    mystring=''

                    for a in [k.get_payload() for k in msg.walk() if k.get_content_type() == 'text/plain']:
                        mystring+=a+' '

                    dt = pd.read_csv("D:\\Riss Technologies\\programs\\spam_design\\static\\spamham.csv")

                    msgs = dt.values[:, 1]
                    labels = dt.values[:, 0]

                    xtrain, xtest, ytrain, ytest = train_test_split(msgs, labels, test_size=0.2, random_state=0)

                    with open("vector.pkl", "rb") as handle:
                        vector = pickle.load(handle)

                    xtest_vector = vector.transform(xtest)
                    xtrain_vector = vector.transform(xtrain)

                    Rf = RandomForestClassifier()

                    Rf.fit(xtrain_vector, ytrain)

                    msgvector = vector.transform([mystring])

                    predicted = Rf.predict(msgvector)

                email_subject = msg['subject']
                email_from = msg['from']
                print ('From : ' + email_from + '\n')
                print ('Subject : ' + email_subject + '\n')
                print('Message : ' + mystring + '\n')
                print("The message is : " + predicted[0])

                text.append(mystring)
                category.append(predicted[0])

                break
            break

        return [text,category]


#read_email_from_gmail()