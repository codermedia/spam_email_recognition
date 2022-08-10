# no need to import smtplib for this code
# no need to import time for this code
import imaplib
import email

import pandas
from textblob.classifiers import NaiveBayesClassifier

from clean import cleaning

pd = pandas.read_csv('D:\\Research\\spam_email_detection\static\\spamham.csv')

x = pd.values[:2000, 1]
y = pd.values[:2000, 0]

train = []

for i in range(len(x)):
    train.append((x[i], y[i]))

ab = NaiveBayesClassifier(train)

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

        for i in range(latest_email_id,latest_email_id - 5, -1):
            result, data = mail.fetch(str(i), '(RFC822)' )

            for response_part in data:
                if isinstance(response_part, tuple):
                    # from_bytes, not from_string
                    msg = email.message_from_bytes(response_part[1])
                    k=msg

                    mystring=''

                    for a in [k.get_payload() for k in msg.walk() if k.get_content_type() == 'text/plain']:
                        mystring+=a+' '

                    dt = cleaning(mystring)

                    string = ''
                    for i in dt:
                        string += i + " "

                    s = ab.classify(string)

                email_subject = msg['subject']
                email_from = msg['from']
                print ('From : ' + email_from + '\n')
                print ('Subject : ' + email_subject + '\n')
                print('Message : ' + mystring + '\n')
                print("The message is : " + s)

                text.append(mystring)
                category.append(s)

                break
            break

        return [text, category]


#read_email_from_gmail()