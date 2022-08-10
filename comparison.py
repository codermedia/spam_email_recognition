from main import *  # global_import


def accuracy_rf():
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

    score = accuracy_score(predicted, ytest)

    return score


def accuracy_nb():
    dt = pd.read_csv("D:\\Research\\spam_email_detection\\static\\spamham.csv")

    x = dt.values[:1000, 1]
    y = dt.values[:1000, 0]

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

    score = accuracy_score(predict, ytest)

    return score


def accuracy_lstm():
    keras.backend.clear_session()
    df = pd.read_csv('D:\\Research\\spam_email_detection\\static\\spamham.csv')

    df['Category'] = df['Category'].replace("ham", 1)
    df['Category'] = df['Category'].replace("spam", 0)

    msgs = df.values[:, 1]
    labels = df.values[:, 0]

    for message_length in range(len(list(msgs))):
        msgs[message_length] = " ".join(cleaning(msgs[message_length]))

    X_train, X_test, y_train, y_test = train_test_split(msgs, labels, test_size=0.3, random_state=0)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=20)
    X_test_pad = pad_sequences(X_test_seq, maxlen=20)

    lstm_model = Sequential()
    lstm_model.add(Embedding(7982 + 1, 20, input_length=20))
    lstm_model.add(LSTM(64))
    lstm_model.add(Dense(256, name='FC1', activation='relu'))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(Dense(8, activation='relu'))
    lstm_model.add(Dense(1, activation='sigmoid'))

    lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    lstm_model.summary()

    history = lstm_model.fit(X_train_pad, y_train, validation_split=0.3, batch_size=128, epochs=20)

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
        if round(i) == 0:
            lst2.append("spam")

        else:
            lst2.append("ham")

    score = accuracy_score(lst, predicted)

    return score


def accuracy_cnn():
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

    optimizer = optimizers.Adam(lr=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(trainFeatures, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS)

    with open('D:\\Research\\spam_email_detection\\tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    model_json = model.to_json()
    with open("D:\\Research\\spam_email_detection\\model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("D:\\Research\\spam_email_detection\\model.h5")

    x = model.predict(testFeatures)

    predicted = []

    for i in x:
        predicted.append(round(i[0]))

    lst = []

    for i in test_labels:
        lst.append(i)

    score = accuracy_score(lst, predicted)

    return score


def find_accurate():
    accuracies = dict()

    accuracies['RF'] = accuracy_rf()
    accuracies['NB'] = accuracy_nb()
    accuracies['LSTM'] = accuracy_lstm()
    accuracies['CNN-GWO'] = accuracy_cnn()

    filename = "D:\\Research\\spam_email_detection\\static\\spam_ham_results.csv"

    headers = "Algorithm,Message,Category\n"

    with open(filename, "w") as f:
        f.write(headers)

        for key, value in accuracies.items():
            if value == max(accuracies.values()):
                #print(key, value)

                if key == 'RF':
                    from detection_rf import read_email_from_gmail
                    text, category = read_email_from_gmail()

                elif key == 'NB':
                    from detection_naive import read_email_from_gmail
                    text, category = read_email_from_gmail()

                elif key == 'LSTM':
                    from detection_lstm import read_email_from_gmail
                    text, category = read_email_from_gmail()

                else:
                    from detection_cnn import read_email_from_gmail
                    text, category = read_email_from_gmail()

        for i in range(len(category)):
            f.write(key + "," + " ".join(text[i].split()).replace(",", "") + "," + category[i] + "\n")
