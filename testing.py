message = "Even my brother is not like to speak with me. They treat me like aids patent."

from main import *
from sa import *

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


with open('D:\\Research\\spam_email_detection\\testingtokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
model_json = model.to_json()
with open("D:\\Research\\spam_email_detection\\testingmodel.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("D:\\Research\\spam_email_detection\\testingmodel.h5")

x = model.predict(testFeatures)

try:
    model.optimizer = GWO.BaseGWO(model.problem, model.epoch, model.pop_size)
    model.solution, model.best_fit = model.optimizer.solve()

except:
    pass

yhat = model.prediction(solution=model.solution, x_data=testFeatures)

predicted = []

for i in x:
    predicted.append(round(i[0]))

lst = []

for i in test_labels:
    lst.append(i)

c = confusion_matrix(lst, predicted)

plot_confusion_matrix(c, figsize=(6, 4), hide_ticks=True, cmap=plt.cm.Blues)
plt.xticks(range(2), ['Spam', 'Ham'], fontsize=16)
plt.yticks(range(2), ['Spam', 'Ham'], fontsize=16)
plt.ylabel("Actual label")
plt.xlabel("Predicted label")

plt.show()