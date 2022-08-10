import pickle

import numpy as np
from keras.engine.saving import model_from_json
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score
from mealpy.swarm_based import GWO


class HybridMlp:
    def __init__(self, dataset, n_hidden_nodes, epoch, pop_size):
        self.X_train, self.y_train, self.X_test, self.y_test = dataset[0], dataset[1], dataset[2], dataset[3]
        self.n_hidden_nodes = n_hidden_nodes
        self.epoch = epoch
        self.pop_size = pop_size

        self.n_inputs = self.X_train.shape[1]
        self.model, self.problem_size, self.n_dims, self.problem = None, None, None, None
        self.optimizer, self.solution, self.best_fit = None, None, None

    def create_network(self):
        path1 = "D:\\Research\\spam_email_detection\\model.h5"
        path2 = "D:\\Research\\spam_email_detection\\model.json"
        path3 = "D:\\Research\\spam_email_detection\\tokenizer.pickle"

        jhandle = open(path2, 'r')

        jsoncontent = jhandle.read()

        jhandle.close()

        loadedmodel = model_from_json(jsoncontent)

        loadedmodel.load_weights(path1)

        self.model = loadedmodel

    def create_problem(self):
        self.problem = {
            "obj_func": self.objective_function,
            "lb": [-1, ] * self.n_dims,
            "ub": [1, ] * self.n_dims,
            "minmax": "max",
            "verbose": True,
        }

    def decode_solution(self, solution):
        weight_sizes = [(w.shape, np.size(w)) for w in self.model.get_weights()]

        weights = []
        cut_point = 0
        for ws in weight_sizes:
            temp = np.reshape(solution[cut_point: cut_point + ws[1]], ws[0])
            # [0: 15], (3, 5),
            weights.append(temp)
            cut_point += ws[1]

        self.model.set_weights(weights)

    def prediction(self, solution, x_data):
        try:
            self.decode_solution(solution)

        except:
            pass

        lst = [x_data]

        path3 = "D:\\Research\\spam_email_detection\\tokenizer.pickle"

        with open(path3, "rb") as h:
            tokenizer = pickle.load(h)

        f = tokenizer.texts_to_sequences(lst)
        trainFeatures = pad_sequences(f, 100, padding='post')

        p = self.model.predict(trainFeatures)

        return p[0][0]

    def training(self):
        self.create_network()

        try:
            self.optimizer = GWO.BaseGWO(self.problem, self.epoch, self.pop_size)
            self.solution, self.best_fit = self.optimizer.solve()

        except:
            pass

    def objective_function(self, solution):  # Used in training process
        #self.decode_solution(solution)
        yhat = self.model.predict(self.X_train)
        yhat = np.argmax(yhat, axis=-1).astype('int')
        acc = accuracy_score(self.y_train, yhat)
        return acc
