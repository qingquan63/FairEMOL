import torch
import numpy as np
import FairEMOL as ea
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import nn


def sigmoid(x):
    return .5 * (1 + np.tanh(.5 * x))


def weights_init(m, rand_type='uniformity'):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        if rand_type == 'uniformity':
            n = m.in_features
            y = 1.0 / np.sqrt(n)
            torch.nn.init.uniform_(m.weight, -y, y)
            torch.nn.init.zeros_(m.bias)
        elif rand_type == 'normal':
            torch.nn.init.normal_(m.weight, 0.0, 0.05)
            torch.nn.init.zeros_(m.bias)


class IndividualNet(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, dropout=0.3, name='ricci'):
        super(IndividualNet, self).__init__()
        self.name = name
        self.num_hidden = len(n_hidden)
        self.n_hidden = n_hidden

        if self.num_hidden == 1:
            # #hidden layers = 1
            self.hidden_1_1 = torch.nn.Linear(n_feature, n_hidden[0])  # hidden layer
            self.out = torch.nn.Linear(n_hidden[0], n_output)
        else:
            # #hidden layers = 2
            self.hidden_2_1 = torch.nn.Linear(n_feature, n_hidden[0])
            self.hidden_2_2 = torch.nn.Linear(n_hidden[0], n_hidden[1])

            self.out = torch.nn.Linear(n_hidden[1], n_output)

        self.dropout_value = dropout
        if dropout > 0:
            self.dropout = nn.Dropout(self.dropout_value)
        else:
            self.dropout = None
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.num_hidden == 1:
            x = self.hidden_1_1(x)
            if self.dropout_value > 0:
                x = self.dropout(x)

            x = self.relu(x)
            pred_logits = self.out(x)
        else:
            x = self.hidden_2_1(x)
            if self.dropout_value > 0:
                x = self.dropout(x)
            x = self.relu(x)
            x = self.hidden_2_2(x)
            if self.dropout_value > 0:
                x = self.dropout(x)
            x = self.relu(x)
            pred_logits = self.out(x)

        pred_label = torch.sigmoid(pred_logits)

        return pred_logits, pred_label


def mutate(model, var):
    with torch.no_grad():
        for name, param in model.named_parameters():
            weighs = np.array(param.detach())
            weighs += np.random.normal(loc=0, scale=var, size=param.shape)
            model.state_dict()[name].data.copy_(torch.Tensor(weighs))

    return model


class Population_NN:
    def __init__(self, train_data_norm, train_data, train_y, test_data, test_data_norm, test_y, pop_size, n_feature,
                 n_hidden, n_output, sensitive_attributions, positive_y):
        self.train_data = train_data
        self.test_data = test_data
        self.train_y = train_y
        self.test_y = test_y
        self.pop_size = pop_size
        self.n_feature = n_feature
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.population = None
        self.learning_rate = 0.01
        self.batch_size = 500
        self.positive_y = positive_y
        self.train_data_norm = train_data_norm
        self.test_data_norm = test_data_norm
        self.netpara = None
        self.sensitive_attributions = sensitive_attributions

    def initialization(self):
        population = []
        for i in range(self.pop_size):
            pop = IndividualNet(self.n_feature, self.n_hidden, self.n_output)
            pop.apply(weights_init)
            population.append(pop)
        self.population = population

    def train_model(self):
        All_logits = np.array([])
        PopObj = np.zeros([self.pop_size, 3])
        Groups_info = []
        for idx in range(self.pop_size):
            individual = self.population[idx]
            x_train = torch.Tensor(self.train_data_norm)
            y_train = torch.Tensor(self.train_y)
            y_train = y_train.view(y_train.shape[0], 1)

            x_test = torch.Tensor(self.test_data_norm)
            y_test = torch.Tensor(self.test_y)
            y_test = y_test.view(y_test.shape[0], 1)

            optimizer = torch.optim.Adam(individual.parameters(), lr=self.learning_rate, weight_decay=1e-5)
            loss_fn = torch.nn.BCEWithLogitsLoss()  # Combined with the sigmoid

            train = TensorDataset(x_train, y_train)
            train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
            individual.train()
            avg_loss = 0.

            for i, (x_batch, y_batch) in enumerate(train_loader):
                y_pred = individual(x_batch)

                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()  # clear gradients for next train
                loss.backward()  # -> accumulates the gradient (by addition) for each parameter
                optimizer.step()  # -> update weights and biases
                avg_loss += loss.item() / len(train_loader)
            with torch.no_grad():
                a = 0
                if a == 1:
                    # print('The formation in test data: ')
                    logits = sigmoid(np.array(individual(x_test).detach()))
                    accuracy, individual_fairness, group_fairness, Groups = ea.ana_evaluation(self.test_data,
                                                                                           self.test_data_norm,
                                                                                           logits, y_test,
                                                                                           self.sensitive_attributions,
                                                                                           2)
                else:
                    # print('The formation in train data: ')
                    logits = sigmoid(np.array(individual(x_train).detach()))
                    accuracy, individual_fairness, group_fairness, Groups = ea.ana_evaluation(self.train_data,
                                                                                           self.train_data_norm,
                                                                                           logits, y_train,
                                                                                           self.sensitive_attributions,
                                                                                           2)

                Groups_info.append(Groups)
                PopObj[idx][:] = np.array([accuracy, individual_fairness, group_fairness])
                if idx != 0:
                    All_logits = np.concatenate([All_logits, logits], axis=1)
                else:
                    All_logits = np.array(logits)

        return PopObj, Groups_info

    def mutation(self, idx):
        Offspring = []
        for pop_idx in idx:
            parent = self.population[pop_idx]
            parent = mutate(parent, 0.001)
            Offspring.append(parent)

        return Offspring

