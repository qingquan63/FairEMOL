# -*- coding: utf-8 -*-
import numpy as np
import FairEMOL as ea
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from FairEMOL.EvolutionaryCodes.load_data import load_data
from itertools import product


def get_label(logits):
    pred_label = logits
    pred_label[np.where(pred_label >= 0.5)] = 1
    pred_label[np.where(pred_label < 0.5)] = 0
    pred_label = pred_label.reshape(1, logits.shape[0] * logits.shape[1])
    pred_label = pred_label.reshape(1, -1)
    return pred_label


def generalized_entropy_index(benefits, alpha):
    # The method is from "Unified Approach to Quantifying Algorithmic Unfairness:
    # Measuring Individual & Group Unfairness via Inequality Indices"
    mu = torch.mean(benefits)
    individual_fitness = torch.mean(torch.pow(benefits / mu, alpha) - 1) / ((alpha - 1) * alpha)
    return individual_fitness


def get_average(group_values, plan):
    if plan == 1:
        values = 0.0
        count = 0
        num_group = len(group_values)
        for i in range(num_group):
            if i == (num_group - 1):
                break
            for j in range(i+1, num_group):
                values += (torch.abs(group_values[i] - group_values[j]))
                count += 1
        if count == 0:
            return torch.zeros(1)[0]
        else:
            return values/count

    else:
        values = torch.zeros(1)[0]
        count = 0
        num_group = len(group_values)
        for i in range(num_group):
            if i == (num_group - 1):
                break
            for j in range(i + 1, num_group):
                values = torch.max(values, torch.abs(group_values[i] - group_values[j]))
                # values.append(torch.abs(group_values[i] - group_values[j]))
                count += 1
        if count == 0:
            return torch.zeros(1)[0]
        else:
            return values


def calcul_all_fairness(data, logits, truelabel, sensitive_attributions, alpha):
    # The method is from "Unified Approach to Quantifying Algorithmic Unfairness:
    # Measuring Individual & Group Unfairness via Inequality Indices"
    sum_num = logits.shape[0] * logits.shape[1]

    benefits = logits - truelabel + 1  # new version in section 3.1
    benefits_mean = torch.mean(benefits)

    attribution = data.columns
    Within_fairness = 0.0
    Group_fairness = 0.0

    group_dict = {}

    for sens in sensitive_attributions:
        temp = []
        for attr in attribution:
            temp1 = sens + '_'
            if temp1 in attr:
                temp.append(attr)
        group_dict.update({sens: temp})

    group_attr = []
    for sens in sensitive_attributions:
        group_attr.append(group_dict[sens])
    for item in product(*eval(str(group_attr))):
        group = item
        flag = np.ones([1, sum_num]) == np.ones([1, sum_num])
        for g in group:
            flag = flag & data[g]
        g_num = np.sum(flag)
        if g_num != 0:
            g_idx = np.array(np.where(flag)).reshape([1, g_num])
            g_benefits = benefits[g_idx].reshape([1, g_num])
            g_fairness = generalized_entropy_index(g_benefits, alpha)
            g_benefits_mean = torch.mean(g_benefits)
            g_individual_fairness = (g_num / sum_num) * (torch.pow(g_benefits_mean / benefits_mean, alpha)) * g_fairness
            g_group_fairness = (g_num / (sum_num * (alpha - 1) * alpha)) * (
                    torch.pow(g_benefits_mean / benefits_mean, alpha) - 1)
            Within_fairness += g_individual_fairness
            Group_fairness += g_group_fairness

    Individual_fairness = generalized_entropy_index(benefits, alpha)

    Group_losses = {"Individual_fairness": Individual_fairness, "Group_fairness": Group_fairness}

    return Group_losses


def get_obj_vals(Group_infos, obj_classes):
    res = np.zeros([1, len(obj_classes)])
    for idx, obj_name in enumerate(obj_classes):
        res[0, idx] = Group_infos[obj_name]
    return res


class FairProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=None, learning_rate=0.01, batch_size=500, sensitive_attributions=None,
                 epoches=2, dataname='german', objectives_class=None, dirname=None, preserve_sens_in_net=0,
                 seed_split_traintest=2021, weight_decay=1e-3, start_time=0, is_ensemble=True):
        if objectives_class is None:
            objectives_class = ['Error', 'Individual_fairness', 'Group_fairness']
            M = len(objectives_class)
        if sensitive_attributions is None:
            if dataname == 'german':
                sensitive_attributions = ['sex']
            else:
                print('There is no dataset called ', dataname)
        name = 'FairnessProblem'
        Dim = 5  # 初始化Dim（决策变量维数）
        maxormins = [1] * M
        varTypes = [1] * Dim
        lb = [0] * Dim
        ub = [10] * Dim
        lbin = [1] * Dim
        ubin = [1] * Dim

        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        self.preserve_sens_in_net = preserve_sens_in_net
        self.seed_split_traintest = seed_split_traintest
        self.weight_decay = weight_decay
        DATA, dataset_obj = load_data(dataname, preserve_sens_in_net=preserve_sens_in_net,
                                      sensitive_attributions=sensitive_attributions, is_ensemble=is_ensemble)
        self.train_data = DATA['train_data']
        self.train_data_norm = DATA['train_data_norm']
        self.train_label = DATA['train_label']
        self.train_y = DATA['train_y']

        self.valid_data = DATA['valid_data']
        self.valid_data_norm = DATA['valid_data_norm']
        self.valid_label = DATA['valid_label']
        self.valid_y = DATA['valid_y']

        if is_ensemble:
            self.ensemble_data = DATA['ensemble_data']
            self.ensemble_data_norm = DATA['ensemble_data_norm']
            self.ensemble_label = DATA['ensemble_label']
            self.ensemble_y = DATA['ensemble_y']

            self.ensemble_data = DATA['ensemble_data']
            self.ensemble_data_norm = DATA['ensemble_data_norm']
            self.ensemble_label = DATA['ensemble_label']
            self.ensemble_y = DATA['ensemble_y']

            self.ensemble_org = DATA['ensemble_org']
            self.num_ensemble = self.ensemble_org.shape[0]

        self.test_data = DATA['test_data']
        self.test_data_norm = DATA['test_data_norm']
        self.test_label = DATA['test_label']
        self.test_y = DATA['test_y']

        self.data_org = DATA['org_data']
        self.train_org = DATA['train_org']
        self.valid_org = DATA['valid_org']
        self.test_org = DATA['test_org']

        self.positive_class_name = DATA['positive_class_name']
        self.positive_class = DATA['positive_class']

        self.Groups_info = DATA['Groups_info']
        self.privileged_class_names = DATA['privileged_class_names']
        self.num_train = self.train_org.shape[0]
        self.num_valid = self.valid_org.shape[0]
        self.num_test = self.test_org.shape[0]

        self.is_ensemble = is_ensemble
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.sensitive_attributions = sensitive_attributions
        self.epoches = epoches
        self.M = M
        self.num_features = self.train_data_norm.shape[1]
        self.dataname = dataname
        self.dataset_obj = dataset_obj
        self.objectives_class = objectives_class
        self.dirname = 'EvolutionaryCodes/' + dirname
        self.cal_sens_name = [sensitive_attributions[0]]
        self.DATA = DATA
        self.ran_flag = 0
        self.start_time = start_time
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.more_info_num = None
        self.use_gpu = False

        self.x_train = None
        self.y_train = None
        self.y_train = None

        self.x_test = None
        self.y_test = None
        self.y_test = None

        self.x_valid = None
        self.y_valid = None
        self.y_valid = None

        self.base_num = 10

    def do_pre(self):

        if self.is_ensemble:
            self.x_ensemble = torch.Tensor(self.ensemble_data_norm)
            self.y_ensemble = torch.Tensor(self.ensemble_y)
            self.y_ensemble = self.y_ensemble.view(self.y_ensemble.shape[0], 1)

        self.x_train = torch.Tensor(self.train_data_norm)
        self.y_train = torch.Tensor(self.train_y)
        self.y_train = self.y_train.view(self.y_train.shape[0], 1)

        self.x_test = torch.Tensor(self.test_data_norm)
        self.y_test = torch.Tensor(self.test_y)
        self.y_test = self.y_test.view(self.y_test.shape[0], 1)

        self.x_valid = torch.Tensor(self.valid_data_norm)
        self.y_valid = torch.Tensor(self.valid_y)
        self.y_valid = self.y_valid.view(self.y_valid.shape[0], 1)

        sens_attr = self.cal_sens_name
        group_dicts = self.Groups_info['group_dict_train']
        s_labels = group_dicts[sens_attr[0]][0]

        sens_idxs_train = self.Groups_info['sens_idxs_train']
        s_labels_train = sens_idxs_train[s_labels][0]
        S_train = np.zeros([1, self.num_train])
        S_train[0, s_labels_train] = 1
        s_train = torch.Tensor(S_train)

        sens_idxs_valid = self.Groups_info['sens_idxs_valid']
        s_labels_valid = sens_idxs_valid[s_labels][0]
        S_valid = np.zeros([1, self.num_valid])
        S_valid[0, s_labels_valid] = 1
        s_valid = torch.Tensor(S_valid)

        sens_idxs_test = self.Groups_info['sens_idxs_test']
        s_labels_test = sens_idxs_test[s_labels][0]
        S_test = np.zeros([1, self.num_test])
        S_test[0, s_labels_test] = 1
        s_test = torch.Tensor(S_test)

        train_idx = torch.arange(start=0, end=self.train_data_norm.shape[0], step=1)
        valid_idx = torch.arange(start=0, end=self.valid_data_norm.shape[0], step=1)
        test_idx = torch.arange(start=0, end=self.test_data_norm.shape[0], step=1)

        train = TensorDataset(self.x_train, self.y_train, s_train[0], train_idx)
        test = TensorDataset(self.x_test, self.y_test, s_test[0], test_idx)
        valid = TensorDataset(self.x_valid, self.y_valid, s_valid[0], valid_idx)

        self.train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test, batch_size=self.batch_size, shuffle=True)

        if self.is_ensemble:
            sens_idxs_ensemble = self.Groups_info['sens_idxs_ensemble']
            s_labels_ensemble = sens_idxs_ensemble[s_labels][0]
            S_ensemble = np.zeros([1, self.num_ensemble])
            S_ensemble[0, s_labels_ensemble] = 1
            s_ensemble = torch.Tensor(S_ensemble)
            ensemble_idx = torch.arange(start=0, end=self.ensemble_data_norm.shape[0], step=1)
            ensemble = TensorDataset(self.x_ensemble, self.y_ensemble, s_ensemble[0], ensemble_idx)
            self.ensemble_loader = DataLoader(ensemble, batch_size=self.batch_size, shuffle=True)

        sens_attr = self.cal_sens_name
        group_dicts = self.Groups_info['group_dict_train']
        s_labels = group_dicts[sens_attr[0]][0]

        sens_idxs_train = self.Groups_info['sens_idxs_train']
        s_labels_train = sens_idxs_train[s_labels][0]
        S_train = np.zeros([1, self.num_train])
        S_train[0, s_labels_train] = 1

        sens_idxs_valid = self.Groups_info['sens_idxs_valid']
        s_labels_valid = sens_idxs_valid[s_labels][0]
        S_valid = np.zeros([1, self.num_valid])
        S_valid[0, s_labels_valid] = 1

        sens_idxs_test = self.Groups_info['sens_idxs_test']
        s_labels_test = sens_idxs_test[s_labels][0]
        S_test = np.zeros([1, self.num_test])
        S_test[0, s_labels_test] = 1

        self.use_gpu = torch.cuda.is_available()
        self.use_gpu = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Groups_info = ea.Cal_objectives(self.train_data, self.train_data_norm,
                                        np.array(self.train_y).reshape(1, -1),
                                        np.array(self.train_y).reshape(1, -1), self.sensitive_attributions,
                                        2, obj_names=self.objectives_class)

        if len(self.Groups_info['group_dict_train']) > 1:
            with open('Result/' + self.dataname + '/' + self.start_time + '/detect/train_sensitive_name.txt', 'a+') as file:
                for sens_name in self.Groups_info['sens_idxs_train']:
                    if '+' in sens_name:
                        file.write(sens_name + ',')
                file.close()
            with open('Result/' + self.dataname + '/' + self.start_time + '/detect/valid_sensitive_name.txt', 'a+') as file:
                for sens_name in self.Groups_info['sens_idxs_valid']:
                    if '+' in sens_name:
                        file.write(sens_name + ',')
                file.close()

            with open('Result/' + self.dataname + '/' + self.start_time + '/detect/test_sensitive_name.txt', 'a+') as file:
                for sens_name in self.Groups_info['sens_idxs_test']:
                    if '+' in sens_name:
                        file.write(sens_name + ',')
                file.close()

            idx_sens_train = np.zeros([1, self.num_train])
            idx_sens_valid = np.zeros([1, self.num_valid])
            idx_sens_test = np.zeros([1, self.num_test])

            idx_count = 0
            for sens_name in self.Groups_info['sens_idxs_train']:
                if '+' in sens_name:
                    idx_sens_train[0][self.Groups_info['sens_idxs_train'][sens_name][0]] = idx_count
                    idx_count += 1
            np.savetxt('Result/' + self.dataname + '/' + self.start_time + '/detect/train_idxs_sensitive.txt', idx_sens_train)

            idx_count = 0
            for sens_name in self.Groups_info['sens_idxs_valid']:
                if '+' in sens_name:
                    idx_sens_valid[0][self.Groups_info['sens_idxs_valid'][sens_name][0]] = idx_count
                    idx_count += 1
            np.savetxt('Result/' + self.dataname + '/' + self.start_time + '/detect/valid_idxs_sensitive.txt', idx_sens_valid)

            idx_count = 0
            for sens_name in self.Groups_info['sens_idxs_test']:
                if '+' in sens_name:
                    idx_sens_test[0][self.Groups_info['sens_idxs_test'][sens_name][0]] = idx_count
                    idx_count += 1
            np.savetxt('Result/' + self.dataname + '/' + self.start_time + '/detect/test_idxs_sensitive.txt', idx_sens_test)

            if self.is_ensemble:
                with open('Result/' + self.dataname + '/' + self.start_time + '/detect/ensemble_sensitive_name.txt', 'a+') as file:
                    for sens_name in self.Groups_info['sens_idxs_ensemble']:
                        if '+' in sens_name:
                            file.write(sens_name + ',')
                    file.close()
                idx_sens_ensemble = np.zeros([1, self.num_ensemble])
                idx_count = 0
                for sens_name in self.Groups_info['sens_idxs_ensemble']:
                    if '+' in sens_name:
                        idx_sens_ensemble[0][self.Groups_info['sens_idxs_ensemble'][sens_name][0]] = idx_count
                        idx_count += 1
                np.savetxt('Result/' + self.dataname + '/' + self.start_time + '/detect/ensemble_idxs_sensitive.txt', idx_sens_ensemble)

        else:

            with open('Result/' + self.dataname + '/' + self.start_time + '/detect/train_sensitive_name.txt', 'a+') as file:
                for sens_name in self.Groups_info['sens_idxs_train']:
                    file.write(sens_name + ',')
                file.close()
            with open('Result/' + self.dataname + '/' + self.start_time + '/detect/valid_sensitive_name.txt', 'a+') as file:
                for sens_name in self.Groups_info['sens_idxs_valid']:
                    file.write(sens_name + ',')
                file.close()

            with open('Result/' + self.dataname + '/' + self.start_time + '/detect/test_sensitive_name.txt', 'a+') as file:
                for sens_name in self.Groups_info['sens_idxs_test']:
                    file.write(sens_name + ',')
                file.close()

            idx_sens_train = np.zeros([1, self.num_train])
            idx_sens_valid = np.zeros([1, self.num_valid])
            idx_sens_test = np.zeros([1, self.num_test])

            idx_count = 0
            for sens_name in self.Groups_info['sens_idxs_train']:
                idx_sens_train[0][self.Groups_info['sens_idxs_train'][sens_name][0]] = idx_count
                idx_count += 1
            np.savetxt('Result/' + self.dataname + '/' + self.start_time + '/detect/train_idxs_sensitive.txt', idx_sens_train)

            idx_count = 0
            for sens_name in self.Groups_info['sens_idxs_valid']:
                idx_sens_valid[0][self.Groups_info['sens_idxs_valid'][sens_name][0]] = idx_count
                idx_count += 1
            np.savetxt('Result/' + self.dataname + '/' + self.start_time + '/detect/valid_idxs_sensitive.txt', idx_sens_valid)

            idx_count = 0
            for sens_name in self.Groups_info['sens_idxs_test']:
                idx_sens_test[0][self.Groups_info['sens_idxs_test'][sens_name][0]] = idx_count
                idx_count += 1
            np.savetxt('Result/' + self.dataname + '/' + self.start_time + '/detect/test_idxs_sensitive.txt', idx_sens_test)

            if self.is_ensemble:
                with open('Result/' + self.dataname + '/' + self.start_time + '/detect/ensemble_sensitive_name.txt', 'a+') as file:
                    for sens_name in self.Groups_info['sens_idxs_ensemble']:
                        file.write(sens_name + ',')
                    file.close()
                idx_sens_ensemble = np.zeros([1, self.num_ensemble])
                idx_count = 0
                for sens_name in self.Groups_info['sens_idxs_ensemble']:
                    idx_sens_ensemble[0][self.Groups_info['sens_idxs_ensemble'][sens_name][0]] = idx_count
                    idx_count += 1
                np.savetxt('Result/' + self.dataname + '/' + self.start_time + '/detect/ensemble_idxs_sensitive.txt', idx_sens_ensemble)
                np.savetxt('Result/' + self.dataname + '/' + self.start_time + '/detect/ensemble_truelabel.txt', np.array(self.ensemble_y))

        np.savetxt('Result/' + self.dataname + '/' + self.start_time + '/detect/train_truelabel.txt', np.array(self.train_y))
        np.savetxt('Result/' + self.dataname + '/' + self.start_time + '/detect/valid_truelabel.txt', np.array(self.valid_y))
        np.savetxt('Result/' + self.dataname + '/' + self.start_time + '/detect/test_truelabel.txt', np.array(self.test_y))

    def getFeature(self):
        return self.num_features

    def aimFunc(self, pop, gen=0, dirName=None, loss_type=-1):  # 目标函数

        self.ran_flag = 1
        popsize = len(pop)
        pred_label_train = np.zeros([popsize, self.num_train])
        pred_label_valid = np.zeros([popsize, self.num_valid])
        pred_label_test = np.zeros([popsize, self.num_test])
        if self.is_ensemble:
            pred_label_ensemble = np.zeros([popsize, self.num_ensemble])
        pred_logits_train = np.zeros([popsize, self.num_train])
        pred_logits_valid = np.zeros([popsize, self.num_valid])
        pred_logits_test = np.zeros([popsize, self.num_test])
        if self.is_ensemble:
            pred_logits_ensemble = np.zeros([popsize, self.num_ensemble])
        AllObj_valid = np.zeros([popsize, len(self.objectives_class)])
        AllObj_test = np.zeros([popsize, len(self.objectives_class)])
        AllObj_train = np.zeros([popsize, len(self.objectives_class)])
        AllObj_ensemble = np.zeros([popsize, len(self.objectives_class)])
        pop_logits_test = np.zeros([popsize, self.test_data_norm.shape[0]])
        for idx in range(popsize):

            individual = pop.Chrom[idx]

            if self.use_gpu:
                individual.cuda()
            lr_now = self.learning_rate
            optimizer = torch.optim.SGD(individual.parameters(), lr=lr_now, momentum=0.9,
                                        weight_decay=self.weight_decay)
            loss_fn = torch.nn.BCEWithLogitsLoss()
            if self.use_gpu:
                loss_fn.cuda()
            if loss_type != -1:
                # exploration: loss_type != -1
                for epoch in range(self.epoches):
                    individual.train()

                    for i, (x_batch, y_batch, s_batch, data_idx) in enumerate(self.train_loader):
                        if self.use_gpu:
                            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
                        y_logits, y_pred = individual(x_batch)

                        loss_acc = loss_fn(y_logits, y_batch)
                        if "Individual_fairness" in self.objectives_class or "Group_fairness" in self.objectives_class:
                            group_losses = calcul_all_fairness(self.train_data.loc[data_idx.detach()], y_pred,
                                                           y_batch, self.sensitive_attributions, 2)

                        if loss_type != -1:
                            if loss_type[idx] == 'Error':
                                loss = loss_acc
                            elif loss_type[idx] == 'Individual_fairness':
                                loss = group_losses['Individual_fairness']
                            elif loss_type[idx] == 'Group_fairness':
                                loss = group_losses['Group_fairness']
                            else:
                                loss = loss_acc

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
            else:
                # exploitation: loss_type = -1
                for epoch in range(self.epoches):
                    individual.train()

                    for i, (x_batch, y_batch, s_batch, data_idx) in enumerate(self.train_loader):
                        if self.use_gpu:
                            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
                        y_logits, y_pred = individual(x_batch)

                        loss_list = []
                        if "Error" in self.objectives_class:
                            loss_list.append(loss_fn(y_logits, y_batch))

                        losses_ = calcul_all_fairness(self.train_data.loc[data_idx.detach()], y_pred,
                                                           y_batch, self.sensitive_attributions, 2)
                        if "Individual_fairness" in self.objectives_class:
                            loss_list.append(losses_['Individual_fairness'])

                        if "Group_fairness" in self.objectives_class:
                            loss_list.append(losses_['Group_fairness'])

                        loss = loss_list[np.random.permutation(len(loss_list))[0]]

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

            with torch.no_grad():
                individual.eval()
                if self.use_gpu:
                    individual.cpu()

                # in the test data
                logit_temp, pred_sigmoid_temp = individual(self.x_test)
                logits_test = np.array(pred_sigmoid_temp.detach())
                pred_label_test[idx][:] = get_label(logits_test.reshape(1, -1).copy())
                pop_logits_test[idx][:] = logits_test.reshape(1, -1)
                pred_logits_test[idx][:] = logits_test.reshape(1, -1)
                Groups_info_test = ea.Cal_objectives(
                    self.test_data,
                    self.test_data_norm,
                    logits_test, self.y_test,
                    self.sensitive_attributions,
                    2, obj_names=self.objectives_class)

                # in the train data
                logit_temp, pred_sigmoid_temp = individual(self.x_train)
                logits_train = np.array(pred_sigmoid_temp.detach())
                pred_label_train[idx][:] = get_label(logits_train.reshape(1, -1).copy())
                pred_logits_train[idx][:] = logits_train.reshape(1, -1)
                Groups_info_train = ea.Cal_objectives(self.train_data,
                                                self.train_data_norm,
                                                logits_train, self.y_train,
                                                self.sensitive_attributions,
                                                2, obj_names=self.objectives_class)

                # in the validation data
                logit_temp, pred_sigmoid_temp = individual(self.x_valid)
                logits_valid = np.array(pred_sigmoid_temp.detach())
                pred_label_valid[idx][:] = get_label(logits_valid.reshape(1, -1).copy())
                pred_logits_valid[idx][:] = logits_valid.reshape(1, -1)
                Groups_info_valid = ea.Cal_objectives(self.valid_data,
                                                    self.valid_data_norm,
                                                    logits_valid, self.y_valid,
                                                    self.sensitive_attributions,
                                                    2, obj_names=self.objectives_class)

                if self.is_ensemble:
                    # in the ensemble data
                    logit_temp, pred_sigmoid_temp = individual(self.x_ensemble)
                    logits_ensemble = np.array(pred_sigmoid_temp.detach())
                    pred_label_ensemble[idx][:] = get_label(logits_ensemble.reshape(1, -1).copy())
                    pred_logits_ensemble[idx][:] = logits_ensemble.reshape(1, -1)
                    Groups_info_ensemble = ea.Cal_objectives(self.ensemble_data,
                                                          self.ensemble_data_norm,
                                                          logits_ensemble, self.y_ensemble,
                                                          self.sensitive_attributions,
                                                          2,obj_names=self.objectives_class)

                AllObj_train[idx][:] = get_obj_vals(Groups_info_train, self.objectives_class)[0]
                AllObj_test[idx][:] = get_obj_vals(Groups_info_test, self.objectives_class)[0]
                AllObj_valid[idx][:] = get_obj_vals(Groups_info_valid, self.objectives_class)[0]
                if self.is_ensemble:
                    AllObj_ensemble[idx][:] = get_obj_vals(Groups_info_ensemble, self.objectives_class)[0]

        pop.CV = np.zeros([popsize, 1])
        pop.ObjV = AllObj_valid
        pop.ObjV_train = AllObj_train
        pop.ObjV_valid = AllObj_valid
        pop.ObjV_test = AllObj_test
        pop.ObjV_ensemble = AllObj_ensemble if self.is_ensemble else None

        pop.pred_label_train = pred_label_train
        pop.pred_label_valid = pred_label_valid
        pop.pred_label_test = pred_label_test
        pop.pred_label_ensemble = pred_label_ensemble if self.is_ensemble else None

        pop.pred_logits_train = pred_logits_train
        pop.pred_logits_valid = pred_logits_valid
        pop.pred_logits_test = pred_logits_test
        pop.pred_logits_ensemble = pred_logits_ensemble if self.is_ensemble else None

        return AllObj_train, AllObj_valid, AllObj_test
