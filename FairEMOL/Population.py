# -*- coding: utf-8 -*-
import os
import numpy as np
from EvolutionaryCodes.nets import IndividualNet, weights_init
import torch
import time
import copy
from torch import nn


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m, mode='fan_in', nonlinearity='relu')
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class Population:
    def __init__(self, Encoding, Field, NIND, Chrom=None, ObjV=None, ObjV_train=None, ObjV_valid=None, ObjV_ensemble=None, ObjV_test=None,
                 FitnV=None, CV=None, Phen=None, isNN = 0, n_feature=108, n_hidden=100, n_output=1, parameters={},
                 logits=None, pred_label_train=None, pred_label_valid=None, pred_label_ensemble=None, pred_label_test=None,
                 pred_logits_train=None, pred_logits_valid=None, pred_logits_ensemble=None, pred_logits_test=None, info_id=None, family_list=None, is_ensemble=False):

        if isinstance(NIND, int) and NIND >= 0:
            self.sizes = NIND
        else:
            raise RuntimeError('error in Population: Size error. (种群规模设置有误，必须为非负整数。)')
        self.ChromNum = 1
        self.isNN = 1

        self.Encoding = 'NN'
        if Encoding is None:
            self.Field = None
            self.Chrom = None
        else:
            self.Field = Field.copy()
            self.Chrom = copy.deepcopy(Chrom) if Chrom is not None else None
        if Chrom is not None:
            self.Lind = 0
            with torch.no_grad():
                for _ in self.Chrom[0].named_parameters():
                    self.Lind += 1  # 计算染色体的长度
        else:
            self.Lind = 0

        self.ObjV = ObjV.copy() if ObjV is not None else None
        self.ObjV_train = ObjV_train.copy() if ObjV_train is not None else None
        self.ObjV_valid = ObjV_valid.copy() if ObjV_valid is not None else None
        self.ObjV_ensemble = ObjV_ensemble.copy() if ObjV_ensemble is not None else None
        self.ObjV_test = ObjV_test.copy() if ObjV_test is not None else None
        self.FitnV = FitnV.copy() if FitnV is not None else None
        self.CV = CV.copy() if CV is not None else None
        self.Phen = Phen.copy() if Phen is not None else None
        self.logits = logits.copy() if logits is not None else None
        self.pred_label_train = pred_label_train.copy() if pred_label_train is not None else None
        self.pred_label_valid = pred_label_valid.copy() if pred_label_valid is not None else None
        self.pred_label_ensemble = pred_label_ensemble.copy() if pred_label_ensemble is not None else None
        self.pred_label_test = pred_label_test.copy() if pred_label_test is not None else None
        self.pred_logits_train = pred_logits_train.copy() if pred_logits_train is not None else None
        self.pred_logits_valid = pred_logits_valid.copy() if pred_logits_valid is not None else None
        self.pred_logits_ensemble = pred_logits_ensemble.copy() if pred_logits_ensemble is not None else None
        self.pred_logits_test = pred_logits_test.copy() if pred_logits_test is not None else None
        self.n_feature = n_feature
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.start_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime(time.time()))
        self.parameters = parameters
        self.is_ensemble = is_ensemble

    def initChrom(self, NIND=None, dataname='ricci', dropout=0):
        if NIND is not None:
            self.sizes = NIND
        self.ObjV = None
        self.ObjV_train = None
        self.ObjV_valid = None
        self.ObjV_ensemble = None
        self.ObjV_test = None
        self.FitnV = None
        self.CV = None
        population = []
        pop_ids = []
        for i in range(self.sizes):
            pop = copy.deepcopy(IndividualNet(self.n_feature, self.n_hidden, self.n_output, name=dataname, dropout=dropout))
            pop.apply(weights_init)
            population.append(pop)
            pop_ids.append(i)
        self.Chrom = copy.deepcopy(population)
        self.Lind = 0
        with torch.no_grad():
            for name, param in self.Chrom[0].named_parameters():
                self.Lind += 1  # 计算染色体的长度

    def decoding(self):
        Phen = copy.deepcopy(self.Chrom)
        return Phen

    def copy(self):
        return Population(self.Encoding,
                          self.Field,
                          self.sizes,
                          copy.deepcopy(self.Chrom),
                          self.ObjV,
                          self.ObjV_train,
                          self.ObjV_valid,
                          self.ObjV_ensemble,
                          self.ObjV_test,
                          self.FitnV,
                          self.CV,
                          self.Phen,
                          parameters=self.parameters if self.parameters is not None and self.parameters is not None else None,
                          logits=self.logits,
                          pred_label_train=self.pred_label_train,
                          pred_label_valid=self.pred_label_valid,
                          pred_label_ensemble=self.pred_label_ensemble,
                          pred_label_test=self.pred_label_test,
                          pred_logits_train=self.pred_logits_train,
                          pred_logits_valid=self.pred_logits_valid,
                          pred_logits_ensemble=self.pred_logits_ensemble,
                          pred_logits_test=self.pred_logits_test,
                          )

    def __getitem__(self, index):
        if not isinstance(index, (slice, np.ndarray, list, int, np.int32, np.int64)):
            raise RuntimeError(
                'error in Population: index must be an integer, a 1-D list, or a 1-D array. ('
                'index必须是一个整数，一维的列表或者一维的向量。)')
        if isinstance(index, slice):
            NIND = (index.stop - (index.start if index.start is not None else 0)) // (
                index.step if index.step is not None else 1)
            index_array = index
        else:
            index_array = np.array(index).reshape(-1)
            if index_array.dtype == bool:
                NIND = int(np.sum(index_array))
            else:
                NIND = len(index_array)
            if len(index_array) == 0:
                index_array = []

        if self.Encoding is None:
            NewChrom = None
        else:
            if self.Chrom is None:
                raise RuntimeError('error in Population: Chrom is None. (种群染色体矩阵未初始化。)')
            NewChrom = []
            for i in index_array:
                temp = copy.deepcopy(self.Chrom[i])
                NewChrom.append(temp)
            Phen_ = []
            for i in index_array:
                Phen_.append(self.Phen[i])

        return Population(self.Encoding,
                          self.Field,
                          NIND,
                          NewChrom,
                          self.ObjV[index_array] if self.ObjV is not None else None,
                          self.ObjV_train[index_array] if self.ObjV_train is not None else None,
                          self.ObjV_valid[index_array] if self.ObjV_valid is not None else None,
                          self.ObjV_ensemble[index_array] if self.ObjV_ensemble is not None else None,
                          self.ObjV_test[index_array] if self.ObjV_test is not None else None,
                          self.FitnV[index_array] if self.FitnV is not None else None,
                          self.CV[index_array] if self.CV is not None else None,
                          Phen_ if self.Phen is not None else None,
                          isNN=self.isNN,
                          parameters=self.parameters if self.parameters is not None and self.parameters is not None else None,
                          logits=self.logits[index_array] if self.logits is not None else None,
                          pred_label_train=self.pred_label_train[index_array] if self.pred_label_train is not None else None,
                          pred_label_valid=self.pred_label_valid[index_array] if self.pred_label_valid is not None else None,
                          pred_label_ensemble = self.pred_label_ensemble[index_array] if self.pred_label_ensemble is not None else None,
                          pred_label_test=self.pred_label_test[index_array] if self.pred_label_test is not None else None,
                          pred_logits_train=self.pred_logits_train[index_array] if self.pred_logits_train is not None else None,
                          pred_logits_valid=self.pred_logits_valid[index_array] if self.pred_logits_valid is not None else None,
                          pred_logits_ensemble=self.pred_logits_ensemble[index_array] if self.pred_logits_ensemble is not None else None,
                          pred_logits_test=self.pred_logits_test[index_array] if self.pred_logits_test is not None else None,
                          )

    def shuffle(self):
        shuff = np.arange(self.sizes)
        np.random.shuffle(shuff)  # 打乱顺序
        if self.Encoding is None:
            self.Chrom = None
        else:
            if self.Chrom is None:
                raise RuntimeError('error in Population: Chrom is None. (种群染色体矩阵未初始化。)')
            self.Chrom = copy.deepcopy(self.Chrom[shuff, :])
        self.ObjV = self.ObjV[shuff, :] if self.ObjV is not None else None
        self.ObjV_train = self.ObjV_train[shuff, :] if self.ObjV_train is not None else None
        self.ObjV_valid = self.ObjV_valid[shuff, :] if self.ObjV_valid is not None else None
        self.ObjV_ensemble = self.ObjV_ensemble[shuff, :] if self.ObjV_ensemble is not None else None
        self.ObjV_test = self.ObjV_test[shuff, :] if self.ObjV_test is not None else None
        self.FitnV = self.FitnV[shuff] if self.FitnV is not None else None
        self.CV = self.CV[shuff, :] if self.CV is not None else None
        self.Phen = self.Phen[shuff, :] if self.Phen is not None else None
        self.logits = self.logits[shuff, :] if self.logits is not None else None
        self.pred_label_train = self.pred_label_train[shuff, :] if self.pred_label_train is not None else None
        self.pred_label_valid = self.pred_label_valid[shuff, :] if self.pred_label_valid is not None else None
        self.pred_label_ensemble = self.pred_label_ensemble[shuff, :] if self.pred_label_ensemble is not None else None
        self.pred_label_test = self.pred_label_test[shuff, :] if self.pred_label_test is not None else None
        self.pred_logits_train = self.pred_logits_train[shuff, :] if self.pred_logits_train is not None else None
        self.pred_logits_valid = self.pred_logits_valid[shuff, :] if self.pred_logits_valid is not None else None
        self.pred_logits_ensemble = self.pred_logits_ensemble[shuff, :] if self.pred_logits_ensemble is not None else None
        self.pred_logits_test = self.pred_logits_test[shuff, :] if self.pred_logits_test is not None else None

    def __setitem__(self, index, pop):  # 种群个体赋值（种群个体替换）
        # 对index进行格式处理
        if not isinstance(index, (slice, np.ndarray, list, int, np.int32, np.int64)):
            raise RuntimeError(
                'error in Population: index must be an integer, a 1-D list, or a 1-D array. (index必须是一个整数，一维的列表或者一维的向量。)')
        if isinstance(index, slice):
            index_array = index
        else:
            index_array = np.array(index).reshape(-1)
            if len(index_array) == 0:
                index_array = []
        if self.Encoding is not None:
            if self.Encoding != pop.Encoding:
                raise RuntimeError('error in Population: Encoding disagree. (两种群染色体的编码方式必须一致。)')
            if np.all(self.Field == pop.Field) == False:
                raise RuntimeError('error in Population: Field disagree. (两者的译码矩阵必须一致。)')
            if self.Chrom is None:
                raise RuntimeError('error in Population: Chrom is None. (种群染色体矩阵未初始化。)')
            self.Chrom[index_array] = copy.deepcopy(pop.Chrom)
        if self.ObjV is not None:
            if pop.ObjV is None:
                raise RuntimeError('error in Population: ObjV disagree. (两者的目标函数值矩阵必须要么同时为None要么同时不为None。)')
            self.ObjV[index_array] = pop.ObjV
        if self.FitnV is not None:
            if pop.FitnV is None:
                raise RuntimeError('error in Population: FitnV disagree. (两者的适应度列向量必须要么同时为None要么同时不为None。)')
            self.FitnV[index_array] = pop.FitnV

        if self.logits is not None:
            if pop.logits is None:
                raise RuntimeError('error in Population: logits disagree. (两者的logits向量必须要么同时为None要么同时不为None。)')
            self.logits[index_array] = pop.logits

        if self.pred_label_train is not None:
            if pop.pred_label_train is None:
                raise RuntimeError('error in Population: pred_label_train disagree. (两者的logits向量必须要么同时为None要么同时不为None。)')
            self.pred_label_train[index_array] = pop.pred_label_train

        if self.pred_label_valid is not None:
            if pop.pred_label_valid is None:
                raise RuntimeError('error in Population: pred_label_valid disagree. (两者的logits向量必须要么同时为None要么同时不为None。)')
            self.pred_label_valid[index_array] = pop.pred_label_valid

        if self.pred_label_ensemble is not None:
            if pop.pred_label_ensemble is None:
                raise RuntimeError('error in Population: pred_label_ensemble disagree. (两者的logits向量必须要么同时为None要么同时不为None。)')
            self.pred_label_ensemble[index_array] = pop.pred_label_ensemble

        if self.pred_label_test is not None:
            if pop.pred_label_test is None:
                raise RuntimeError('error in Population: logits disagree. (两者的logits向量必须要么同时为None要么同时不为None。)')
            self.pred_label_test[index_array] = pop.pred_label_test

        if self.pred_logits_train is not None:
            if pop.pred_logits_train is None:
                raise RuntimeError('error in Population: pred_label_train disagree. (两者的logits向量必须要么同时为None要么同时不为None。)')
            self.pred_logits_train[index_array] = pop.pred_logits_train

        if self.pred_logits_valid is not None:
            if pop.pred_logits_valid is None:
                raise RuntimeError('error in Population: pred_logits_valid disagree. (两者的logits向量必须要么同时为None要么同时不为None。)')
            self.pred_logits_valid[index_array] = pop.pred_logits_valid

        if self.pred_logits_ensemble is not None:
            if pop.pred_logits_ensemble is None:
                raise RuntimeError('error in Population: pred_logits_ensemble disagree. (两者的logits向量必须要么同时为None要么同时不为None。)')
            self.pred_logits_ensemble[index_array] = pop.pred_logits_ensemble

        if self.pred_logits_test is not None:
            if pop.pred_logits_test is None:
                raise RuntimeError('error in Population: logits disagree. (两者的logits向量必须要么同时为None要么同时不为None。)')
            self.pred_logits_test[index_array] = pop.pred_logits_test

        if self.CV is not None:
            if pop.CV is None:
                raise RuntimeError('error in Population: CV disagree. (两者的违反约束程度矩阵必须要么同时为None要么同时不为None。)')
            self.CV[index_array] = pop.CV
        if self.Phen is not None:
            if pop.Phen is None:
                raise RuntimeError('error in Population: Phen disagree. (两者的表现型矩阵必须要么同时为None要么同时不为None。)')
            self.Phen[index_array] = pop.Phen
        self.sizes = self.Phen.shape[0]  # 更新种群规模

    def __add__(self, pop):
        if self.Encoding is None:
            NewChrom = None
        else:
            if self.Encoding != pop.Encoding:
                raise RuntimeError('error in Population: Encoding disagree. (两种群染色体的编码方式必须一致。)')
            if self.Chrom is None or pop.Chrom is None:
                raise RuntimeError('error in Population: Chrom is None. (种群染色体矩阵未初始化。)')
            if np.all(self.Field == pop.Field) == False:
                raise RuntimeError('error in Population: Field disagree. (两者的译码矩阵必须一致。)')
            temp1 = copy.deepcopy(self.Chrom)
            temp2 = copy.deepcopy(pop.Chrom)
            NewChrom = np.transpose(np.hstack([temp1, temp2]))
            NewPhen = np.transpose(np.hstack([self.Phen, pop.Phen]))
        NIND = self.sizes + pop.sizes  # 得到合并种群的个体数
        return Population(self.Encoding,
                          self.Field,
                          NIND,
                          NewChrom,
                          np.vstack([self.ObjV, pop.ObjV]) if self.ObjV is not None and pop.ObjV is not None else None,
                          np.vstack([self.ObjV_train, pop.ObjV_train]) if self.ObjV_train is not None and pop.ObjV_train is not None else None,
                          np.vstack([self.ObjV_valid, pop.ObjV_valid]) if self.ObjV_valid is not None and pop.ObjV_valid is not None else None,
                          np.vstack([self.ObjV_ensemble, pop.ObjV_ensemble]) if self.ObjV_ensemble is not None and pop.ObjV_ensemble is not None else None,
                          np.vstack([self.ObjV_test, pop.ObjV_test]) if self.ObjV_test is not None and pop.ObjV_test is not None else None,
                          np.vstack(
                              [self.FitnV, pop.FitnV]) if self.FitnV is not None and pop.FitnV is not None else None,
                          np.vstack([self.CV, pop.CV]) if self.CV is not None and pop.CV is not None else None,
                          NewPhen if self.Phen is not None and pop.Phen is not None else None,
                          parameters=self.parameters if self.parameters is not None and pop.parameters is not None else None,
                          logits=np.vstack(
                              [self.logits, pop.logits]) if self.logits is not None and pop.logits is not None else None,
                          pred_label_train=np.vstack(
                              [self.pred_label_train,
                               pop.pred_label_train]) if self.pred_label_train is not None and pop.pred_label_train is not None else None,
                          pred_label_valid=np.vstack(
                              [self.pred_label_valid,
                               pop.pred_label_valid]) if self.pred_label_valid is not None and pop.pred_label_valid is not None else None,
                          pred_label_ensemble=np.vstack(
                              [self.pred_label_ensemble,
                               pop.pred_label_ensemble]) if self.pred_label_ensemble is not None and pop.pred_label_ensemble is not None else None,
                          pred_label_test=np.vstack(
                              [self.pred_label_test,
                               pop.pred_label_test]) if self.pred_label_test is not None and pop.pred_label_test is not None else None,
                          pred_logits_train = np.vstack(
                                [self.pred_logits_train,
                                 pop.pred_logits_train]) if self.pred_logits_train is not None and pop.pred_logits_train is not None else None,
                          pred_logits_valid = np.vstack(
                                [self.pred_logits_valid,
                                 pop.pred_logits_valid]) if self.pred_logits_valid is not None and pop.pred_logits_valid is not None else None,
                          pred_logits_ensemble = np.vstack(
                                [self.pred_logits_ensemble,
                                 pop.pred_logits_ensemble]) if self.pred_logits_ensemble is not None and pop.pred_logits_ensemble is not None else None,
                          pred_logits_test = np.vstack(
                                [self.pred_logits_test,
                                 pop.pred_logits_test]) if self.pred_logits_test is not None and pop.pred_logits_test is not None else None,
                          )

    def __len__(self):
        return self.sizes

    def save_network(self, gen, save_dir, network=None):
        if network is None:
            count_NN = len(self.Chrom)
            network = copy.deepcopy(self.Chrom)
        else:
            count_NN = len(network)
        for idx in range(count_NN):
            NN = network[idx]
            save_filename = 'nets/gen%d_net%s.pth' % (gen, idx)
            save_path = os.path.join(save_dir, save_filename)
            torch.save(NN, save_path)

    def save(self, dirName='Result', Gen=0, NNmodel=None, Res_metrics=None,
             All_objs_train=None, All_objs_valid=None, All_objs_ensemble=None, All_objs_test=None, true_y=None, poplogits=None, runtime=0):

        if self.sizes > 0:
            # if self.FitnV is not None:
            #     save_filename = '/others/Gen%d_FitnV.csv' % Gen
            #     np.savetxt(dirName + save_filename, self.FitnV, delimiter=',')

            if All_objs_train is not None:
                save_filename = '/allobjs/ALL_Objs_train_gen%d_sofar.csv' % Gen
                record_gen = list(All_objs_train.keys())
                with open(dirName + save_filename, 'a+') as file:
                    for gen in record_gen:
                        popobj = All_objs_train[gen]
                        for i in range(popobj.shape[0]):
                            line = gen + ','
                            line += ','.join(str(x) for x in popobj[i, :]) + '\n'
                            file.write(line)
                    file.close()

            if All_objs_valid is not None:
                save_filename = '/allobjs/ALL_Objs_valid_gen%d_sofar.csv' % Gen
                record_gen = list(All_objs_valid.keys())
                with open(dirName + save_filename, 'a+') as file:
                    for gen in record_gen:
                        popobj = All_objs_valid[gen]
                        for i in range(popobj.shape[0]):
                            line = gen + ','
                            line += ','.join(str(x) for x in popobj[i, :]) + '\n'
                            file.write(line)
                    file.close()

            if All_objs_ensemble is not None:
                save_filename = '/allobjs/ALL_Objs_ensemble_gen%d_sofar.csv' % Gen
                record_gen = list(All_objs_ensemble.keys())
                with open(dirName + save_filename, 'a+') as file:
                    for gen in record_gen:
                        popobj = All_objs_ensemble[gen]
                        for i in range(popobj.shape[0]):
                            line = gen + ','
                            line += ','.join(str(x) for x in popobj[i, :]) + '\n'
                            file.write(line)
                    file.close()

            if All_objs_test is not None:
                save_filename = '/allobjs/ALL_Objs_test_gen%d_sofar.csv' % Gen
                record_gen = list(All_objs_test.keys())
                with open(dirName + save_filename, 'a+') as file:
                    for gen in record_gen:
                        popobj = All_objs_test[gen]
                        for i in range(popobj.shape[0]):
                            line = gen + ','
                            line += ','.join(str(x) for x in popobj[i, :]) + '\n'
                            file.write(line)
                    file.close()

    def printPare(self, test_org, record_parameter, base_res_path=None):
        start_time = self.parameters['start_time']
        if not os.path.exists(base_res_path + start_time):
            os.makedirs(base_res_path + start_time)
            os.makedirs(base_res_path + start_time + '/nets')
            os.makedirs(base_res_path + start_time + '/allobjs')
            os.makedirs(base_res_path + start_time + '/detect')

        with open(base_res_path + start_time + '/Parameters.txt', 'a+') as file:
            for name in self.parameters:
                if name in record_parameter:
                    strname = name + ' : ' + str(self.parameters[name]) + '\n'
                    file.write(strname)
            file.close()

    def setisNN(self, isNN):
        self.isNN = isNN
        self.Encoding = 'NN'

    def set_indiv_logjts(self, logits):
        self.logits = logits
        print('set logit vector')
        return Population(self.Encoding,
                          self.Field,
                          self.sizes,
                          copy.deepcopy(self.Chrom),
                          self.ObjV,
                          self.ObjV_train,
                          self.ObjV_valid,
                          self.ObjV_ensemble,
                          self.ObjV_test,
                          self.FitnV,
                          self.CV,
                          self.Phen,
                          parameters=self.parameters if self.parameters is not None and self.parameters is not None else None,
                          logits=self.logits,
                          pred_label_train=self.pred_label_train,
                          pred_label_valid=self.pred_label_valid,
                          pred_label_ensemble=self.pred_label_ensemble,
                          pred_label_test=self.pred_label_test,
                          pred_logits_train=self.pred_logits_train,
                          pred_logits_valid=self.pred_logits_valid,
                          pred_logits_ensemble=self.pred_logits_ensemble)

    def get_indiv_logjts(self):
        return self.logits
