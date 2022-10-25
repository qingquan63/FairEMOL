import torch
import numpy as np
import copy


class Mutation_NN:

    def __init__(self, mu=0, var=0.001, p=1, last_num=3):
        # 网络产生的噪音服从正太分布
        self.mu = mu  # 噪音的标准差
        self.var = var  # 噪音的标准差
        self.p = p
        self.last_num = last_num

    def do(self, OldChrom):  # 执行变异
        Offspring = []
        newChrom = copy.deepcopy(OldChrom)
        num_NN = len(OldChrom)
        with torch.no_grad():
            for idx in range(num_NN):
                parent = newChrom[idx]
                if num_NN - idx > self.last_num:
                    if np.random.rand() < self.p:
                        for name, param in parent.named_parameters():
                            weighs = np.array(param.detach())
                            # print(name)
                            # print('  before: ', weighs)
                            if 'bias' not in name:
                                weighs += np.random.normal(loc=self.mu, scale=self.var, size=param.shape)
                            parent.state_dict()[name].data.copy_(torch.Tensor(weighs))
                else:
                    # last num individuals are always performed
                    for name, param in parent.named_parameters():
                        weighs = np.array(param.detach())
                        weighs += np.random.normal(loc=self.mu, scale=self.var, size=param.shape)
                        parent.state_dict()[name].data.copy_(torch.Tensor(weighs))
                Offspring.append(parent)

        return Offspring

    def getHelp(self):  # 查看内核中的变异算子的API文档
        print('check yourself!')


class Crossover_NN:

    def __init__(self, p=1):
        # 网络产生的噪音服从正太分布
        self.p = p

    def do(self, OldChrom1, OldChrom2, coefficient_p=0.5):  # 执行变异
        Offspring = []
        newChrom1 = copy.deepcopy(OldChrom1)
        newChrom2 = copy.deepcopy(OldChrom2)
        num_NN = len(OldChrom1)
        with torch.no_grad():
            for idx in range(num_NN):
                parent1 = newChrom1[idx]
                parent2 = newChrom2[idx]
                if np.random.rand() < self.p:
                    offspring1 = copy.deepcopy(parent1)
                    offspring2 = copy.deepcopy(parent2)
                    for name, param in parent1.named_parameters():
                        par1_weight = np.array(param.detach())
                        par2_weight = np.array(eval('parent2.'+name))
                        # coefficient = np.random.rand()
                        plan = 2
                        if plan == 1:
                            off1_weight = par1_weight * coefficient_p + par2_weight * (1 - coefficient_p)
                            off2_weight = par2_weight * coefficient_p + par1_weight * (1 - coefficient_p)
                        else:
                            coefficient_p = np.random.uniform(0, 1, par1_weight.shape)
                            off1_weight = par1_weight * coefficient_p + par2_weight * (1 - coefficient_p)
                            off2_weight = par2_weight * coefficient_p + par1_weight * (1 - coefficient_p)

                        offspring1.state_dict()[name].data.copy_(torch.Tensor(off1_weight))
                        offspring2.state_dict()[name].data.copy_(torch.Tensor(off2_weight))
                    Offspring.append(offspring1)
                    Offspring.append(offspring2)
                else:
                    Offspring.append(parent1)
                    Offspring.append(parent2)

        return Offspring

    def getHelp(self):  # 查看内核中的变异算子的API文档
        print('check yourself!')





