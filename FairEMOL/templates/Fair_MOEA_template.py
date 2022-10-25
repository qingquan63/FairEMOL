# -*- coding: utf-8 -*-
import numpy as np
import FairEMOL as ea
from sys import path as paths
from os import path
import torch
import time
import os
import copy

paths.append(path.split(path.split(path.realpath(__file__))[0])[0])


def save_model(population, gen, filepath='nets/gen%d_net%s.pth', start_time=None, base_res_path=None):
    for idx in range(len(population)):
        NN = population.Chrom[idx]
        save_filename = filepath % (gen, idx)
        save_path = os.path.join(base_res_path + start_time, save_filename)
        torch.save(NN.state_dict(), save_path)


def k_tournament(k, fitness, num):
    selection_idxs = []
    fitness = fitness.reshape(1, -1)
    N = len(fitness[0])

    for i in range(num):
        cmps = np.random.randint(0, N, k)
        fits = fitness[0, cmps]
        best_idx = np.argmax(fits)
        selection_idxs.append(cmps[best_idx])

    return selection_idxs


class Fair_MOEA_template(ea.MoeaAlgorithm):

    def __init__(self, problem, start_time, population, muta_mu=0, muta_var=0.001, objectives=None,
                 calculmetric=20, run_id=0, MOEAs=1, mutation_p=0.2, crossover_p=0.8,
                 record_parameter=None, is_ensemble=False):
        ea.MoeaAlgorithm.__init__(self, problem, population)  # 先调用父类构造方法
        if objectives is None:
            objectives = ['Individual_fairness', 'Group_fairness']
        if population.ChromNum != 1:
            raise RuntimeError('传入的种群对象必须是单染色体的种群类型。')
        self.name = 'NSGA2'
        if self.problem.M < 10:
            self.ndSort = ea.ndsortESS  # 采用ENS_SS进行非支配排序
        else:
            self.ndSort = ea.ndsortTNS  # 高维目标采用T_ENS进行非支配排序，速度一般会比ENS_SS要快
        self.selFunc = 'tour'  # 选择方式，采用锦标赛选择

        self.recOper = ea.Crossover_NN(crossover_p)
        self.mutOper = ea.Mutation_NN(mu=muta_mu, var=muta_var, p=mutation_p)

        self.start_time = start_time
        self.dirName = 'Result/'
        self.objectives_class = objectives
        self.calculmetric = calculmetric
        self.run_id = run_id
        self.MOEAs = MOEAs
        self.mutation_p = mutation_p
        self.crossover_p = crossover_p
        self.muta_mu = muta_mu
        self.muta_var_org = muta_var
        self.muta_var = muta_var
        self.record_parameter = record_parameter
        self.is_ensemble = is_ensemble

    def reinsertion(self, population, offspring, NUM, isNN=1):
        population = population + offspring
        population.setisNN(1)

        if self.MOEAs == 'NSGAII':
            # NSGA--II
            [levels, criLevel] = self.ndSort(population.ObjV, NUM, None, population.CV,
                                             self.problem.maxormins)  # 对NUM个个体进行非支配分层
            dis = ea.crowdis(population.ObjV, levels)  # 计算拥挤距离
            population.FitnV[:, 0] = np.argsort(np.lexsort(np.array([dis, -levels])), kind='mergesort')  # 计算适应度
            chooseFlag = ea.selecting('dup', population.FitnV, NUM)  # 调用低级选择算子dup进行基于适应度排序的选择，保留NUM个个体
        elif self.MOEAs == 'Single-Objective':
            # Single-Objective
            objs = copy.deepcopy(population.ObjV)
            population.FitnV[:, 0] = -objs[:, 0]  # 计算适应度
            chooseFlag_sort1 = np.argsort(objs[:, 0])
            chooseFlag_sort = np.argsort(chooseFlag_sort1)
            chooseFlag_idx = np.where(chooseFlag_sort < NUM)
            chooseFlag = chooseFlag_idx[0]
        elif self.MOEAs == 'SRA':
            # SRA
            ObjV = copy.deepcopy(population.ObjV)
            chooseFlag = ea.SRA_env_selection(ObjV, NUM)
        elif self.MOEAs == 'SRA_new':
            # SRA new
            [levels, criLevel] = self.ndSort(population.ObjV, NUM, None, population.CV,
                                             self.problem.maxormins)  # 对NUM个个体进行非支配分层
            ObjV = copy.deepcopy(population.ObjV)
            chooseFlag = ea.SRA_env_selection2(ObjV, NUM, levels)

        return population[chooseFlag], chooseFlag

    def update_passtime(self):
        self.passTime += time.time() - self.timeSlot

    def update_timeslot(self):
        self.timeSlot = time.time()

    def run(self, prophetPop=None):

        base_res_path = 'Result/{}/'.format(self.problem.dataname)
        if not os.path.exists(base_res_path):
            os.makedirs(base_res_path)

        self.population.printPare(self.problem.test_org, self.record_parameter, base_res_path)

        population = self.population
        NIND = population.sizes
        self.problem.do_pre()
        self.initialization(is_ensemble=self.is_ensemble)
        population.initChrom(dataname=self.problem.dataname)
        self.update_passtime()
        population.save(dirName=base_res_path, Gen=-1, NNmodel=population.Chrom)
        self.update_timeslot()
        gen = 1
        self.call_aimFunc(population, gen=gen, dirName=base_res_path)

        if self.MOEAs == 'NSGAII' or self.MOEAs == 'Single-Objective':
            [levels, criLevel] = self.ndSort(population.ObjV, NIND, None, population.CV,
                                             self.problem.maxormins)
            population.FitnV = (1 / levels).reshape(-1, 1)

        self.update_passtime()
        self.add_gen2info(population, gen)
        self.update_timeslot()

        while not self.terminated(population):
            # self.muta_var = self.muta_var_org - gen * (self.muta_var_org * 0.9) / self.MAXGEN
            # self.mutOper = ea.Mutation_NN(mu=self.muta_mu, var=self.muta_var, p=self.mutation_p)
            gen += 1
            print('Gen', gen)
            K_num = 2
            diff_num = 1
            if "Individual_fairness" in self.problem.objectives_class:
                diff_num += 1
            if "Group_fairness" in self.problem.objectives_class:
                diff_num += 1
            MOEA_sel_num = NIND - K_num * diff_num

            if self.MOEAs == 'Single-Objective':
                # single-objective
                acc_obj = -population.ObjV[:, 0]
                better_parents = k_tournament(2, acc_obj.reshape(-1, 1), MOEA_sel_num)
            elif self.MOEAs == 'NSGAII':
                better_parents = k_tournament(2, population.FitnV, MOEA_sel_num)
            elif self.MOEAs == 'SRA':
                # SRA
                better_parents = list(np.random.randint(0, NIND, MOEA_sel_num))
            elif self.MOEAs == 'SRA_new':
                # SRA + extreme
                better_parents = list(np.random.randint(0, NIND, MOEA_sel_num))

            extreme = np.argmin(population.ObjV, axis=0)

            offspring_better = population[better_parents].copy()
            offsprint_extrme = population[extreme].copy()

            if self.crossover_p > 0:
                offspring_better.Chrom[0:np.int(np.floor(MOEA_sel_num / 2) * 2)] = self.recOper.do(
                    offspring_better.Chrom[0:np.int(np.floor(MOEA_sel_num / 2))],
                    offspring_better.Chrom[np.int(np.floor(MOEA_sel_num / 2)):np.int(np.floor(MOEA_sel_num / 2) * 2)],
                    np.random.uniform(0, 1, 1))
            offspring_better.Chrom = self.mutOper.do(offspring_better.Chrom)
            # exploitation: loss_type = -1
            self.call_aimFunc(offspring_better, gen=gen, dirName=base_res_path, loss_type=-1)
            offspring = offspring_better.copy()

            if len(self.problem.objectives_class) > 1:
                # exploration
                offsprint_extrme.Chrom = self.mutOper.do(offsprint_extrme.Chrom)
                for k in range(K_num):
                    if "Individual_fairness" in self.problem.objectives_class or "Group_fairness" in self.problem.objectives_class:
                        self.call_aimFunc(offsprint_extrme, gen=gen, dirName=base_res_path,
                                          loss_type=self.problem.objectives_class)
                        offspring = offspring + offsprint_extrme
                    else:
                        self.call_aimFunc(offsprint_extrme[0], gen=gen, dirName=base_res_path,
                                          loss_type=['Error'])  # 0 is the index of "Error"
                        offspring = offspring + offsprint_extrme[0]

            population, chooseidx = self.reinsertion(population, offspring, NIND, isNN=1)

            self.update_passtime()
            self.add_gen2info(population, gen)
            self.update_timeslot()
            if np.mod(gen, self.calculmetric) == 0 or gen == 2:
                self.update_passtime()

                save_model(population, gen, filepath='nets/gen%d_net%s.pth', start_time=self.start_time,
                           base_res_path=base_res_path)

                now_time = time.strftime("%d %H.%M:%S", time.localtime(time.time()))
                print(now_time, " Run ID ", self.run_id, ", Gen :", gen, ", runtime:", str(self.passTime))
                np.savetxt(base_res_path + self.start_time + '/detect/passtime_gen{}.txt'.format(gen),
                           np.array([self.passTime]))

                np.savetxt(base_res_path + self.start_time + '/detect/pop_logits_train{}.txt'.format(gen),
                           population.pred_logits_train)
                np.savetxt(base_res_path + self.start_time + '/detect/pop_logits_valid{}.txt'.format(gen),
                           population.pred_logits_valid)
                if self.is_ensemble:
                    np.savetxt(base_res_path + self.start_time + '/detect/pop_logits_ensemble{}.txt'.format(gen),
                               population.pred_logits_ensemble)
                np.savetxt(base_res_path + self.start_time + '/detect/pop_logits_test{}.txt'.format(gen),
                           population.pred_logits_test)

                np.savetxt(base_res_path + self.start_time + '/detect/popobj_train{}.txt'.format(gen),
                           population.ObjV_train)
                np.savetxt(base_res_path + self.start_time + '/detect/popobj_valid{}.txt'.format(gen),
                           population.ObjV_valid)
                if self.is_ensemble:
                    np.savetxt(base_res_path + self.start_time + '/detect/popobj_ensemble{}.txt'.format(gen),
                               population.ObjV_ensemble)
                np.savetxt(base_res_path + self.start_time + '/detect/popobj_test{}.txt'.format(gen),
                           population.ObjV_test)

                self.update_timeslot()

        population.save(dirName=base_res_path + self.start_time, Gen=gen, NNmodel=population,
                        All_objs_train=self.all_objetives_train,
                        All_objs_valid=self.all_objetives_valid,
                        All_objs_test=self.all_objetives_test,
                        All_objs_ensemble=self.all_objetives_ensemble,
                        true_y=np.array(self.problem.test_y),
                        runtime=self.passTime)

        print("Run ID ", self.run_id, "finished!")
        return self.finishing(population)
