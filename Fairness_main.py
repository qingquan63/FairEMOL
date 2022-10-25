# -*- coding: utf-8 -*-
import FairEMOL as ea
from FairEMOL.FairProblem import FairProblem
import time
import numpy as np
import sys


def run(parameters):
    start_time = parameters['start_time']
    print('The time is ', start_time)
    """===============================实例化问题对象============================"""
    problem = FairProblem(M=len(parameters['objectives_class']), learning_rate=parameters['learning_rate'],
                          batch_size=parameters['batch_size'],
                          sensitive_attributions=parameters['sensitive_attributions'],
                          epoches=parameters['epoches'], dataname=parameters['dataname'],
                          objectives_class=parameters['objectives_class'],
                          dirname='Result/' + parameters['start_time'],
                          seed_split_traintest=parameters['seed_split_traintest'],
                          start_time=parameters['start_time'],
                          is_ensemble=parameters["is_ensemble"])
    """==================================种群设置==============================="""
    Encoding = 'NN'
    NIND = parameters['NIND']
    Field = ea.crtfld('BG', problem.varTypes, problem.ranges, problem.borders,
                      [10] * len(problem.varTypes))  # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND, n_feature=problem.getFeature(),
                               n_hidden=parameters['n_hidden'],
                               n_output=parameters['n_output'],
                               parameters=parameters,
                               logits=np.zeros([NIND, problem.test_data.shape[0]]),
                               is_ensemble=parameters["is_ensemble"])  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    """================================算法参数设置============================="""
    myAlgorithm = ea.Fair_MOEA_template(problem=problem, start_time=start_time,
                                        population=population,
                                        muta_mu=parameters['muta_mu'],
                                        muta_var=parameters['muta_var'],
                                        calculmetric=parameters['logMetric'],
                                        run_id=parameters['run_id'],
                                        MOEAs=parameters['MOEAs'],
                                        mutation_p=parameters["mutation_p"],
                                        crossover_p=parameters["crossover_p"],
                                        record_parameter=parameters["record_parameter"],
                                        is_ensemble=parameters["is_ensemble"])
    myAlgorithm.MAXGEN = parameters['MAXGEN']
    myAlgorithm.logTras = parameters['logTras']
    myAlgorithm.verbose = parameters['verbose']
    myAlgorithm.drawing = parameters['drawing']

    myAlgorithm.run()


if __name__ == '__main__':
    try:
        run_ids = sys.argv[1]
    except:
        run_ids = 1

    run_ids = int(run_ids)

    # list_objs = ['Error', "Equalized_odds", "Error_diff", "Discovery_ratio", "Predictive_equality",
    #              "FOR_diff", "FOR_ratio", "FNR_diff", "FNR_ratio"]
    list_objs = ['Error', 'Individual_fairness', 'Group_fairness']

    for run_id in range(run_ids, run_ids + 110):
        if (run_id >= 0) & (run_id <= 10):
            dataname = 'german'
            sensitive_attributions = ['sex', 'age']
            objectives_class = list_objs
            learning_rate, weight_decay, muta_var, n_hidden = 0.004, 0.001, 0.01, [64]
            batch_size, epoches = 40, 1
            mutation_p, crossover_p = 1, 1
            MOEAs = 'SRA_new'  # NSGAII, SRA, SRA_new, Single-Objective

        MAXGEN, popsize = 10, 10
        is_ensemble = False  # 是否进行ensemble
        drawing = 0
        logMetric = 10

        preserve_sens_in_net = 0
        logTras = 1
        n_output = 1
        muta_mu = 0
        verbose = False
        dirName = 'Result//'
        start_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))  # 时间 决定文件夹的名字，保证唯一性
        seed_split_traintest = 1234
        Encoding = 'BG'  # 无所谓，因为原来算法需要一个参数，实际并不影响进程
        record_parameter = ['run_id', 'start_time', 'learning_rate', 'batch_size', 'n_hidden', 'n_output', 'epoches',
                            'muta_mu', 'muta_var', 'NIND', 'MAXGEN', 'dataname', 'sensitive_attributions',
                            'objectives_class', 'preserve_sens_in_net', 'weight_decay', "cal_obj_plan", "global_seed",
                            "dropout", "MOEAs", "crossover_p", "mutation_p", "obj_is_logits", "is_ensemble"]
        parameters = {'start_time': start_time, 'learning_rate': learning_rate, 'batch_size': batch_size,
                      'n_hidden': n_hidden, 'n_output': n_output, 'epoches': epoches, 'muta_mu': muta_mu,
                      'muta_var': muta_var, 'NIND': popsize, 'MAXGEN': MAXGEN,
                      'logTras': logTras, 'verbose': verbose, 'drawing': drawing, 'dirName': dirName,
                      'dataname': dataname, 'sensitive_attributions': sensitive_attributions,
                      'logMetric': logMetric, 'objectives_class': objectives_class,
                      'preserve_sens_in_net': preserve_sens_in_net, 'seed_split_traintest': seed_split_traintest,
                      'run_id': run_id, "MOEAs": MOEAs, "crossover_p": crossover_p, "mutation_p": mutation_p,
                      'record_parameter': record_parameter, "is_ensemble": is_ensemble}

        print(parameters)
        run(parameters)
