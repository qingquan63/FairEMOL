# -*- coding: utf-8 -*-
import numpy as np
import FairEMOL as ea
import warnings
import time
# from FairEMOL.EvolutionaryCodes.Run_metric import Alg_Evaluation


def maxminnorm(array):
    maxcols = array.max(axis=0)
    mincols = array.min(axis=0)
    data_shape = array.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    t = np.empty((data_rows, data_cols))
    for i in range(data_cols):
        t[:, i] = (array[:, i] - mincols[i]) / (maxcols[i] - mincols[i])
    return t


class Algorithm:
    def __init__(self):
        self.name = 'Algorithm'
        self.problem = None
        self.population = None
        self.MAXGEN = None
        self.currentGen = None
        self.MAXTIME = None
        self.timeSlot = None
        self.passTime = None
        self.MAXEVALS = None
        self.evalsNum = None
        self.MAXSIZE = None
        self.logTras = None
        self.log = None
        self.verbose = None
        self.logits = {}
        self.Metric = {}
        self.PopObj = {}
        self.logMetric = None
        self.dirName = None
        self.all_objetives_valid = {}
        self.all_objetives_test = {}
        self.all_objetives_train = {}
        self.all_objetives_ensemble = {}
        self.is_ensemble = None

    def initialization(self):
        pass

    def run(self, pop):
        pass

    def logging(self, pop):
        pass

    def stat(self, pop):
        pass

    def terminated(self, pop):
        pass

    def finishing(self, pop):
        pass

    def check(self, pop):

        # 检测数据非法值
        if np.any(np.isnan(pop.ObjV)):
            warnings.warn(
                "Warning: Some elements of ObjV are NAN, please check the calculation of ObjV.(ObjV的部分元素为NAN，请检查目标函数的计算。)",
                RuntimeWarning)
        elif np.any(np.isinf(pop.ObjV)):
            warnings.warn(
                "Warning: Some elements of ObjV are Inf, please check the calculation of ObjV.(ObjV的部分元素为Inf，请检查目标函数的计算。)",
                RuntimeWarning)
        if pop.CV is not None:
            if np.any(np.isnan(pop.CV)):
                warnings.warn(
                    "Warning: Some elements of CV are NAN, please check the calculation of CV.(CV的部分元素为NAN，请检查CV的计算。)",
                    RuntimeWarning)
            elif np.any(np.isinf(pop.CV)):
                warnings.warn(
                    "Warning: Some elements of CV are Inf, please check the calculation of CV.(CV的部分元素为Inf，请检查CV的计算。)",
                    RuntimeWarning)

    def add_gen2info(self, pop, gen):

        self.all_objetives_valid[str(gen)] = pop.ObjV_valid
        self.all_objetives_test[str(gen)] = pop.ObjV_test
        self.all_objetives_train[str(gen)] = pop.ObjV_train
        if self.is_ensemble:
            self.all_objetives_ensemble[str(gen)] = pop.ObjV_ensemble
        else:
            self.all_objetives_ensemble = None

    def call_aimFunc(self, pop, gen=0, dirName=None, loss_type=-1):

        pop.Phen = pop.decoding()
        if self.problem is None:
            raise RuntimeError('error: problem has not been initialized. (算法模板中的问题对象未被初始化。)')
        self.problem.aimFunc(pop, gen=gen, dirName=dirName, loss_type=loss_type)  # 调用问题类的aimFunc()

        self.evalsNum = self.evalsNum + pop.sizes if self.evalsNum is not None else pop.sizes  # 更新评价次数

        if not isinstance(pop.ObjV, np.ndarray) or pop.ObjV.ndim != 2 or pop.ObjV.shape[0] != pop.sizes or \
                pop.ObjV.shape[1] != self.problem.M:
            raise RuntimeError('error: ObjV is illegal. (目标函数值矩阵ObjV的数据格式不合法，请检查目标函数的计算。)')
        if pop.CV is not None:
            if not isinstance(pop.CV, np.ndarray) or pop.CV.ndim != 2 or pop.CV.shape[0] != pop.sizes:
                raise RuntimeError('error: CV is illegal. (违反约束程度矩阵CV的数据格式不合法，请检查CV的计算。)')

    def train_nets(self, pop, Gen, epoch=1, iscal_metric=1, changeNets=0, problem=None, runtime=0):
        if self.problem is None:
            raise RuntimeError('error: problem has not been initialized. (算法模板中的问题对象未被初始化。)')
        if changeNets == 0:
            popnew = pop.copy()
            resin_train, resin_test, pop_logits_test = self.problem.train_nets(popnew, epoch)  # 调用问题类的aimFunc()
        else:
            resin_train, resin_test, pop_logits_test = self.problem.train_nets(pop, epoch)  # 调用问题类的aimFunc()
            for i in range(len(pop)):
                pop.logits[i, :] = pop_logits_test[i, :]

        ###### 计算 metric ######
        if iscal_metric == 1:
            popsize = len(pop)
            dataset_obj = self.problem.dataset_obj
            true_label = self.problem.test_label.tolist()
            Res_metrics = {}
            supported_tag = 'numerical-for-NN'
            all_possible = set(true_label)
            posi_calss = dataset_obj.get_positive_class_val(supported_tag)
            all_possible.remove(posi_calss)
            negative_calss = all_possible.pop()
            for i in range(popsize):
                pred_label = pop_logits_test[i, :]
                pred_label = [posi_calss if x >= 0.5 else negative_calss for x in pred_label]
                # (dataset_obj, problem, logits, predic_label, true_label, test_org, supported_tag):
                metric_res = Alg_Evaluation(dataset_obj, self.problem, pop_logits_test[i, :],
                                            pred_label, true_label, self.problem.test_org, supported_tag)
                Res_metrics[str(i)] = metric_res
                # print("Calculating metrics: " + str(i + 1) + " / " + str(popsize))

            ## 打印metrics
            nowmetric = Res_metrics
            if pop.ObjV is not None:
                [levels, _] = ea.ndsortDED(pop.ObjV, needLevel=1, CV=pop.CV,
                                           maxormins=self.problem.maxormins)  # 非支配分层
                levels = np.where(levels == 1)[0]
            else:
                levels = np.array(range(popsize))
            save_filename = '/fulltrain/fulltrain_inGen%d_metric.csv' % Gen
            all_sensitive_attributes = list(nowmetric[str(0)].keys())
            with open(self.dirName + save_filename, 'a+') as file:
                for sens in all_sensitive_attributes:
                    all_colums_name = list(nowmetric[str(0)][all_sensitive_attributes[0]].keys())
                    line = str(runtime) + ",is non-dom,sensitive attributes,"
                    # 写下metric的名字
                    for colum in all_colums_name:
                        line = line + colum + ','
                    file.write(line + "\n")

                    for idx in range(len(nowmetric)):
                        info = nowmetric[str(idx)]
                        if idx in levels:
                            line = "individual " + str(idx) + ",1," + sens
                        else:
                            line = "individual " + str(idx) + ",0," + sens
                        for colum in all_colums_name:
                            line = line + ',' + str(info[sens][colum])
                        file.write(line + "\n")
                    file.write("\n")
                file.close()

        ###### 打印在train、test的目标值 ######
        save_filename = '/fulltrain/fulltrain_inGen%d_trainObj.csv' % Gen
        with open(self.dirName + save_filename, 'a+') as file:
            for i in range(resin_train.shape[0]):
                line = ','.join(str(x) for x in resin_train[i, :]) + '\n'
                file.write(line)
            file.close()

        save_filename = '/fulltrain/fulltrain_inGen%d_testObj.csv' % Gen
        with open(self.dirName + save_filename, 'a+') as file:
            for i in range(resin_test.shape[0]):
                line = ','.join(str(x) for x in resin_test[i, :]) + '\n'
                file.write(line)
            file.close()

        save_filename = '/fulltrain/fulltrain_inGen%d_testlogits.csv' % Gen
        true_y = np.array(problem.test_y)
        with open(self.dirName + save_filename, 'a+') as file:
            logits = np.array(pop_logits_test)
            true_y = true_y.reshape(1, -1)
            line = ','.join(str(x) for x in true_y[0]) + '\n'
            file.write(line)
            for rows in range(logits.shape[0]):
                popobj = logits[rows, :]
                line = ','.join(str(x) for x in popobj) + '\n'
                file.write(line)
            file.close()

    def get_predy(self, logits):
        pred_label = logits
        pred_label[np.where(pred_label >= 0.5)] = 1
        pred_label[np.where(pred_label < 0.5)] = 0
        pred_label = pred_label.reshape(1, logits.shape[0] * logits.shape[1])
        pred_label = pred_label.reshape(1, -1)
        return pred_label

    def display(self):
        self.passTime += time.time() - self.timeSlot  # 更新用时记录，不计算display()的耗时
        headers = []
        widths = []
        values = []
        for key in self.log.keys():
            # 设置单元格宽度
            if key == 'gen':
                width = max(3, len(str(self.MAXGEN - 1)))  # 因为字符串'gen'长度为3，所以最小要设置长度为3
            elif key == 'eval':
                width = 8  # 因为字符串'eval'长度为4，所以最小要设置长度为4
            else:
                width = 13  # 预留13位显示长度，若数值过大，表格将无法对齐，此时若要让表格对齐，需要自定义算法模板重写该函数
            headers.append(key)
            widths.append(width)
            value = self.log[key][-1] if len(self.log[key]) != 0 else "-"
            if isinstance(value, float):
                values.append("%.5E" % value)  # 格式化浮点数，输出时只保留至小数点后5位
            else:
                values.append(value)
        if len(self.log['gen']) == 1:  # 打印表头
            header_regex = '|'.join(['{}'] * len(headers))
            header_str = header_regex.format(*[str(key).center(width) for key, width in zip(headers, widths)])
            print("=" * len(header_str))
            print(header_str)
            print("-" * len(header_str))
        if len(self.log['gen']) != 0:  # 打印表格最后一行
            value_regex = '|'.join(['{}'] * len(values))
            value_str = value_regex.format(*[str(value).center(width) for value, width in zip(values, widths)])
            print(value_str)
        self.timeSlot = time.time()  # 更新时间戳


class MoeaAlgorithm(Algorithm):  # 多目标优化算法模板父类

    def __init__(self, problem, population):
        super().__init__()  # 先调用父类构造函数
        self.problem = problem
        self.population = population
        self.logTras = 1  # 默认设置logTras的值为1
        self.verbose = True  # 默认设置verbose的值为True
        self.drawing = 1  # 默认设置drawing的值为1
        self.ax = None  # 存储动态图像
        self.logMetric = 0

    def initialization(self, is_ensemble=False):
        self.ax = None  # 初始化ax
        self.passTime = 0  # 初始化passTime
        self.log = None  # 初始化log
        self.currentGen = 0  # 初始为第0代
        self.evalsNum = 0  # 初始化评价次数为0
        self.log = {'gen': [], 'eval': []} if self.logTras != 0 else None  # 初始化log
        self.timeSlot = time.time()  # 开始计时
        self.is_ensemble = is_ensemble

    def logging(self, pop):
        self.passTime += time.time() - self.timeSlot  # 更新用时记录，不计算logging的耗时
        self.PopObj[str(self.currentGen)] = pop.ObjV

        if len(self.log['gen']) == 0:  # 初始化log的各个键值
            if self.problem.ReferObjV is not None:
                self.log['gd'] = []
                self.log['igd'] = []
            self.log['hv'] = []
            self.log['spacing'] = []
        self.log['gen'].append(self.currentGen)
        self.log['eval'].append(self.evalsNum)  # 记录评价次数
        [levels, _] = ea.ndsortDED(pop.ObjV, needLevel=1, CV=pop.CV, maxormins=self.problem.maxormins)  # 非支配分层
        NDSet = pop[np.where(levels == 1)[0]]  # 只保留种群中的非支配个体，形成一个非支配种群
        if self.problem.ReferObjV is not None:
            self.log['gd'].append(ea.indicator.GD(NDSet.ObjV, self.problem.ReferObjV))  # 计算GD指标
            self.log['igd'].append(ea.indicator.IGD(NDSet.ObjV, self.problem.ReferObjV))  # 计算IGD指标
            self.log['hv'].append(ea.indicator.HV(NDSet.ObjV, self.problem.ReferObjV))  # 计算HV指标
        else:
            pass
            # self.log['hv'].append(ea.indicator.HV(NDSet.ObjV))  # 计算HV指标
        # self.log['spacing'].append(ea.indicator.Spacing(NDSet.ObjV))  # 计算Spacing指标
        self.timeSlot = time.time()  # 更新时间戳

    def draw(self, pop, EndFlag=False):

        if not EndFlag:
            self.passTime += time.time() - self.timeSlot  # 更新用时记录，不计算画图的耗时
            # 绘制动画
            if self.drawing == 2:
                # 绘制目标空间动态图
                if pop.ObjV.shape[1] > 3:
                    # objs = maxminnorm(pop.ObjV)
                    objs = pop.ObjV
                else:
                    objs = pop.ObjV
                self.ax = ea.moeaplot(objs, 'objective values', False, self.ax, self.currentGen, gridFlag=True)
            elif self.drawing == 3:
                # 绘制决策空间动态图
                self.ax = ea.varplot(pop.Phen, 'decision variables', False, self.ax, self.currentGen, gridFlag=False)
            self.timeSlot = time.time()  # 更新时间戳
        else:
            # 绘制最终结果图
            if self.drawing != 0:
                if pop.ObjV.shape[1] == 2 or pop.ObjV.shape[1] == 3:
                    # ea.moeaplot(pop.ObjV, 'Pareto Front', saveFlag=True, gridFlag=True)
                    pass
                else:
                    ea.moeaplot(pop.ObjV, 'Value Path', saveFlag=True, gridFlag=False)

    def stat(self, pop):

        feasible = np.where(np.all(pop.CV <= 0, 1))[0] if pop.CV is not None else np.arange(pop.sizes)  # 找到满足约束条件的个体的下标
        if len(feasible) > 0:
            feasiblePop = pop[feasible]  # 获取满足约束条件的个体
            if self.logTras != 0 and self.currentGen % self.logTras == 0:
                self.logging(feasiblePop)  # 记录日志
                if self.verbose:
                    self.display()  # 打印日志
            self.draw(feasiblePop)  # 展示输出

    def terminated(self, pop):
        self.passTime += time.time() - self.timeSlot  # 更新耗时
        self.check(pop)  # 检查种群对象的关键属性是否有误
        self.stat(pop)  # 进行统计分析，更新进化记录器

        # 判断是否终止进化，由于代数是从0数起，因此在比较currentGen和MAXGEN时需要对currentGen加1
        if (self.MAXTIME is not None and self.passTime >= self.MAXTIME) or self.currentGen + 1 >= self.MAXGEN:
            self.timeSlot = time.time()  # 更新时间戳
            return True
        else:
            self.timeSlot = time.time()  # 更新时间戳
            self.currentGen += 1  # 进化代数+1
            return False

    def finishing(self, pop, globalNDSet=None):

        if globalNDSet is None:
            # 得到非支配种群
            [levels, _] = ea.ndsortDED(pop.ObjV, needLevel=1, CV=pop.CV, maxormins=self.problem.maxormins)  # 非支配分层
            NDSet = pop[np.where(levels == 1)[0]]  # 只保留种群中的非支配个体，形成一个非支配种群
            if NDSet.CV is not None:  # CV不为None说明有设置约束条件
                NDSet = NDSet[np.where(np.all(NDSet.CV <= 0, 1))[0]]  # 最后要彻底排除非可行解
        else:
            NDSet = globalNDSet
        if self.logTras != 0 and NDSet.sizes != 0 and (
                len(self.log['gen']) == 0 or self.log['gen'][-1] != self.currentGen):  # 补充记录日志和输出
            self.logging(NDSet)
            if self.verbose:
                self.display()
        self.passTime += time.time() - self.timeSlot  # 更新用时记录，因为已经要结束，因此不用再更新时间戳
        self.draw(NDSet, EndFlag=True)  # 显示最终结果图
        # 返回帕累托最优个体以及最后一代种群
        return [NDSet, pop]

    def get_metric(self, population):
        popsize = len(population)
        dataset_obj = self.problem.dataset_obj
        true_label = self.problem.test_label.tolist()
        Res_metrics = {}
        supported_tag = 'numerical-for-NN'
        all_possible = set(true_label)
        posi_calss = dataset_obj.get_positive_class_val(supported_tag)
        all_possible.remove(posi_calss)
        negative_calss = all_possible.pop()
        for i in range(popsize):
            indiv = population[i]
            pred_label = indiv.logits.reshape(-1).tolist()
            logits = pred_label.copy()
            pred_label = [posi_calss if x >= 0.5 else negative_calss for x in pred_label]
            metric_res = Alg_Evaluation(dataset_obj, self.problem, logits,
                                        pred_label, true_label, self.problem.test_org, supported_tag)
            Res_metrics[str(i)] = metric_res
        return Res_metrics


class SoeaAlgorithm(Algorithm):  # 单目标优化算法模板父类
    def __init__(self, problem, population):
        super().__init__()  # 先调用父类构造函数
        self.problem = problem
        self.population = population
        self.trappedValue = 0  # 默认设置trappedValue的值为0
        self.maxTrappedCount = 1000  # 默认设置maxTrappedCount的值为1000
        self.logTras = 1  # 默认设置logTras的值为1
        self.verbose = True  # 默认设置verbose的值为True
        self.drawing = 1  # 默认设置drawing的值为1
        # 以下为用户不需要设置的属性
        self.BestIndi = None  # 存储算法所找到的最优的个体
        self.trace = None  # 进化记录器
        self.trappedCount = None  # 定义trappedCount，在initialization()才对其进行初始化为0
        self.ax = None  # 存储动态图像

    def initialization(self):
        self.ax = None  # 初始化ax
        self.passTime = 0  # 初始化passTime
        self.trappedCount = 0  # 初始化“进化停滞”计数器
        self.currentGen = 0  # 初始为第0代
        self.evalsNum = 0  # 初始化评价次数为0
        self.BestIndi = ea.Population(None, None, 0)  # 初始化BestIndi为空的种群对象
        self.log = {'gen': [], 'eval': []} if self.logTras != 0 else None  # 初始化log
        self.trace = {'f_best': [], 'f_avg': []}  # 重置trace
        # 开始计时
        self.timeSlot = time.time()

    def logging(self, pop):
        self.passTime += time.time() - self.timeSlot  # 更新用时记录，不计算logging的耗时
        if len(self.log['gen']) == 0:  # 初始化log的各个键值
            self.log['f_opt'] = []
            self.log['f_max'] = []
            self.log['f_avg'] = []
            self.log['f_min'] = []
            self.log['f_std'] = []
        self.log['gen'].append(self.currentGen)
        self.log['eval'].append(self.evalsNum)  # 记录评价次数
        self.log['f_opt'].append(self.BestIndi.ObjV[0][0])  # 记录算法所找到的最优个体的目标函数值
        self.log['f_max'].append(np.max(pop.ObjV))
        self.log['f_avg'].append(np.mean(pop.ObjV))
        self.log['f_min'].append(np.min(pop.ObjV))
        self.log['f_std'].append(np.std(pop.ObjV))
        self.timeSlot = time.time()  # 更新时间戳

    def draw(self, pop, EndFlag=False):
        if not EndFlag:
            self.passTime += time.time() - self.timeSlot  # 更新用时记录，不计算画图的耗时
            # 绘制动画
            if self.drawing == 2:
                metric = np.array(self.trace['f_best']).reshape(-1, 1)
                self.ax = ea.soeaplot(metric, Label='Objective Value', saveFlag=False, ax=self.ax, gen=self.currentGen,
                                      gridFlag=False)  # 绘制动态图
            elif self.drawing == 3:
                self.ax = ea.varplot(pop.Phen, Label='decision variables', saveFlag=False, ax=self.ax,
                                     gen=self.currentGen, gridFlag=False)
            self.timeSlot = time.time()  # 更新时间戳
        else:
            # 绘制最终结果图
            if self.drawing != 0:
                metric = np.vstack(
                    [self.trace['f_avg'], self.trace['f_best']]).T
                ea.trcplot(metric, [['种群个体平均目标函数值', '种群最优个体目标函数值']], xlabels=[['Number of Generation']],
                           ylabels=[['Value']], gridFlags=[[False]])

    def stat(self, pop):
        # 进行进化记录
        feasible = np.where(np.all(pop.CV <= 0, 1))[0] if pop.CV is not None else np.arange(pop.sizes)  # 找到满足约束条件的个体的下标
        if len(feasible) > 0:
            feasiblePop = pop[feasible]
            bestIndi = feasiblePop[np.argmax(feasiblePop.FitnV)]  # 获取最优个体
            if self.BestIndi.sizes == 0:
                self.BestIndi = bestIndi  # 初始化global best individual
            else:
                delta = (
                                self.BestIndi.ObjV - bestIndi.ObjV) * self.problem.maxormins if self.problem.maxormins is not None else self.BestIndi.ObjV - bestIndi.ObjV
                # 更新“进化停滞”计数器
                self.trappedCount += 1 if np.abs(delta) < self.trappedValue else 0
                # 更新global best individual
                if delta > 0:
                    self.BestIndi = bestIndi
            # 更新trace
            self.trace['f_best'].append(bestIndi.ObjV[0][0])
            self.trace['f_avg'].append(np.mean(feasiblePop.ObjV))
            if self.logTras != 0 and self.currentGen % self.logTras == 0:
                self.logging(feasiblePop)  # 记录日志
                if self.verbose:
                    self.display()  # 打印日志
            self.draw(feasiblePop)  # 展示输出

    def terminated(self, pop):
        self.check(pop)  # 检查种群对象的关键属性是否有误
        self.stat(pop)  # 分析记录当代种群的数据
        self.passTime += time.time() - self.timeSlot  # 更新耗时
        self.timeSlot = time.time()  # 更新时间戳
        # 判断是否终止进化，由于代数是从0数起，因此在比较currentGen和MAXGEN时需要对currentGen加1
        if (
                self.MAXTIME is not None and self.passTime >= self.MAXTIME) or self.currentGen + 1 >= self.MAXGEN or self.trappedCount >= self.maxTrappedCount:
            return True
        else:
            self.currentGen += 1  # 进化代数+1
            return False

    def finishing(self, pop):
        feasible = np.where(np.all(pop.CV <= 0, 1))[0] if pop.CV is not None else np.arange(pop.sizes)  # 找到满足约束条件的个体的下标
        if len(feasible) > 0:
            feasiblePop = pop[feasible]
            if self.logTras != 0 and (len(self.log['gen']) == 0 or self.log['gen'][-1] != self.currentGen):  # 补充记录日志和输出
                self.logging(feasiblePop)
                if self.verbose:
                    self.display()
        self.passTime += time.time() - self.timeSlot  # 更新用时记录，因为已经要结束，因此不用再更新时间戳
        self.draw(pop, EndFlag=True)  # 显示最终结果图
        # 返回最优个体以及最后一代种群
        return [self.BestIndi, pop]

    def get_metric(self, problem):
        pass
