import numpy as np


class GroupInfo:
    def __init__(self, group_name, group_fairness, individual_fairness, truelabel, pred_label, logits, total_num):
        self.group_name = group_name
        self.group_fairness = group_fairness
        self.pred_label = pred_label
        self.truelabel = truelabel
        self.logits = logits
        self.count = logits.shape[0]*logits.shape[1]
        self.accuracy = np.sum(pred_label == truelabel)
        self.individual_fairness = individual_fairness
        self.total_num = total_num
        self.count = pred_label.shape[0]*pred_label.shape[1]
        self.accuracy = np.sum(pred_label == truelabel)/self.count

    def getAccuracy(self):
        return self.accuracy

    def getGroupName(self):
        return self.group_name

    def getCount(self):
        return self.count

    def getAccuracy(self):
        return self.accuracy

    def getIndividualFairness(self):
        return self.individual_fairness

    def getGroupFairness(self):
        return self.group_fairness


class GroupsInfo:
    def __init__(self, groups):
        self.num_group = len(groups)
        self.group_name = []
        self.group_fairness = []
        self.pred_label = []
        self.truelabel = []
        self.logits = []
        self.count = []
        self.accuracy = []
        self.individual_fairness = []
        for group in groups:
            self.group_name.append(group.getGroupName())
            self.individual_fairness.append(group.getIndividualFairness())
            self.group_fairness.append(group.getGroupFairness())
            self.accuracy.append(group.getAccuracy())
            self.count.append(group.getCount())

    def getAccuracy(self):
        return self.accuracy

    def getGroupName(self):
        return self.group_name

    def getCount(self):
        return self.count

    def getAccuracy(self):
        return self.accuracy

    def getIndividualFairness(self):
        return self.individual_fairness

    def getGroupFairness(self):
        return self.group_fairness


