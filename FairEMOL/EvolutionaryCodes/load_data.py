from FairEMOL.EvolutionaryCodes.data.objects.list import DATASETS, get_dataset_names
from FairEMOL.EvolutionaryCodes.data.objects.ProcessedData import ProcessedData
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np
from itertools import product



def make_class_attr_num(dataframe, positive_val):
    dataframe = dataframe.replace({positive_val: 1})
    # dataframe = dataframe.replace("[^1]", 0, regex=True)
    dataframe[dataframe != 1] = 0
    return dataframe


supported_tags = ["original", "numerical", "numerical-binsensitive",
                  "categorical-binsensitive", "numerical-for-NN", "original_info"]


def get_smaller(data_org, data, label, rate):
    Sss = StratifiedShuffleSplit(n_splits=2, test_size=rate, random_state=0)
    Plan1, Plan2 = Sss.split(data_org, label)

    org_data = data_org.iloc[Plan1[1]]
    data = data.loc[Plan1[1]]
    data_label = label.loc[Plan1[1]]

    org_data.reset_index(inplace=True, drop=True)
    data.reset_index(inplace=True, drop=True)
    data_label.reset_index(inplace=True, drop=True)

    return org_data, data, data_label


def analyze_data(ana_data, alldata, dataset, sensitive_attributions):
    attribution = alldata.columns
    # sensitive_attributions = dataset.get_sensitive_attributes()
    group_dict = {}
    sens_idxs_name = []
    test_num = ana_data.shape[0]
    sens_idxs = {}
    for sens in sensitive_attributions:
        temp = []
        for attr in attribution:
            temp1 = sens + '_'
            if temp1 in attr:
                temp.append(attr)
                sens_idxs.update({attr: np.array(np.where(ana_data[attr])).reshape(1, -1)})
                sens_idxs_name.append(attr)

        group_dict.update({sens: temp})

    group_attr = []
    for sens in sensitive_attributions:
        group_attr.append(group_dict[sens])

    for item in product(*eval(str(group_attr))):
        group = item
        flag = np.ones([1, test_num]) == np.ones([1, test_num])
        for g in group:
            flag = flag & ana_data[g]
        g_num = np.sum(flag)
        name = "+".join(str(x) for x in group)
        if g_num != 0:
            # sens_idxs_name.append(name)
            g_idx = np.array(np.where(flag)).reshape([1, g_num])
            sens_idxs.update({name: g_idx})
            if name not in sens_idxs_name:
                sens_idxs_name.append(name)
        else:
            sens_idxs.update({name: None})
            if name not in sens_idxs_name:
                sens_idxs_name.append(name)

    return sens_idxs_name, sens_idxs, group_dict


def load_data(dataname, datatype="numerical-for-NN", preserve_sens_in_net=0, sensitive_attributions=None, is_ensemble=False):
    available_datasets = get_dataset_names()
    if dataname not in available_datasets:
        print('There is no dataset named', dataname)
        return None

    if not is_ensemble:
        for dataset in DATASETS:
            save_csv = 0
            is_smaller = 0
            if dataset.dataset_name == dataname:
                processed_dataset = ProcessedData(dataset)
                data = processed_dataset.create_train_test_splits()
                data = data[datatype]

                label_name = dataset.get_class_attribute()
                data.reset_index(inplace=True, drop=True)
                data_x = data.drop(columns=label_name)
                data_label = data[label_name]

                temp = data_label.copy()
                temp.loc[(data_label != 1)] = dataset.get_negative_class_val("")
                temp.loc[(data_label == 1)] = dataset.get_positive_class_val("")
                data_label = temp
                org_data = processed_dataset.get_orig_data()

                # train+validation = 80%      test = 20%
                sss1 = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=20210811)
                plan1, plan2 = sss1.split(data_x, data_label)

                test_org = org_data.iloc[plan1[1]]
                test_data = data_x.loc[plan1[1]]
                test_label = data_label.loc[plan1[1]]

                trainvaliddata_org = org_data.iloc[plan1[0]]
                trainvaliddata = data_x.loc[plan1[0]]
                trainvalidlabel = data_label.loc[plan1[0]]
                trainvaliddata.reset_index(inplace=True, drop=True)
                trainvalidlabel.reset_index(inplace=True, drop=True)
                trainvaliddata_org.reset_index(inplace=True, drop=True)

                sss2 = StratifiedShuffleSplit(n_splits=2, test_size=0.25, random_state=20210811)
                plan3, plan4 = sss2.split(trainvaliddata, trainvalidlabel)

                train_org = trainvaliddata_org.loc[plan3[0]]
                train_data = trainvaliddata.loc[plan3[0]]
                train_label = trainvalidlabel.loc[plan3[0]]

                valid_org = trainvaliddata_org.loc[plan3[1]]
                valid_data = trainvaliddata.loc[plan3[1]]
                valid_label = trainvalidlabel.loc[plan3[1]]

                if dataset.dataset_name == 'adult':
                    try:
                        train_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/AdultData/adult1/train_org.csv', index_col=0)
                        train_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/AdultData/adult1/train_data.csv', index_col=0)
                        train_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/AdultData/adult1/train_label.csv', index_col=0)
                        valid_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/AdultData/adult1/valid_data.csv', index_col=0)
                        valid_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/AdultData/adult1/valid_label.csv', index_col=0)
                        valid_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/AdultData/adult1/valid_org.csv', index_col=0)
                        test_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/AdultData/adult1/test_data.csv', index_col=0)
                        test_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/AdultData/adult1/test_label.csv', index_col=0)
                        test_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/AdultData/adult1/test_org.csv', index_col=0)

                        train_label = train_label['income-per-year']
                        valid_label = valid_label['income-per-year']
                        test_label = test_label['income-per-year']
                    except:
                        train_org = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/AdultData/adult1/train_org.csv', index_col=0)
                        train_data = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/AdultData/adult1/train_data.csv', index_col=0)
                        train_label = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/AdultData/adult1/train_label.csv', index_col=0)
                        valid_data = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/AdultData/adult1/valid_data.csv', index_col=0)
                        valid_label = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/AdultData/adult1/valid_label.csv', index_col=0)
                        valid_org = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/AdultData/adult1/valid_org.csv', index_col=0)
                        test_data = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/AdultData/adult1/test_data.csv', index_col=0)
                        test_label = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/AdultData/adult1/test_label.csv', index_col=0)
                        test_org = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/AdultData/adult1/test_org.csv', index_col=0)

                        train_label = train_label['income-per-year']
                        valid_label = valid_label['income-per-year']
                        test_label = test_label['income-per-year']

                if dataset.dataset_name == 'propublica-recidivism':
                    try:
                        train_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/COMPAS/compas/train_org.csv', index_col=0)
                        train_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/COMPAS/compas/train_data.csv', index_col=0)
                        train_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/COMPAS/compas/train_label.csv', index_col=0)
                        valid_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/COMPAS/compas/valid_data.csv', index_col=0)
                        valid_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/COMPAS/compas/valid_label.csv', index_col=0)
                        valid_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/COMPAS/compas/valid_org.csv', index_col=0)
                        test_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/COMPAS/compas/test_data.csv', index_col=0)
                        test_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/COMPAS/compas/test_label.csv', index_col=0)
                        test_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/COMPAS/compas/test_org.csv', index_col=0)

                        train_label = train_label['two_year_recid']
                        valid_label = valid_label['two_year_recid']
                        test_label = test_label['two_year_recid']
                    except:
                        train_org = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/COMPAS/compas/train_org.csv', index_col=0)
                        train_data = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/COMPAS/compas/train_data.csv', index_col=0)
                        train_label = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/COMPAS/compas/train_label.csv', index_col=0)
                        valid_data = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/COMPAS/compas/valid_data.csv', index_col=0)
                        valid_label = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/COMPAS/compas/valid_label.csv', index_col=0)
                        valid_org = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/COMPAS/compas/valid_org.csv', index_col=0)
                        test_data = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/COMPAS/compas/test_data.csv', index_col=0)
                        test_label = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/COMPAS/compas/test_label.csv', index_col=0)
                        test_org = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/COMPAS/compas/test_org.csv', index_col=0)

                        train_label = train_label['two_year_recid']
                        valid_label = valid_label['two_year_recid']
                        test_label = test_label['two_year_recid']

                if dataset.dataset_name == 'german':
                    try:
                        train_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/German/german/train_org.csv', index_col=0)
                        train_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/German/german/train_data.csv', index_col=0)
                        train_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/German/german/train_label.csv', index_col=0)
                        valid_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/German/german/valid_data.csv', index_col=0)
                        valid_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/German/german/valid_label.csv', index_col=0)
                        valid_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/German/german/valid_org.csv', index_col=0)
                        test_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/German/german/test_data.csv', index_col=0)
                        test_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/German/german/test_label.csv', index_col=0)
                        test_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/German/german/test_org.csv', index_col=0)

                        train_label = train_label['credit']
                        valid_label = valid_label['credit']
                        test_label = test_label['credit']
                    except:
                        train_org = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/German/german/train_org.csv', index_col=0)
                        train_data = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/German/german/train_data.csv', index_col=0)
                        train_label = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/German/german/train_label.csv', index_col=0)
                        valid_data = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/German/german/valid_data.csv', index_col=0)
                        valid_label = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/German/german/valid_label.csv', index_col=0)
                        valid_org = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/German/german/valid_org.csv', index_col=0)
                        test_data = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/German/german/test_data.csv', index_col=0)
                        test_label = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/German/german/test_label.csv', index_col=0)
                        test_org = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/German/german/test_org.csv', index_col=0)

                        train_label = train_label['credit']
                        valid_label = valid_label['credit']
                        test_label = test_label['credit']

                if dataset.dataset_name == 'default':
                    try:
                        train_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/Default/default/train_org.csv', index_col=0)
                        train_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/Default/default/train_data.csv', index_col=0)
                        train_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/Default/default/train_label.csv', index_col=0)
                        valid_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/Default/default/valid_data.csv', index_col=0)
                        valid_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/Default/default/valid_label.csv', index_col=0)
                        valid_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/Default/default/valid_org.csv', index_col=0)
                        test_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/Default/default/test_data.csv', index_col=0)
                        test_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/Default/default/test_label.csv', index_col=0)
                        test_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/Default/default/test_org.csv', index_col=0)

                        train_label = train_label['default_payment_next_month']
                        valid_label = valid_label['default_payment_next_month']
                        test_label = test_label['default_payment_next_month']
                    except:
                        train_org = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/Default/default/train_org.csv', index_col=0)
                        train_data = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/Default/default/train_data.csv', index_col=0)
                        train_label = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/Default/default/train_label.csv', index_col=0)
                        valid_data = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/Default/default/valid_data.csv', index_col=0)
                        valid_label = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/Default/default/valid_label.csv', index_col=0)
                        valid_org = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/Default/default/valid_org.csv', index_col=0)
                        test_data = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/Default/default/test_data.csv', index_col=0)
                        test_label = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/Default/default/test_label.csv', index_col=0)
                        test_org = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/Default/default/test_org.csv', index_col=0)

                        train_label = train_label['default_payment_next_month']
                        valid_label = valid_label['default_payment_next_month']
                        test_label = test_label['default_payment_next_month']

                if dataset.dataset_name == 'LSAT':
                    try:
                        train_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/LSAT/lsat/train_org.csv', index_col=0)
                        train_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/LSAT/lsat/train_data.csv', index_col=0)
                        train_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/LSAT/lsat/train_label.csv', index_col=0)
                        valid_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/LSAT/lsat/valid_data.csv', index_col=0)
                        valid_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/LSAT/lsat/valid_label.csv', index_col=0)
                        valid_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/LSAT/lsat/valid_org.csv', index_col=0)
                        test_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/LSAT/lsat/test_data.csv', index_col=0)
                        test_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/LSAT/lsat/test_label.csv', index_col=0)
                        test_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/LSAT/lsat/test_org.csv', index_col=0)

                        train_label = train_label['first_pf']
                        valid_label = valid_label['first_pf']
                        test_label = test_label['first_pf']
                    except:
                        train_org = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/LSAT/lsat/train_org.csv', index_col=0)
                        train_data = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/LSAT/lsat/train_data.csv', index_col=0)
                        train_label = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/LSAT/lsat/train_label.csv', index_col=0)
                        valid_data = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/LSAT/lsat/valid_data.csv', index_col=0)
                        valid_label = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/LSAT/lsat/valid_label.csv', index_col=0)
                        valid_org = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/LSAT/lsat/valid_org.csv', index_col=0)
                        test_data = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/LSAT/lsat/test_data.csv', index_col=0)
                        test_label = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/LSAT/lsat/test_label.csv', index_col=0)
                        test_org = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/LSAT/lsat/test_org.csv', index_col=0)

                        train_label = train_label['first_pf']
                        valid_label = valid_label['first_pf']
                        test_label = test_label['first_pf']

                if dataset.dataset_name == 'bank':
                    try:
                        train_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/Bank/bank/train_org.csv', index_col=0)
                        train_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/Bank/bank/train_data.csv', index_col=0)
                        train_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/Bank/bank/train_label.csv', index_col=0)
                        valid_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/Bank/bank/valid_data.csv', index_col=0)
                        valid_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/Bank/bank/valid_label.csv', index_col=0)
                        valid_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/Bank/bank/valid_org.csv', index_col=0)
                        test_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/Bank/bank/test_data.csv', index_col=0)
                        test_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/Bank/bank/test_label.csv', index_col=0)
                        test_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/Bank/bank/test_org.csv', index_col=0)

                        train_label = train_label['y']
                        valid_label = valid_label['y']
                        test_label = test_label['y']
                    except:
                        train_org = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/Bank/bank/train_org.csv', index_col=0)
                        train_data = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/Bank/bank/train_data.csv', index_col=0)
                        train_label = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/Bank/bank/train_label.csv', index_col=0)
                        valid_data = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/Bank/bank/valid_data.csv', index_col=0)
                        valid_label = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/Bank/bank/valid_label.csv', index_col=0)
                        valid_org = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/Bank/bank/valid_org.csv', index_col=0)
                        test_data = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/Bank/bank/test_data.csv', index_col=0)
                        test_label = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/Bank/bank/test_label.csv', index_col=0)
                        test_org = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/Bank/bank/test_org.csv', index_col=0)

                        train_label = train_label['y']
                        valid_label = valid_label['y']
                        test_label = test_label['y']

                if dataset.dataset_name == 'dutch':
                    try:
                        train_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/Dutch/dutch/train_org.csv', index_col=0)
                        train_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/Dutch/dutch/train_data.csv', index_col=0)
                        train_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/Dutch/dutch/train_label.csv', index_col=0)
                        valid_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/Dutch/dutch/valid_data.csv', index_col=0)
                        valid_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/Dutch/dutch/valid_label.csv', index_col=0)
                        valid_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/Dutch/dutch/valid_org.csv', index_col=0)
                        test_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/Dutch/dutch/test_data.csv', index_col=0)
                        test_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/Dutch/dutch/test_label.csv', index_col=0)
                        test_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/Dutch/dutch/test_org.csv', index_col=0)

                        train_label = train_label['occupation']
                        valid_label = valid_label['occupation']
                        test_label = test_label['occupation']
                    except:
                        train_org = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/Dutch/dutch/train_org.csv', index_col=0)
                        train_data = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/Dutch/dutch/train_data.csv', index_col=0)
                        train_label = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/Dutch/dutch/train_label.csv', index_col=0)
                        valid_data = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/Dutch/dutch/valid_data.csv', index_col=0)
                        valid_label = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/Dutch/dutch/valid_label.csv', index_col=0)
                        valid_org = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/Dutch/dutch/valid_org.csv', index_col=0)
                        test_data = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/Dutch/dutch/test_data.csv', index_col=0)
                        test_label = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/Dutch/dutch/test_label.csv', index_col=0)
                        test_org = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/Dutch/dutch/test_org.csv', index_col=0)

                        train_label = train_label['occupation']
                        valid_label = valid_label['occupation']
                        test_label = test_label['occupation']


                if dataset.dataset_name == 'student_mat':
                    try:
                        train_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/Student_mat/student_mat/train_org.csv', index_col=0)
                        train_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/Student_mat/student_mat/train_data.csv', index_col=0)
                        train_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/Student_mat/student_mat/train_label.csv', index_col=0)
                        valid_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/Student_mat/student_mat/valid_data.csv', index_col=0)
                        valid_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/Student_mat/student_mat/valid_label.csv', index_col=0)
                        valid_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/Student_mat/student_mat/valid_org.csv', index_col=0)
                        test_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/Student_mat/student_mat/test_data.csv', index_col=0)
                        test_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/Student_mat/student_mat/test_label.csv', index_col=0)
                        test_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/Student_mat/student_mat/test_org.csv', index_col=0)

                        train_label = train_label['G3']
                        valid_label = valid_label['G3']
                        test_label = test_label['G3']
                    except:
                        train_org = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/Student_mat/student_mat/train_org.csv', index_col=0)
                        train_data = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/Student_mat/student_mat/train_data.csv', index_col=0)
                        train_label = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/Student_mat/student_mat/train_label.csv', index_col=0)
                        valid_data = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/Student_mat/student_mat/valid_data.csv', index_col=0)
                        valid_label = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/Student_mat/student_mat/valid_label.csv', index_col=0)
                        valid_org = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/Student_mat/student_mat/valid_org.csv', index_col=0)
                        test_data = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/Student_mat/student_mat/est_data.csv', index_col=0)
                        test_label = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/Student_mat/student_mat/test_label.csv', index_col=0)
                        test_org = pd.read_csv('EvolutionaryCodes/FairEMOL/EvolutionaryCodes/data/Student_mat/student_mat/test_org.csv', index_col=0)

                        train_label = train_label['G3']
                        valid_label = valid_label['G3']
                        test_label = test_label['G3']

                train_org.reset_index(inplace=True, drop=True)
                train_data.reset_index(inplace=True, drop=True)
                train_label.reset_index(inplace=True, drop=True)
                valid_data.reset_index(inplace=True, drop=True)
                valid_label.reset_index(inplace=True, drop=True)
                valid_org.reset_index(inplace=True, drop=True)
                test_data.reset_index(inplace=True, drop=True)
                test_label.reset_index(inplace=True, drop=True)
                test_org.reset_index(inplace=True, drop=True)

                if is_smaller == 1:
                    preserve = 1000
                    n_data = train_data.shape[0]
                    rate = np.min([preserve / n_data, 1])
                    if rate < 1:
                        train_org, train_data, train_label = get_smaller(train_org, train_data, train_label, rate)
                        valid_org, valid_data, valid_label = get_smaller(valid_org, valid_data, valid_label, rate)
                        test_org, test_data, test_label = get_smaller(test_org, test_data, test_label, rate)

                save_name = 'dataset_name'
                save_folder = 'dataset_name'
                if save_csv == 1:
                    test_org.to_csv('FairEMOL/EvolutionaryCodes/data/{}/{}/test_org.csv'.format(save_folder, save_name), index=True, header=True)
                    test_data.to_csv('FairEMOL/EvolutionaryCodes/data/{}/{}/test_data.csv'.format(save_folder, save_name), index=True, header=True)
                    test_label.to_csv('FairEMOL/EvolutionaryCodes/data/{}/{}/test_label.csv'.format(save_folder, save_name), index=True, header=True)

                    train_org.to_csv('FairEMOL/EvolutionaryCodes/data/{}/{}/train_org.csv'.format(save_folder, save_name), index=True, header=True)
                    train_data.to_csv('FairEMOL/EvolutionaryCodes/data/{}/{}/train_data.csv'.format(save_folder, save_name), index=True, header=True)
                    train_label.to_csv('FairEMOL/EvolutionaryCodes/data/{}/{}/train_label.csv'.format(save_folder, save_name), index=True, header=True)

                    valid_data.to_csv('FairEMOL/EvolutionaryCodes/data/{}/{}/valid_data.csv'.format(save_folder, save_name), index=True, header=True)
                    valid_label.to_csv('FairEMOL/EvolutionaryCodes/data/{}/{}/valid_label.csv'.format(save_folder, save_name), index=True, header=True)
                    valid_org.to_csv('FairEMOL/EvolutionaryCodes/data/{}/{}/valid_org.csv'.format(save_folder, save_name), index=True, header=True)

                # 是否删除sensitive attribution
                if preserve_sens_in_net == 1:
                    newtrain_data = train_data.copy()
                    newtest_data = test_data.copy()
                    newvalid_data = valid_data.copy()
                else:
                    attribution = train_data.columns
                    sens_dict = []
                    for sens in sensitive_attributions:
                        for attr in attribution:
                            temp = sens + '_'
                            if temp in attr:
                                sens_dict.append(attr)
                    newtrain_data = train_data.copy()
                    newtest_data = test_data.copy()
                    newvalid_data = valid_data.copy()
                    newdata_x = data_x.copy()
                    newtrain_data.drop(columns=sens_dict, inplace=True)
                    newtest_data.drop(columns=sens_dict, inplace=True)
                    newvalid_data.drop(columns=sens_dict, inplace=True)
                    newdata_x.drop(columns=sens_dict, inplace=True)

                # Normalization
                normalize = StandardScaler()

                # Fitting only on training data
                normalize.fit(newdata_x)
                train_data_norm = normalize.transform(newtrain_data)

                test_data_norm = normalize.transform(newtest_data)

                valid_data_norm = normalize.transform(newvalid_data)

                train_y = make_class_attr_num(train_label.copy(), dataset.get_positive_class_val(""))
                test_y = make_class_attr_num(test_label.copy(), dataset.get_positive_class_val(""))
                valid_y = make_class_attr_num(valid_label.copy(), dataset.get_positive_class_val(""))

                DATA_names = ['train_data', 'train_data_norm', 'train_label', 'train_y', 'train_org',
                              'valid_data', 'valid_data_norm', 'valid_label', 'valid_y', 'valid_org',
                              'test_data', 'test_data_norm', 'test_label', 'test_y', 'test_org'
                              'org_data', 'positive_class', 'positive_class_name',
                              'Groups_info', 'privileged_class_names', 'sens_attrs']
                DATA = dict((k, []) for k in DATA_names)

                DATA['train_data'] = train_data
                DATA['train_data_norm'] = train_data_norm
                DATA['train_label'] = train_label
                DATA['train_y'] = train_y.astype('int')

                DATA['valid_data'] = valid_data
                DATA['valid_data_norm'] = valid_data_norm
                DATA['valid_label'] = valid_label
                DATA['valid_y'] = valid_y.astype('int')

                DATA['test_data'] = test_data
                DATA['test_data_norm'] = test_data_norm
                DATA['test_label'] = test_label
                DATA['test_y'] = test_y.astype('int')

                DATA['positive_class'] = dataset.get_positive_class_val("")
                DATA['privileged_class_names'] = dataset.privileged_class_names

                DATA['org_data'] = org_data
                DATA['train_org'] = train_org
                DATA['test_org'] = test_org
                DATA['valid_org'] = valid_org

                DATA['positive_class_name'] = dataset.get_class_attribute()
                DATA['sens_attrs'] = dataset.get_sensitive_attributes()

                data_obj = dataset

                sens_idxs_name_train, sens_idxs_train, group_dict_train = analyze_data(train_data, data, dataset, sensitive_attributions)
                sens_idxs_name_valid, sens_idxs_valid, group_dict_valid = analyze_data(valid_data, data, dataset, sensitive_attributions)
                sens_idxs_name_test, sens_idxs_test, group_dict_test = analyze_data(test_data, data, dataset, sensitive_attributions)

                Groups_name = ['sens_idxs_train', 'sens_idxs_name_train', 'group_dict_train',
                              'sens_idxs_valid', 'sens_idxs_name_valid',  'group_dict_valid',
                              'sens_idxs_test', 'sens_idxs_name_test', 'group_dict_test']
                Groups_info = dict((k, []) for k in Groups_name)
                Groups_info['sens_idxs_train'] = sens_idxs_train
                Groups_info['sens_idxs_name_train'] = sens_idxs_name_train
                Groups_info['group_dict_train'] = group_dict_train
                Groups_info['sens_idxs_valid'] = sens_idxs_valid
                Groups_info['sens_idxs_name_valid'] = sens_idxs_name_valid
                Groups_info['group_dict_valid'] = group_dict_valid
                Groups_info['sens_idxs_test'] = sens_idxs_test
                Groups_info['sens_idxs_name_test'] = sens_idxs_name_test
                Groups_info['group_dict_test'] = group_dict_test

                DATA['Groups_info'] = Groups_info
    else:
        for dataset in DATASETS:
            save_csv = 0
            is_smaller = 0
            read_data = 1
            if dataset.dataset_name == dataname:
                processed_dataset = ProcessedData(dataset)
                data = processed_dataset.create_train_test_splits()
                data = data[datatype]

                label_name = dataset.get_class_attribute()
                data.reset_index(inplace=True, drop=True)
                data_x = data.drop(columns=label_name)
                data_label = data[label_name]

                temp = data_label.copy()
                temp.loc[(data_label != 1)] = dataset.get_negative_class_val("")
                temp.loc[(data_label == 1)] = dataset.get_positive_class_val("")
                data_label = temp
                org_data = processed_dataset.get_orig_data()

                sss1 = StratifiedShuffleSplit(n_splits=2, test_size=0.25, random_state=20210811)
                plan1, plan2 = sss1.split(data_x, data_label)

                test_org = org_data.iloc[plan1[1]]
                test_data = data_x.loc[plan1[1]]
                test_label = data_label.loc[plan1[1]]

                trainvalid_ensemble_data_org = org_data.iloc[plan1[0]]
                trainvalid_ensemble_data = data_x.loc[plan1[0]]
                trainvalid_ensemble_label = data_label.loc[plan1[0]]
                trainvalid_ensemble_data.reset_index(inplace=True, drop=True)
                trainvalid_ensemble_label.reset_index(inplace=True, drop=True)
                trainvalid_ensemble_data_org.reset_index(inplace=True, drop=True)

                # train+validation = 62.5%              ensemble = 12.5%
                sss2 = StratifiedShuffleSplit(n_splits=2, test_size=0.166667, random_state=20210811)
                plan3, plan4 = sss2.split(trainvalid_ensemble_data, trainvalid_ensemble_label)

                ensemble_org = trainvalid_ensemble_data_org.loc[plan3[1]]
                ensemble_data = trainvalid_ensemble_data.loc[plan3[1]]
                ensemble_label = trainvalid_ensemble_label.loc[plan3[1]]

                trainvalid_org = trainvalid_ensemble_data_org.loc[plan3[0]]
                trainvalid_data = trainvalid_ensemble_data.loc[plan3[0]]
                trainvalid_label = trainvalid_ensemble_label.loc[plan3[0]]
                trainvalid_org.reset_index(inplace=True, drop=True)
                trainvalid_data.reset_index(inplace=True, drop=True)
                trainvalid_label.reset_index(inplace=True, drop=True)

                sss3 = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=20210811)
                plan5, plan6 = sss3.split(trainvalid_data, trainvalid_label)

                train_org = trainvalid_org.loc[plan5[0]]
                train_data = trainvalid_data.loc[plan5[0]]
                train_label = trainvalid_label.loc[plan5[0]]

                valid_org = trainvalid_org.loc[plan5[1]]
                valid_data = trainvalid_data.loc[plan5[1]]
                valid_label = trainvalid_label.loc[plan5[1]]

                if read_data == 1:
                    if dataset.dataset_name == 'adult':
                        train_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Adult/adult1/train_org.csv', index_col=0)
                        train_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Adult/adult1/train_data.csv', index_col=0)
                        train_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Adult/adult1/train_label.csv', index_col=0)

                        valid_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Adult/adult1/valid_data.csv', index_col=0)
                        valid_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Adult/adult1/valid_label.csv', index_col=0)
                        valid_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Adult/adult1/valid_org.csv', index_col=0)

                        ensemble_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Adult/adult1/ensemble_data.csv', index_col=0)
                        ensemble_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Adult/adult1/ensemble_label.csv', index_col=0)
                        ensemble_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Adult/adult1/ensemble_org.csv', index_col=0)

                        test_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Adult/adult1/test_data.csv', index_col=0)
                        test_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Adult/adult1/test_label.csv', index_col=0)
                        test_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Adult/adult1/test_org.csv', index_col=0)

                        train_label = train_label['income-per-year']
                        valid_label = valid_label['income-per-year']
                        test_label = test_label['income-per-year']
                        ensemble_label = ensemble_label['income-per-year']

                    if dataset.dataset_name == 'propublica-recidivism':
                        train_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/COMPAS/compas/train_org.csv', index_col=0)
                        train_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/COMPAS/compas/train_data.csv', index_col=0)
                        train_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/COMPAS/compas/train_label.csv', index_col=0)

                        valid_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/COMPAS/compas/valid_data.csv', index_col=0)
                        valid_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/COMPAS/compas/valid_label.csv', index_col=0)
                        valid_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/COMPAS/compas/valid_org.csv', index_col=0)

                        ensemble_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/COMPAS/compas/ensemble_data.csv', index_col=0)
                        ensemble_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/COMPAS/compas/ensemble_label.csv', index_col=0)
                        ensemble_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/COMPAS/compas/ensemble_org.csv', index_col=0)

                        test_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/COMPAS/compas/test_data.csv', index_col=0)
                        test_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/COMPAS/compas/test_label.csv', index_col=0)
                        test_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/COMPAS/compas/test_org.csv', index_col=0)

                        train_label = train_label['two_year_recid']
                        valid_label = valid_label['two_year_recid']
                        test_label = test_label['two_year_recid']
                        ensemble_label = ensemble_label['two_year_recid']

                    if dataset.dataset_name == 'german':
                        train_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/German/german/train_org.csv', index_col=0)
                        train_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/German/german/train_data.csv', index_col=0)
                        train_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/German/german/train_label.csv', index_col=0)

                        valid_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/German/german/valid_data.csv', index_col=0)
                        valid_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/German/german/valid_label.csv', index_col=0)
                        valid_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/German/german/valid_org.csv', index_col=0)

                        ensemble_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/German/german/ensemble_data.csv', index_col=0)
                        ensemble_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/German/german/ensemble_label.csv', index_col=0)
                        ensemble_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/German/german/ensemble_org.csv', index_col=0)

                        test_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/German/german/test_data.csv', index_col=0)
                        test_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/German/german/test_label.csv', index_col=0)
                        test_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/German/german/test_org.csv', index_col=0)

                        train_label = train_label['credit']
                        valid_label = valid_label['credit']
                        test_label = test_label['credit']
                        ensemble_label = ensemble_label['credit']

                    if dataset.dataset_name == 'default':
                        train_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Default/default/train_org.csv', index_col=0)
                        train_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Default/default/train_data.csv', index_col=0)
                        train_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Default/default/train_label.csv', index_col=0)

                        valid_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Default/default/valid_data.csv', index_col=0)
                        valid_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Default/default/valid_label.csv', index_col=0)
                        valid_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Default/default/valid_org.csv', index_col=0)

                        ensemble_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Default/default/ensemble_data.csv', index_col=0)
                        ensemble_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Default/default/ensemble_label.csv', index_col=0)
                        ensemble_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Default/default/ensemble_org.csv', index_col=0)

                        test_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Default/default/test_data.csv', index_col=0)
                        test_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Default/default/test_label.csv', index_col=0)
                        test_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Default/default/test_org.csv', index_col=0)

                        train_label = train_label['default_payment_next_month']
                        valid_label = valid_label['default_payment_next_month']
                        test_label = test_label['default_payment_next_month']
                        ensemble_label = ensemble_label['default_payment_next_month']

                    if dataset.dataset_name == 'LSAT':
                        train_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/LSAT/lsat/train_org.csv', index_col=0)
                        train_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/LSAT/lsat/train_data.csv', index_col=0)
                        train_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/LSAT/lsat/train_label.csv', index_col=0)

                        valid_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/LSAT/lsat/valid_data.csv', index_col=0)
                        valid_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/LSAT/lsat/valid_label.csv', index_col=0)
                        valid_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/LSAT/lsat/valid_org.csv', index_col=0)

                        ensemble_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/LSAT/lsat/ensemble_data.csv', index_col=0)
                        ensemble_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/LSAT/lsat/ensemble_label.csv', index_col=0)
                        ensemble_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/LSAT/lsat/ensemble_org.csv', index_col=0)

                        test_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/LSAT/lsat/test_data.csv', index_col=0)
                        test_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/LSAT/lsat/test_label.csv', index_col=0)
                        test_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/LSAT/lsat/test_org.csv', index_col=0)

                        train_label = train_label['first_pf']
                        valid_label = valid_label['first_pf']
                        test_label = test_label['first_pf']
                        ensemble_label = ensemble_label['first_pf']

                    if dataset.dataset_name == 'bank':
                        train_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Bank/bank/train_org.csv', index_col=0)
                        train_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Bank/bank/train_data.csv', index_col=0)
                        train_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Bank/bank/train_label.csv', index_col=0)

                        valid_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Bank/bank/valid_data.csv', index_col=0)
                        valid_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Bank/bank/valid_label.csv', index_col=0)
                        valid_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Bank/bank/valid_org.csv', index_col=0)

                        ensemble_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Bank/bank/ensemble_data.csv', index_col=0)
                        ensemble_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Bank/bank/ensemble_label.csv', index_col=0)
                        ensemble_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Bank/bank/ensemble_org.csv', index_col=0)

                        test_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Bank/bank/test_data.csv', index_col=0)
                        test_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Bank/bank/test_label.csv', index_col=0)
                        test_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Bank/bank/test_org.csv', index_col=0)

                        train_label = train_label['y']
                        valid_label = valid_label['y']
                        test_label = test_label['y']
                        ensemble_label = ensemble_label['y']

                    if dataset.dataset_name == 'dutch':
                        train_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Dutch/dutch/train_org.csv', index_col=0)
                        train_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Dutch/dutch/train_data.csv', index_col=0)
                        train_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Dutch/dutch/train_label.csv', index_col=0)

                        valid_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Dutch/dutch/valid_data.csv', index_col=0)
                        valid_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Dutch/dutch/valid_label.csv', index_col=0)
                        valid_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Dutch/dutch/valid_org.csv', index_col=0)

                        ensemble_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Dutch/dutch/ensemble_data.csv', index_col=0)
                        ensemble_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Dutch/dutch/ensemble_label.csv', index_col=0)
                        ensemble_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Dutch/dutch/ensemble_org.csv', index_col=0)

                        test_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Dutch/dutch/test_data.csv', index_col=0)
                        test_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Dutch/dutch/test_label.csv', index_col=0)
                        test_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Dutch/dutch/test_org.csv', index_col=0)

                        train_label = train_label['occupation']
                        valid_label = valid_label['occupation']
                        test_label = test_label['occupation']
                        ensemble_label = ensemble_label['occupation']


                    if dataset.dataset_name == 'student_mat':
                        train_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Student_mat/student_mat/train_org.csv', index_col=0)
                        train_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Student_mat/student_mat/train_data.csv', index_col=0)
                        train_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Student_mat/student_mat/train_label.csv', index_col=0)

                        valid_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Student_mat/student_mat/valid_data.csv', index_col=0)
                        valid_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Student_mat/student_mat/valid_label.csv', index_col=0)
                        valid_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Student_mat/student_mat/valid_org.csv', index_col=0)

                        ensemble_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Student_mat/student_mat/ensemble_data.csv', index_col=0)
                        ensemble_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Student_mat/student_mat/ensemble_label.csv', index_col=0)
                        ensemble_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Student_mat/student_mat/ensemble_org.csv', index_col=0)

                        test_data = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Student_mat/student_mat/test_data.csv', index_col=0)
                        test_label = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Student_mat/student_mat/test_label.csv', index_col=0)
                        test_org = pd.read_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/Student_mat/student_mat/test_org.csv', index_col=0)

                        train_label = train_label['G3']
                        valid_label = valid_label['G3']
                        test_label = test_label['G3']
                        ensemble_label = ensemble_label['G3']

                train_org.reset_index(inplace=True, drop=True)
                train_data.reset_index(inplace=True, drop=True)
                train_label.reset_index(inplace=True, drop=True)
                valid_data.reset_index(inplace=True, drop=True)
                valid_label.reset_index(inplace=True, drop=True)
                valid_org.reset_index(inplace=True, drop=True)
                ensemble_data.reset_index(inplace=True, drop=True)
                ensemble_label.reset_index(inplace=True, drop=True)
                ensemble_org.reset_index(inplace=True, drop=True)
                test_data.reset_index(inplace=True, drop=True)
                test_label.reset_index(inplace=True, drop=True)
                test_org.reset_index(inplace=True, drop=True)

                if is_smaller == 1:
                    preserve = 1000
                    n_data = train_data.shape[0]
                    rate = np.min([preserve / n_data, 1])
                    if rate < 1:
                        train_org, train_data, train_label = get_smaller(train_org, train_data, train_label, rate)
                        valid_org, valid_data, valid_label = get_smaller(valid_org, valid_data, valid_label, rate)
                        ensemble_org, ensemble_data, ensemble_label = get_smaller(ensemble_org, ensemble_data, ensemble_label, rate)
                        test_org, test_data, test_label = get_smaller(test_org, test_data, test_label, rate)

                save_name = 'dutch'
                save_folder = 'Dutch'
                if save_csv == 1:
                    test_org.to_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/{}/{}/test_org.csv'.format(save_folder, save_name), index=True, header=True)
                    test_data.to_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/{}/{}/test_data.csv'.format(save_folder, save_name), index=True, header=True)
                    test_label.to_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/{}/{}/test_label.csv'.format(save_folder, save_name), index=True, header=True)

                    train_org.to_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/{}/{}/train_org.csv'.format(save_folder, save_name), index=True, header=True)
                    train_data.to_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/{}/{}/train_data.csv'.format(save_folder, save_name), index=True, header=True)
                    train_label.to_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/{}/{}/train_label.csv'.format(save_folder, save_name), index=True, header=True)

                    valid_data.to_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/{}/{}/valid_data.csv'.format(save_folder, save_name), index=True, header=True)
                    valid_label.to_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/{}/{}/valid_label.csv'.format(save_folder, save_name), index=True, header=True)
                    valid_org.to_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/{}/{}/valid_org.csv'.format(save_folder, save_name), index=True, header=True)

                    ensemble_data.to_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/{}/{}/ensemble_data.csv'.format(save_folder, save_name), index=True, header=True)
                    ensemble_label.to_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/{}/{}/ensemble_label.csv'.format(save_folder, save_name), index=True, header=True)
                    ensemble_org.to_csv('FairEMOL/EvolutionaryCodes/data/ensemble_data/{}/{}/ensemble_org.csv'.format(save_folder, save_name), index=True, header=True)

                # 是否删除sensitive attribution
                if preserve_sens_in_net == 1:
                    newtrain_data = train_data.copy()
                    newtest_data = test_data.copy()
                    newvalid_data = valid_data.copy()
                    newensemble_data = ensemble_data.copy()
                else:
                    attribution = train_data.columns
                    sens_dict = []
                    for sens in sensitive_attributions:
                        for attr in attribution:
                            temp = sens + '_'
                            if temp in attr:
                                sens_dict.append(attr)
                    newtrain_data = train_data.copy()
                    newtest_data = test_data.copy()
                    newvalid_data = valid_data.copy()
                    newensemble_data = ensemble_data.copy()
                    newdata_x = data_x.copy()

                    newtrain_data.drop(columns=sens_dict, inplace=True)
                    newtest_data.drop(columns=sens_dict, inplace=True)
                    newvalid_data.drop(columns=sens_dict, inplace=True)
                    newensemble_data.drop(columns=sens_dict, inplace=True)
                    newdata_x.drop(columns=sens_dict, inplace=True)

                # Normalization
                normalize = StandardScaler()

                # Fitting only on training data
                normalize.fit(newdata_x)
                train_data_norm = normalize.transform(newtrain_data)

                # Applying same transformation to test data
                test_data_norm = normalize.transform(newtest_data)

                # Applying same transformation to validation data
                valid_data_norm = normalize.transform(newvalid_data)

                # Applying same transformation to ensemble data
                ensemble_data_norm = normalize.transform(newensemble_data)

                # Change labels into integer
                train_y = make_class_attr_num(train_label.copy(), dataset.get_positive_class_val(""))
                test_y = make_class_attr_num(test_label.copy(), dataset.get_positive_class_val(""))
                valid_y = make_class_attr_num(valid_label.copy(), dataset.get_positive_class_val(""))
                ensemble_y = make_class_attr_num(ensemble_label.copy(), dataset.get_positive_class_val(""))

                DATA_names = ['train_data', 'train_data_norm', 'train_label', 'train_y', 'train_org',
                              'valid_data', 'valid_data_norm', 'valid_label', 'valid_y', 'valid_org',
                              'test_data', 'test_data_norm', 'test_label', 'test_y', 'test_org',
                              'enemble_data', 'enemble_data_norm', 'enemble_label', 'ensemble_y', 'enemble_org',
                              'org_data', 'positive_class',
                              'positive_class_name',
                              'Groups_info', 'privileged_class_names', 'sens_attrs']

                DATA = dict((k, []) for k in DATA_names)

                DATA['train_data'] = train_data
                DATA['train_data_norm'] = train_data_norm
                DATA['train_label'] = train_label
                DATA['train_y'] = train_y.astype('int')

                DATA['valid_data'] = valid_data
                DATA['valid_data_norm'] = valid_data_norm
                DATA['valid_label'] = valid_label
                DATA['valid_y'] = valid_y.astype('int')

                DATA['ensemble_data'] = ensemble_data
                DATA['ensemble_data_norm'] = ensemble_data_norm
                DATA['ensemble_label'] = ensemble_label
                DATA['ensemble_y'] = ensemble_y.astype('int')

                DATA['test_data'] = test_data
                DATA['test_data_norm'] = test_data_norm
                DATA['test_label'] = test_label
                DATA['test_y'] = test_y.astype('int')

                DATA['positive_class'] = dataset.get_positive_class_val("")
                DATA['privileged_class_names'] = dataset.privileged_class_names

                DATA['org_data'] = org_data
                DATA['train_org'] = train_org
                DATA['test_org'] = test_org
                DATA['valid_org'] = valid_org
                DATA['ensemble_org'] = ensemble_org

                DATA['positive_class_name'] = dataset.get_class_attribute()
                DATA['sens_attrs'] = dataset.get_sensitive_attributes()

                data_obj = dataset

                sens_idxs_name_train, sens_idxs_train, group_dict_train = analyze_data(train_data, data, dataset,
                                                                                       sensitive_attributions)
                sens_idxs_name_valid, sens_idxs_valid, group_dict_valid = analyze_data(valid_data, data, dataset,
                                                                                       sensitive_attributions)
                sens_idxs_name_test, sens_idxs_test, group_dict_test = analyze_data(test_data, data, dataset,
                                                                                    sensitive_attributions)

                sens_idxs_name_ensemble, sens_idxs_ensemble, group_dict_ensemble = analyze_data(ensemble_data, data, dataset,
                                                                                    sensitive_attributions)

                Groups_name = ['sens_idxs_train', 'sens_idxs_name_train', 'group_dict_train',
                               'sens_idxs_valid', 'sens_idxs_name_valid', 'group_dict_valid',
                               'sens_idxs_test', 'sens_idxs_name_test', 'group_dict_test',
                               'sens_idxs_ensemble', 'sens_idxs_name_ensemble', 'group_dict_ensemble']

                Groups_info = dict((k, []) for k in Groups_name)
                Groups_info['sens_idxs_train'] = sens_idxs_train
                Groups_info['sens_idxs_name_train'] = sens_idxs_name_train
                Groups_info['group_dict_train'] = group_dict_train
                Groups_info['sens_idxs_valid'] = sens_idxs_valid
                Groups_info['sens_idxs_name_valid'] = sens_idxs_name_valid
                Groups_info['group_dict_valid'] = group_dict_valid
                Groups_info['sens_idxs_test'] = sens_idxs_test
                Groups_info['sens_idxs_name_test'] = sens_idxs_name_test
                Groups_info['group_dict_test'] = group_dict_test
                Groups_info['sens_idxs_ensemble'] = sens_idxs_ensemble
                Groups_info['sens_idxs_name_ensemble'] = sens_idxs_name_ensemble
                Groups_info['group_dict_ensemble'] = group_dict_ensemble

                DATA['Groups_info'] = Groups_info

    return DATA, data_obj

