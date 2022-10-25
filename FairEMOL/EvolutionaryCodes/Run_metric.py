from FairEMOL.EvolutionaryCodes import load_data
from FairEMOL.EvolutionaryCodes.metrics.list import get_metrics
from FairEMOL.EvolutionaryCodes.data.objects.list import DATASETS, get_dataset_names
from FairEMOL.EvolutionaryCodes.data.objects.ProcessedData import ProcessedData
import numpy as np

# dataset_obj=Adult()


def Alg_Evaluation(dataset_obj, problem, logits, predic_label, true_label, test_org, supported_tag):

    processed_dataset = ProcessedData(dataset_obj)
    sensitive_dict = processed_dataset.get_sensitive_values(supported_tag)
    all_sensitive_attributes = dataset_obj.get_sensitive_attributes_with_joint()

    dict_sensitive_lists = {}
    for sens1 in all_sensitive_attributes:
        dict_sensitive_lists[sens1] = np.array(test_org[sens1].values.tolist())

    Result = {}
    for single_sensitive in all_sensitive_attributes:
        results = {}
        privileged_vals = dataset_obj.get_privileged_class_names_with_joint(supported_tag)
        positive_val = dataset_obj.get_positive_class_val(supported_tag)

        for metric in get_metrics(dataset_obj, sensitive_dict, supported_tag):
            result = metric.calc(true_label, predic_label, dict_sensitive_lists, single_sensitive,
                                 privileged_vals, positive_val, problem, logits)
            # print(metric.name, ' : ', result)
            if metric.name == 'AUC':
                for item in result:
                    results[item] = result[item]
            else:
                results[metric.name] = result
        Result[single_sensitive] = results
    return Result

    # print("over")


# DATA = load_data('adult', test_size=0.25)
#
# # train_data = np.array(DATA['train_data'])
# # train_data_norm = np.array(DATA['train_data_norm'])
# # test_data = np.array(DATA['test_data'])
# # test_data_norm = np.array(DATA['test_data_norm'])
# # train_label = np.array(DATA['train_label'])
# test_label = np.array(DATA['test_label'])
# # train_y = np.array(DATA['train_y'])
# # test_y = np.array(DATA['test_y'])
# #
# # org_data = np.array(DATA['org_data'])
# # train_org = np.array(DATA['train_org'])
# test_org = DATA['test_org']
# # positive_class_name = DATA['positive_class_name']
# # positive_class = DATA['positive_class']
#
# Result = Alg_Evaluation(DATASETS[0], test_label, test_label, test_org, 'numerical-for-NN')
# print(Result)

