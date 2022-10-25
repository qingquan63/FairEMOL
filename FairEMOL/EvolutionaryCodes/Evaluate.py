import numpy as np
from itertools import product
import copy
from sklearn.metrics import log_loss


def get_label(logits):
    pred_label = logits
    pred_label[np.where(pred_label >= 0.5)] = 1
    pred_label[np.where(pred_label < 0.5)] = 0
    pred_label = pred_label.reshape(1, logits.shape[0] * logits.shape[1])
    pred_label = pred_label.reshape(1, -1)
    return pred_label


def get_average(group_values, plan):
    if plan == 1:
        values = []
        num_group = len(group_values)
        if num_group > 1:
            for i in range(num_group):
                if i == (num_group - 1):
                    break
                for j in range(i+1, num_group):
                    values.append(np.abs(group_values[i] - group_values[j]))

            return np.mean(values)
        else:
            return 0
    else:
        values = 0.0
        num_group = len(group_values)
        if num_group > 1:
            for i in range(num_group):
                if i == (num_group - 1):
                    break
                for j in range(i + 1, num_group):
                    values = np.max([values, np.abs(group_values[i] - group_values[j])])

            return values
        else:
            return 0


def get_obj(group_values, plan):
    Group_values = copy.deepcopy(group_values)
    if plan == 1:
        # calculate the difference
        values = []
        num_group = len(group_values)
        if num_group > 1:
            for i in range(num_group):
                if i == (num_group - 1):
                    break
                for j in range(i+1, num_group):
                    values.append(np.abs(group_values[i] - group_values[j]))

            return 0.5 * (np.mean(values) + np.max(values))

        elif num_group == 1:
            return 1

        else:
            return 0
    else:
        # calculate the ratio
        values = []
        num_group = len(group_values)
        if num_group > 1:
            for i in range(num_group):
                if i == (num_group - 1):
                    break
                for j in range(i + 1, num_group):
                    if group_values[j] == 0 and group_values[i] == 0:
                        values.append(1)
                    elif group_values[j] == 0 and group_values[i] != 0:
                        values.append(0)
                    elif group_values[j] != 0 and group_values[i] == 0:
                        values.append(0)
                    else:
                        values.append(np.min([(group_values[j]/group_values[i]), (group_values[i]/group_values[j])]))
            return 0.5 * (1 - np.mean(values) + 1 - np.min(values))

        elif num_group == 1:
            return 1

        else:
            return 0


def calcul_all_fairness_objs(data, data_norm, logits, truelabel, sensitive_attributions, alpha, obj_names):
    logits = logits.reshape(1, -1).astype(np.float64)
    truelabel = truelabel.astype(np.float64).reshape(1, -1)
    pred_label = get_label(logits.copy())

    benefits = logits - truelabel + 1  # new version in section 3.1
    benefits_group = copy.deepcopy(benefits)

    total_num = logits.shape[1]

    attribution = data.columns
    group_attr = []
    check_gmuns = []
    group_dict = {}

    Disparate_impact = []
    Calibration_Neg = []
    Predictive_parity = []
    Discovery_ratio = []
    Discovery_diff = []
    Predictive_equality = []
    FPR_ratio = []
    Equal_opportunity = []
    Equalized_odds1 = []
    Equalized_odds2 = []
    Average_odd_diff = []
    Conditional_use_accuracy_equality1 = []
    Conditional_use_accuracy_equality2 = []
    Overall_accuracy = []
    Error_ratio = []
    Error_diff = []
    Statistical_parity = []
    FOR_ratio = []
    FOR_diff = []
    FPR_ratio = []
    FNR_ratio = []
    FNR_diff = []

    for sens in sensitive_attributions:
        temp = []
        for attr in attribution:
            temp1 = sens + '_'
            if temp1 in attr:
                temp.append(attr)
        group_dict.update({sens: temp})

    for sens in sensitive_attributions:
        group_attr.append(group_dict[sens])
    for item in product(*eval(str(group_attr))):
        group = item
        flag = np.ones([1, truelabel.shape[1]]) == np.ones([1, truelabel.shape[1]])
        for g in group:
            flag = flag & data[g]
        g_num = np.sum(flag)
        if g_num != 0:
            check_gmuns.append(g_num)
            g_idx = np.array(np.where(flag)).reshape([1, -1])[0]
            benefits_group[0, g_idx] = np.mean(benefits[0, g_idx])
            # g_logits = logits[0, g_idx]
            g_truelabel = truelabel[0, g_idx]
            g_predlabel = pred_label[0, g_idx]

            # P(d=1 | g)
            # Disparate Impact  or  Statistical Parity
            if "Disparate_impact" in obj_names or "Statistical_parity" in obj_names:
                Disparate_impact.append(np.sum(g_predlabel) / g_num)
                Statistical_parity = Disparate_impact

            # P(y=d | g)
            # Overall accuracy
            if "Overall_accuracy" in obj_names:
                Overall_accuracy.append(np.sum(g_truelabel == g_predlabel) / g_num)

            # P(y != d, g)
            # Error ratio   or   Error diff
            if "Error_ratio" in obj_names or "Error_diff" in obj_names:
                Error_ratio.append(1 - np.sum(g_truelabel == g_predlabel) / total_num)
                Error_diff = Error_ratio

            # P(y=1 | d=1, g)
            # Predictive parity
            if "Predictive_parity" in obj_names:
                if np.sum(g_predlabel) > 0:
                    Predictive_parity.append(np.sum(g_truelabel * g_predlabel) / np.sum(g_predlabel))

            # P(y=0 | d=1, g)
            # Discovery ratio  or   Discovery diff
            if "Discovery_ratio" in obj_names or "Discovery_diff" in obj_names :
                if np.sum(g_predlabel) > 0:
                    Discovery_ratio.append((np.sum((1-g_truelabel) * g_predlabel) / np.sum(g_predlabel)))
                    Discovery_diff = Discovery_ratio

            # P(y=1 | d=0, g)
            # Calibration-   or   FOR ratio    or   FOR diff
            if "Calibration_neg" in obj_names or "FOR_ratio" in obj_names or "FOR_diff" in obj_names:
                if np.sum(1-g_predlabel) > 0:
                    Calibration_Neg.append(np.sum(g_truelabel * (1-g_predlabel)) / np.sum(1-g_predlabel))
                    FOR_ratio = Calibration_Neg
                    FOR_diff = Calibration_Neg

            # P(d=1 | y=0, g)
            # Predictive equality   or   FPR ratio
            if "Predictive_equality" in obj_names or "FPR_ratio" in obj_names:
                if np.sum(1-g_truelabel) > 0:
                    Predictive_equality.append(np.sum(g_predlabel * (1-g_truelabel)) / np.sum(1-g_truelabel))
                    FPR_ratio = Predictive_equality

            # P(d=1 | y=1, g)
            # Equal opportunity
            if "Equal_opportunity" in obj_names:
                if np.sum(g_truelabel) > 0:
                    Equal_opportunity.append(np.sum(g_predlabel * g_truelabel) / np.sum(g_truelabel))

            # P(d=0 | y=1, g)
            # FNR ratio    or    FNR diff
            if "FNR_ratio" in obj_names or "FNR_diff" in obj_names:
                if np.sum(g_truelabel) > 0:
                    FNR_ratio.append(np.sum((1-g_predlabel) * g_truelabel) / np.sum(g_truelabel))
                    FNR_diff = FNR_ratio

            # P(d=1 | y=0, g) and P(d=1 | y=1, g)
            # Equalized odds
            if "Equalized_odds" in obj_names:
                if np.sum(g_truelabel) > 0:
                    Equalized_odds1.append(np.sum(g_predlabel * g_truelabel) / np.sum(g_truelabel))
                if np.sum(1-g_truelabel) > 0:
                    Equalized_odds2.append(np.sum(g_predlabel * (1-g_truelabel)) / np.sum(1-g_truelabel))

            # Conditional use accuracy equality
            if "Conditional_use_accuracy_equality" in obj_names:
                if np.sum(g_predlabel) > 0:
                    Conditional_use_accuracy_equality1.append(np.sum(g_truelabel * g_predlabel) / np.sum(g_predlabel))
                if np.sum(1-g_predlabel) > 0:
                    Conditional_use_accuracy_equality2.append(np.sum((1-g_truelabel) * (1-g_predlabel)) / np.sum(1-g_predlabel))

            # P(d=1 | y=0, g) + P(d=1 | y=1, g)
            # Average odd difference
            if "Average_odd_diff" in obj_names:
                if np.sum(g_truelabel) > 0 and np.sum(1 - g_truelabel) > 0:
                    Average_odd_diff.append((np.sum(g_predlabel * g_truelabel) / np.sum(g_truelabel)) + (
                                np.sum(g_predlabel * (1 - g_truelabel)) / np.sum(1 - g_truelabel)))

    Groups_info = {}
    if "Accuracy" in obj_names:
        Groups_info.update({"Accuracy": np.mean(pred_label == truelabel)})

    # BCE loss
    if "Error" in obj_names:
        BCE_loss = log_loss(truelabel.reshape(-1, 1), np.hstack([1 - logits.reshape(-1, 1), logits.reshape(-1, 1)]))
        Groups_info.update({"Error": BCE_loss})

    # Individual unfairness = within-group + between-group
    if "Individual_fairness" in obj_names:
        Individual_fairness_val = generalized_entropy_index(benefits, alpha)
        Groups_info.update({"Individual_fairness": Individual_fairness_val})

    # Group unfairness = between-group
    if "Group_fairness" in obj_names:
        Group_fairness_val = generalized_entropy_index(benefits_group, alpha)
        Groups_info.update({"Group_fairness": Group_fairness_val})

    # Disparate impact
    if "Disparate_impact" in obj_names:
        Disparate_impact_val = get_obj(Disparate_impact, 2)
        Groups_info.update({"Disparate_impact": Disparate_impact_val})

    # Statistical parity
    if "Statistical_parity" in obj_names:
        Statistical_parity_val = get_obj(Statistical_parity, 1)
        Groups_info.update({"Statistical_parity": Statistical_parity_val})

    # Overall accuracy
    if "Overall_accuracy" in obj_names:
        Overall_accuracy_val = get_obj(Overall_accuracy, 1)
        Groups_info.update({"Overall_accuracy": Overall_accuracy_val})

    # Error ratio
    if "Error_ratio" in obj_names:
        Error_ratio_val = get_obj(Error_ratio, 2)
        Groups_info.update({"Error_ratio": Error_ratio_val})

    # Error diff
    if "Error_diff" in obj_names:
        Error_diff_val = get_obj(Error_diff, 1)
        Groups_info.update({"Error_diff": Error_diff_val})

    # Predictive parity
    if "Predictive_parity" in obj_names:
        Predictive_parity_val = get_obj(Predictive_parity, 1)
        Groups_info.update({"Predictive_parity": Predictive_parity_val})

    # Discovery ratio
    if "Discovery_ratio" in obj_names:
        Discovery_ratio_val = get_obj(Discovery_ratio, 2)
        Groups_info.update({"Discovery_ratio": Discovery_ratio_val})

    # Discovery diff
    if "Discovery_diff" in obj_names:
        Discovery_diff_val = get_obj(Discovery_diff, 1)
        Groups_info.update({"Discovery_diff": Discovery_diff_val})

    # Calibration Neg
    if "Calibration_neg" in obj_names:
        Calibration_Neg_val = get_obj(Calibration_Neg, 1)
        Groups_info.update({"Calibration_neg": Calibration_Neg_val})

    # FOR ratio
    if "FOR_ratio" in obj_names:
        FOR_ratio_val = get_obj(FOR_ratio, 2)
        Groups_info.update({"FOR_ratio": FOR_ratio_val})

    # FOR diff
    if "FOR_diff" in obj_names:
        FOR_diff_val = get_obj(FOR_diff, 1)
        Groups_info.update({"FOR_diff": FOR_diff_val})

    # Predictive equality
    if "Predictive_equality" in obj_names:
        Predictive_equality_val = get_obj(Predictive_equality, 1)
        Groups_info.update({"Predictive_equality": Predictive_equality_val})

    # FPR ratio
    if "FPR_ratio" in obj_names:
        FPR_ratio_val = get_obj(FPR_ratio, 2)
        Groups_info.update({"FPR_ratio": FPR_ratio_val})

    # Equal opportunity
    if "Equal_opportunity" in obj_names:
        Equal_opportunity_val = get_obj(Equal_opportunity, 1)
        Groups_info.update({"Equal_opportunity": Equal_opportunity_val})

    # FNR ratio
    if "FNR_ratio" in obj_names:
        FNR_ratio_val = get_obj(FNR_ratio, 2)
        Groups_info.update({"FNR_ratio": FNR_ratio_val})

    # FNR diff
    if "FNR_diff" in obj_names:
        FNR_diff_val = get_obj(FNR_diff, 1)
        Groups_info.update({"FNR_diff": FNR_diff_val})

    # Equalized odds
    if "Equalized_odds" in obj_names:
        Equalized_odds_val = 0.5 * (get_obj(Equalized_odds1, 1) + get_obj(Equalized_odds2, 1))
        Groups_info.update({"Equalized_odds": Equalized_odds_val})

    # Conditional use accuracy equality
    if "Conditional_use_accuracy_equality" in obj_names:
        Conditional_use_accuracy_equality_val = 0.5 * (
                    get_obj(Conditional_use_accuracy_equality1, 1) + get_obj(Conditional_use_accuracy_equality2, 1))
        Groups_info.update({"Conditional_use_accuracy_equality": Conditional_use_accuracy_equality_val})

    # Average odd difference
    if "Average_odd_diff" in obj_names:
        Average_odd_diff_val = 0.5 * get_obj(Average_odd_diff, 1)
        Groups_info.update({"Average_odd_diff": Average_odd_diff_val})

    return Groups_info


def generalized_entropy_index(b, alpha):
    # https://github.com/Trusted-AI/AIF360/blob/master/aif360/metrics/classification_metric.py#L664
    # pred_label = get_label(b.copy())
    # benefits = pred_label - truelabel + 1  # original version
    # benefits = logits - truelabel + 1  # new version in section 3.1
    # b = benefits
    if alpha == 1:
        return np.mean(np.log((b / np.mean(b)) ** b) / np.mean(b))
    elif alpha == 0:
        return -np.mean(np.log(b / np.mean(b)) / np.mean(b))
    else:
        return np.mean((b / np.mean(b)) ** alpha - 1) / (alpha * (alpha - 1))


def Cal_objectives(data, data_norm, logits, truelabel, sensitive_attributions, alpha, obj_names=None):
    sum_num = logits.shape[0] * logits.shape[1]
    logits = np.array(logits).reshape([1, sum_num])
    truelabel = np.array(truelabel).reshape([1, sum_num])
    Groups_info = calcul_all_fairness_objs(data,
                                           data_norm,
                                           logits,
                                           truelabel,
                                           sensitive_attributions,
                                           alpha, obj_names)
    return Groups_info
