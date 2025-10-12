import numpy as np
from typing import List, Optional
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, recall_score, f1_score, cohen_kappa_score
from sklearn.metrics import roc_auc_score, roc_curve, log_loss, auc, precision_recall_curve

def calculate_metric_for_each_label(multi_label_confusion_matrix: np.ndarray) -> List:
    
    def metric_results(TP, TN, FN, FP):
       
        result = {}
        accuracy = (TP + TN) / (TP + TN + FN + FP)
        result.update({'Acc': round(accuracy, 5)})

        
        if (TP + FP) != 0:
            precision = TP / (TP + FP)
        else:
            precision = 0.00  
        result.update({'Pre': round(precision, 5)})

        if (TP + FN) != 0:
            recall = TP / (TP + FN)
        else:
            recall = 0.00
        result.update({'Sen': round(recall, 5)}) 

        if precision == 0:
            precision = precision + 1e-6
        if recall == 0:
            recall = recall + 1e-6
        F1_score = 1 / (1/2 * (1/precision + 1/recall))
        result.update({'F1': round(F1_score, 5)})

        # calculate specificity, For 2 class
        if (TN + FP) != 0:
            specificity = TN / (TN + FP)
        else:
            specificity = 0.00
        result.update({'Spec': round(specificity, 5)})

        return result

    num_classes = multi_label_confusion_matrix.shape[0]
    result_info_summary = []

    for label_index in range(num_classes):
        all = multi_label_confusion_matrix.sum()
        TP = multi_label_confusion_matrix[label_index][label_index]
        FP = np.sum(multi_label_confusion_matrix[:, label_index]) - TP
        FN = np.sum(multi_label_confusion_matrix[label_index, :]) - TP
        TN = all - TP - FP - FN
        # print(f"TP:{TP}, TN:{TN},FN:{FN},FP:{FP}")
        assert TP + TN + FN + FP == all, "size of TP + TN + FN + FP and Total is not equal!"
        metric_result_info = metric_results(TP, TN, FN, FP)
        result_info_summary.append(metric_result_info)
    return result_info_summary

def calculate_metric_for_multi_label(num_classes: int,
                                     pred_label,
                                     gt_label,
                                     label_probability,
                                     relevant_labels:Optional[List]=None,
                                     metric_average="macro",
                                     AUC_flag=True):
    # print(f"label:{len(pred_label)}, label_probability:{len(label_probability)}")
    # for probability in label_probability:
    #     print(probability)
    labels = None
    relevant_probs = None
    if relevant_labels is not None:
        label_probability = np.array(label_probability)
        relevant_probs = label_probability[:, relevant_labels]
        pred_label = np.argmax(relevant_probs, axis=1)
        pred_label = np.array(relevant_labels)[pred_label]
        labels = relevant_labels

    metric_result_info = {}
    cm = confusion_matrix(gt_label, pred_label, labels=labels)
    metric_result_info.update({'confusion_matrix': cm})

    accuracy = accuracy_score(gt_label, pred_label)
    metric_result_info.update({'Acc': accuracy})
    # multi_class='ovo'
    if AUC_flag:
        if relevant_probs is None:
            AUC = roc_auc_score(np.array(gt_label), np.array(label_probability), multi_class='ovr', average=metric_average)
        else:
            y_true_one_hot = label_binarize(gt_label, classes=relevant_labels)
            AUC = roc_auc_score(y_true_one_hot, relevant_probs, average=metric_average)

        metric_result_info.update({'AUC': AUC})

    precision = precision_score(gt_label, pred_label, labels=labels, average=metric_average)
    metric_result_info.update({'Pre': precision})

    recall = recall_score(gt_label, pred_label, labels=labels, average=metric_average)
    metric_result_info.update({'Sen': recall})  

    F1_score = f1_score(gt_label, pred_label, labels=labels, average=metric_average)
    metric_result_info.update({'F1': F1_score})
    # calculate specificity
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    TNR = TN / (TN + FP)  # specificity
    metric_result_info.update({'Spec': (np.sum(TNR) / num_classes) if relevant_labels is None else np.sum(TNR) / len(relevant_labels)})

    kappa = cohen_kappa_score(gt_label, pred_label, labels=labels)
    metric_result_info.update({'Kappa': kappa})
    return metric_result_info

def calculate_metric_for_binary_label(pred_label, gt_label, label_probability):
    """
                N      P  -> pre
     |      N   TN     FP
    real    P   FN     TP

    """
    metric_result_info = {}
    cm = confusion_matrix(gt_label, pred_label)
    metric_result_info.update({'confusion_matrix': cm})

    accuracy = accuracy_score(gt_label, pred_label)
    metric_result_info.update({'Acc': accuracy})

    AUC = roc_auc_score(np.array(gt_label), np.array(label_probability)[:, 1])
    metric_result_info.update({'AUC': AUC})

    pre, rec, _ = precision_recall_curve(np.array(gt_label), np.array(label_probability)[:, 1])
    pr_auc = auc(rec, pre)
    metric_result_info.update({'AUCPR': pr_auc})

    precision = precision_score(gt_label, pred_label)
    metric_result_info.update({'Pre': precision})
    # precision_1 = (cm[1][1]) / (cm[1][1] + cm[0][1])
    # print('precision_1"', precision_1)
    recall = recall_score(gt_label, pred_label)
    metric_result_info.update({'Sen': recall}) 
    # recall_ = (cm[1][1]) / (cm[1][1] + cm[1][0])
    # print('sensitivity_:', recall_)
    F1_score = f1_score(gt_label, pred_label)
    metric_result_info.update({'F1': F1_score})

    # calculate specificity, For 2 class
    specificity = (cm[0][0]) / (cm[0][0] + cm[0][1])
    metric_result_info.update({'Spec': specificity})

    kappa = cohen_kappa_score(gt_label, pred_label)
    metric_result_info.update({'Kappa': kappa})
    return metric_result_info

def print_metric_info(metric_result_info):
    for metric in metric_result_info:
        if metric == 'confusion_matrix':
            pass
        else:
            print(metric, ":", metric_result_info[metric])
