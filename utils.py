import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PATHS
import os
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

# bootstrap set up
def prepare_df(csv_path):
    """Prepare dataframe for flow_from_dataframe"""
    df = pd.read_csv(csv_path)
    df['class'] = df["is_healthy"].apply(lambda x : "healthy" if x else "unhealthy")
    df['filename'] = (df["Image Id"].astype(str) + ".png")
    return df
        
# metrics
def get_conf(true, pred_proba, thresh):
    """Return tp, fp, tn, fn or confusion matrix values"""
    pred = np.array([int(i) for i in (pred_proba >= thresh)])
    true = np.array(true)
    
    tp = ((true == 1) & (pred == 1)).sum()
    fp = ((true == 0) & (pred == 1)).sum()
    tn = ((true == 0) & (pred == 0)).sum()
    fn = ((true == 1) & (pred == 0)).sum()
    return tp, fp, tn, fn

def get_sensitivity(tp, fp, tn, fn):
    """Return the true positive rate/sensitivity/recall: probability of detection"""
    return tp / (tp + fn + 1e-7)

def get_fpr(tp, fp, tn, fn):
    """Return the false positive rate: probability of false alarm"""
    return fp / (fp + tn + 1e-7)
    
def get_specificity(tp, fp, tn, fn):
    """Return the specificity, true negative rate"""
    return tn / (fp + tn + 1e-7)

def get_ppv(tp, fp, tn, fn):
    """Return the positive predictive value (precision)"""
    return tp / (tp + fp + 1e-7)

def get_npv(tp, fp, tn, fn):
    """Return the negative predictive value"""
    return tn / (fn + tn + 1e-7)

def get_fbeta(tp, fp, tn, fn, beta=1):
    """Return the f-beta score at beta"""
    inv_alpha = 1 + beta**2
    return inv_alpha * tp / ((inv_alpha * tp) + (beta**2 * fp) + fp)

def get_mcc(tp, fp, tn, fn):
    """Return the matthews correlation coeff"""
    num = (tp * tn) - (fp * fn)
    den = np.sqrt((tp + fp)*(tp + fn)*(tn+fp)*(tn+fn))
    return num / den
    
# results generation
def _get_thresh(num_thresh):
    thresh_arr = np.linspace(start=0.0, stop=1.0, num=num_thresh, endpoint=True)
    return thresh_arr

def get_roc(true, pred_proba, num_thresh, return_thresh_arr=False):
    """Returns the ROC values for predicted probabilities for a class"""
    tprs, fprs = [], []
    thresh_arr = _get_thresh(num_thresh)
    for thresh in thresh_arr:
        conf = get_conf(true, pred_proba, thresh=thresh)
        tpr = get_sensitivity(*conf) # recall
        fpr = get_fpr(*conf)
        tprs.append(tpr)
        fprs.append(fpr)
    if return_thresh_arr:
        return fprs, tprs, thresh_arr
    return fprs, tprs

def get_pr(true, pred_proba, num_thresh, return_thresh_arr=False):
    """Return the PR-curve values"""
    rs, ps = [], []
    thresh_arr = _get_thresh(num_thresh)    
    for thresh in thresh_arr:
        conf = get_conf(true, pred_proba, thresh=thresh)
        r = get_sensitivity(*conf) # tpr
        p = get_ppv(*conf) # precision
        rs.append(r)
        ps.append(p)
    if return_thresh_arr:
        return rs, ps, thresh_arr
    return rs, ps

def get_class_proba(preds, class_index):
    """Return softmax probabilities of a class"""
    return preds[:,class_index]

def get_results(true, preds, num_thresh, class_index):
    """Return results for class_index of interest
    
    Parameters
    ----------
    true : true labels, list or nparray
    preds : all (unbatched) predictions from model.predict
    num_thresh : number of thresholds to try
    class_index : class index of interest
    """
    probas = get_class_proba(preds, 0), get_class_proba(preds, 1)
    perfect = (_get_thresh(num_thresh), _get_thresh(num_thresh))
    
    roc = get_roc(true, probas[class_index], num_thresh) 
    pr = get_pr(true, probas[class_index], num_thresh)
    return roc, pr, perfect, probas


def plot_results(ax, roc, pr, perfect, probas, true):
    proba_0, proba_1 = [], []
    for index, label in enumerate(true):
        if label:
            # class of interest is 1 (unhealthy)
            proba_1.append(probas[1][index])
        else:
            proba_0.append(probas[1][index])
    ax[0].hist(x=[proba_0, proba_1], 
               bins=20, color=['gray', 'orange'], alpha=0.5)
    ax[0].set_title("Histogram of classes")
    ax[0].set_xlabel("Probability score")
    ax[0].set_ylabel("Frequency")
    
    ax[1].plot(roc[0], roc[1], 'b-', alpha=0.5)
    ax[1].plot(perfect[0], perfect[1], 'k--')
    ax[1].set_title("ROC")
    ax[1].set_xlabel("False Positive Rate")    
    ax[1].set_ylabel("True Positive Rate")
    
    ax[2].plot(pr[0], pr[1], 'b-', alpha=0.5)
    ax[2].plot(perfect[0], perfect[1][::-1], 'k--')
    ax[2].set_title("PR-curve")
    ax[2].set_xlabel("Recall")
    ax[2].set_ylabel("Precision")    
    
    for _ in ax:
        _.spines.right.set_visible(False)
        _.spines.top.set_visible(False)        
    return ax


    
# tensorflow.keras    
def inspect_dataset(dataset):
    """Return inspect results on 1 batch of dataset (generator)"""
    for item in dataset:
        print("Item type: ", type(item))
        data_batch, labels_batch = item
        print('data batch type: ', type(data_batch))
        print('labels batch type: ', type(labels_batch))
        print('data batch shape:', data_batch.shape)
        print('labels batch shape:', labels_batch.shape)
        break
        
def get_save_path(model_name):
    """Return a save path for an input model name"""
    model_dir = PATHS.models_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_name)
    return model_filepath

def get_last_conv_layer(model):
    """Return the last convolution layer assumed to be the last 4-dim layer"""
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            return layer.name
    raise ValueError("Could not find 4D layer. GradCAM wont work")