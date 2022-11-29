def inspect_dataset(dataset):
    """Return inspect results on dataset"""
    for item in dataset:
        print("Item type: ", type(item))
        data_batch, labels_batch = item
        print('data batch type: ', type(data_batch))
        print('labels batch type: ', type(labels_batch))
        print('data batch shape:', data_batch.shape)
        print('labels batch shape:', labels_batch.shape)
        break
        
def get_save_path(model_name):
    model_dir = PATHS.models_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_name)
    return model_filepath

        
# metrics
def get_sensitivity(tp, fp, tn, fn):
    return tp / (tp + fn)
    
def get_specificity(tp, fp, tn, fn):
    return tn / (fp + tn)

def get_ppv(tp, fp, tn, fn):
    return tp / (tp + fp)

def get_npv(tp, fp, tn, fn):
    return tn / (fn + tn)

def get_fbeta(tp, fp, tn, fn, beta=1):
    inv_alpha = 1 + beta**2
    return inv_alpha * tp / ((inv_alpha * tp) + (beta**2 * fp) + fp)