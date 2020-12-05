import numpy as np
import os
from random import randrange

import sklearn.metrics
from sklearn.metrics import precision_recall_curve, f1_score, roc_auc_score, auc

import matplotlib
matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
from matplotlib import pyplot as plt
from utils_plot import saveFig

def evaluate_algorithm(dataset, algorithm, n_folds, *args, **kargs):
    """
    Evaluate an algorithm using a cross validation split. 

    Params
    ------
    *args: argument list for the input algorithm (e.g. random_forest(), biased_random_forest())
    **kargs: named arguments for the input algorithm
    """
    from utils_plot import plot_roc, highlight
    
    # parameters
    metrics = kargs.pop('metrics', ['precision', 'recall', 'AUPRC', 'AUROC'])
    verbose = kargs.get("verbose", 1)
    output_dir = kargs.pop("output_dir", 'plot')
    save_plot = kargs.pop("save_plot", True)
    algo_name = kargs.pop("method", 'BRAF')

    folds = cross_validation_split(dataset, n_folds)
    
    scores = {metric:[] for metric in metrics}  # list()
    scores_train = {metric:[] for metric in metrics}

    cv_data = []
    for i, fold in enumerate(folds):
        fold_number = i+1
        if verbose: highlight(message="[cross validation] Fold ({})".format(fold_number), symbol='#', border=1)
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        
        kargs['verbose'] = 1 if i == 0 else 0
        kargs['fold_number'] = fold_number
        #######################################################
        predictions = algorithm(train_set, test_set, *args, **kargs)
        y_pred = predictions['y_pred']
        y_true = [row[-1] for row in fold]
        cv_data.append((y_true, y_pred))
        #######################################################

        # performance metrics on the test data
        predictions['scores'] = calculate_metrics(y_true, y_pred, plot=save_plot, 
            method=algo_name, fold=fold_number, phase='test')        

        for metric in metrics: 
            if metric in predictions['scores']: 
                scores[metric].append(predictions['scores'][metric])

        # keep track of training set scores as well 
        if 'scores_train' in predictions: 
            for metric in metrics: 
                if metric in predictions['scores_train']:
                    scores_train[metric].append(predictions['scores_train'][metric]) 

    # [note]
    # The ROC curve within each CV fold can be combined while still maintaining visual clarity
    # However, it's not the case for the PR curve, and therefore, I've decided to 
    # plot the PR curve seperately for each CV fold.
    if save_plot: 
        plot_roc(cv_data, output_dir=output_dir, method=algo_name)

    return scores, scores_train

def cross_validation_split(dataset, n_folds):
    """
    Split a dataset into k folds
    """
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

def to_label_prediction(y_score, p_th=0.5):
    # turn probabilities into label prediction given threshold at 'p_th'
    yhat = np.zeros(len(y_score))
    for i, yp in enumerate(y_score): 
        if yp >= p_th: 
            yhat[i] = 1
    return yhat

def calculate_accuracy(y_true, y_hat):
    """
    Calculate accuracy percentage  
    """
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_hat[i]:
            correct += 1
    return correct / float(len(y_true))

def eval_AUPRC(y_true, y_score, method='BRAF', plot=True, **kargs): 
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auprc = auc(recall, precision)

    if plot:  # plot the precision-recall curves
        plt.clf()
        fold_number = kargs.get('fold', 0)

        y_true = np.array(y_true)
        no_skill = len(y_true[y_true==1]) / len(y_true)

        plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
        plt.plot(recall, precision, marker='.', label=method)
        # axis labels
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # show the legend
        plt .legend()
        
        # show the plot
        # plt.show()
        phase = kargs.get("phase", 'test') 
        prefix = 'precision_recall_curve-train' if phase.startswith('tr') else 'precision_recall_curve'
        filename = kargs.get('filename', '{}-{}'.format(prefix, fold_number) if fold_number > 0 else prefix)
        output_dir = kargs.get("output_dir", "plot")
        output_path = os.path.join(output_dir, filename)  # example path: System.analysisPath
        saveFig(plt, output_path, ext='tif', dpi=300, message='[output] precision recall curve', verbose=True)
    return auprc

def eval_AUROC(y_true, y_score, method='BRAF', plot=False, **kargs):
    auc = roc_auc_score(y_true, y_score)

    if plot: 
        pass # use utils_plot.plot_roc() instead

    # Note: ROC curve in each CV fold can be easily combined while still maintaining visual clarity, 
    #       and therefore, the plotting is deferred until all CV data are collected (see evaluate_algorithm())

    return auc

def calculate_metrics(y_true, y_score, p_th=0.5, **kargs):
    """
    Calculate performance metrics and plot related performance curves (e.g. precision-recall curve). 

    Params
    ------
    y_true: true labels 
    y_score: class conditional probabilities P(y=1|x)
    method: the name of the algorithm (for display only)
    phase: 'train' for the training phase or 'test' for the test phase 
           use this to make distinction between the plot associaetd with the training or test
           
           Most use cases only generate performance plot on the test data and therefore 
           the file name for the plot does not have this keyword. 

           If, however, we wish to diagnose overfitting by comparing the performance gap 
           between the training phase and the test phase, the performance plot can be 
           generated accordingly but with the keyword 'train' added to the plots' file names. 


    """
    # optional plot params
    index = kargs.get("fold", 0)
    plot = kargs.get("plot", True)
    method = kargs.get("method", 'BRAF')
    phase = kargs.get("phase", 'test') # 

    metrics = {}
    metrics['AUROC'] = eval_AUROC(y_true, y_score, fold=index)
    metrics['AUPRC'] = eval_AUPRC(y_true, y_score, fold=index, plot=plot, method=method, phase=phase)

    y_hat = to_label_prediction(y_score, p_th=p_th)          
    # ret['f1'] = f1_score(y_true, y_hat)
    
    nTP = nTN = nFP = nFN = 0
    for i, y in enumerate(y_true): 
        if y == 1: 
            if y_hat[i] == 1: 
                nTP += 1
            else: 
                nFN += 1
        else: # y == 0 
            if y_hat[i] == 0: 
                nTN += 1 
            else: 
                nFP += 1
    metrics['precision'] = nTP/(nTP+nFP+0.0)
    metrics['recall'] = nTP/(nTP+nFN+0.0)
    metrics['accuracy'] = calculate_accuracy(y_true, y_hat)

    return metrics


def perturb(X, cols_x=[], cols_y=[], lower_bound=0, alpha=100.):
    def add_noise():
        min_nonnegative = np.min(X[np.where(X>lower_bound)])
        
        Eps = np.random.uniform(min_nonnegative/(alpha*10), min_nonnegative/alpha, X.shape)

        return X + Eps
    # from pandas import DataFrame

    if isinstance(X, DataFrame):
        from data_processor import toXY
        X, y, fset, lset = toXY(X, cols_x=cols_x, cols_y=cols_y, scaler=None, perturb=False)
        X = add_noise(X)
        dfX = DataFrame(X, columns=fset)
        dfY = DataFrame(y, columns=lset)
        return pd.concat([dfX, dfY], axis=1)

    X = add_noise()
    return X

def fmax_score(labels, predictions, beta = 1.0, pos_label = 1):
    """
        Radivojac, P. et al. (2013). A Large-Scale Evaluation of Computational Protein Function Prediction. Nature Methods, 10(3), 221-227.
        Manning, C. D. et al. (2008). Evaluation in Information Retrieval. In Introduction to Information Retrieval. Cambridge University Press.

    Memo
    ---- 
    1. precision and recall tradeoff doesn't take into account true negative

    """
    # import sklearn.metrics

    precision, recall, threshold = sklearn.metrics.precision_recall_curve(labels, predictions, pos_label)

    # the general formula for positive beta
    f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)

    # if beta == 1, then this is just f1 score, harmonic mean between precision and recall 
    # i = np.nanargmax(f1)

    # return (f1[i], threshold[i])
    return nanmax(f1)

def fmax_score_threshold(labels, predictions, beta = 1.0, pos_label = 1):
    """
    Return the fmax score and the probability threhold where the max of f1 (fmax) is reached
    """
    # import sklearn.metrics
    precision, recall, threshold = sklearn.metrics.precision_recall_curve(labels, predictions, pos_label)

    # the general formula for positive beta
    # ... if beta == 1, then this is just f1 score, harmonic mean between precision and recall 
    f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    i = np.nanargmax(f1)  # the position for which f1 is the max 
    th = threshold[i] if i < len(threshold) else 1.0    # len(threshold) == len(precision) -1 
    # assert f1[i] == nanmax(f1)
    return (f1[i], th)

