from random import randrange
from math import sqrt
import numpy as np
import os, sys

# local modules 
from data_processor import pipeline_data, set_seed, split_by_class, scale
from evaluate import evaluate_algorithm, calculate_metrics

"""
Biased Random Forest (BRAF)

@ Author: Barnett Chiu

Reference 
---------
1. Diabetes dataset: 

   https://github.com/niharikagulati/diabetesprediction
   https://www.kaggle.com/uciml/pima-indians-diabetes-database

2. Data imputation:
  
   https://scikit-learn.org/stable/auto_examples/impute/plot_iterative_imputer_variants_comparison.html
   https://stackoverflow.com/questions/57154209/implementation-of-sklearn-impute-iterativeimputer

3. Random forest: 

   http://www.codeproject.com/Articles/1197167/Random-Forest-Python
"""
def test_split(index, value, dataset):
    """
    Split a dataset based on an attribute (via index) and an attribute value
    """
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

def gini_index(groups, classes):
    """
    Calculate the Gini index for a split dataset.
    """
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini

def get_split(dataset, n_features):
    """
    Select the best split point for a dataset
    """
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    features = list()
    while len(features) < n_features:
        index = randrange(len(dataset[0])-1) # -1 because the last dimension is reserved for the class label
        if index not in features:
            features.append(index)
    for index in features:
        for row in dataset:
            groups = test_split(index, row[index], dataset) # separate dataset into two groups, one with row[index] 
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

def to_terminal(group, prob=True, pos_label=1):
    """
    Create a terminal node value
    """
    outcomes = [row[-1] for row in group]  
    return outcomes
    
    # if prob: 
    #     # return p(y=1 | x)
    #     return outcomes.count(pos_label)/(len(outcomes)+0.0)

    # return max(set(outcomes), key=outcomes.count)  # majority vote

def split(node, max_depth, min_size, n_features, depth, to_prob=True):
    """
    Create child splits for a node or make terminal

    Params
    ------
    max_depth: the maximum depth of the tree
    min_size: the minimum number of instances required to be at a leaf node 
    n_features: number of features at a split point; consider only this many features at a split point 

    """
    left, right = node['groups']
    del(node['groups'])

    # base case 
    ###########################################
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left+right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    
    # recurse on the left and right subtrees
    ###########################################
    # process left child
    if len(left) <= min_size:  # base case 
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left, n_features)  # get_split() -> {'index':b_index, 'value':b_value, 'groups':b_groups}
        split(node['left'], max_depth, min_size, n_features, depth+1, to_prob=to_prob)
    
    # process right child
    if len(right) <= min_size:  # base case
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth+1, to_prob=to_prob)

def predict(node, row):
    """
    Make a prediction with a decision tree.
    """
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            outcomes = node['left']
            return max(set(outcomes), key=outcomes.count)  # majority vote
            # return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            outcomes = node['right']
            return max(set(outcomes), key=outcomes.count)  # majority vote
            # return node['right']

def predict_proba(node, row, pos_label=1):
    """
    Make a probablistic prediction with a decision tree.
    """
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            outcomes = node['left']
            return outcomes.count(pos_label)/(len(outcomes)+0.0)
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            outcomes = node['right']
            return outcomes.count(pos_label)/(len(outcomes)+0.0)

def subsample(dataset, ratio):
    """
    Create a random subsample from the dataset with replacement
    """
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample

def bagging_predict(trees, row, to_prob=True, pos_label=1):
    """
    Make a prediction with a list of bagged trees
    """
    if to_prob: 
        predictions = [predict_proba(tree, row) for tree in trees]
        return np.mean(predictions)  # mean probability estimates from the trees
    else: 
        predictions = [predict(tree, row) for tree in trees]  # foreach DT, make its prediction on the data point 'row'
        return max(set(predictions), key=predictions.count)

def mixture_predict(trees, row, to_prob=True, pos_label=1): 
    pass

def build_tree(train, max_depth, min_size, n_features, to_prob=True):
    """
    Build a decision tree

    Params
    ------
    train: training data 
    max_depth: the maximum depth of the tree
    min_size: the minimum number of instances required to be at a leaf node 
    n_features: number of features at a split point; consider only this many features at a split point 
    to_prob: if False, the leaf node references a label prediction based on the majority vote
             if True, the leaf node counts the fraction of positive class as a probability estimate 

    """
    root = get_split(train, n_features)
    split(root, max_depth, min_size, n_features, 1, to_prob=to_prob) 
    return root

######################################################
#
#  Random Forest
#
######################################################

def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features, **kargs):
    """
    Random Forest Algorithm. 

    Params
    ------
    """
    verbose = kargs.get('verbose', 1)
    to_prob = kargs.get('to_prob', True) # return probability prediction? 
    tEvaluate = kargs.get("evaluate", True)

    predictions = {} # output may contain predictions as well as other meta data e.g. training scores
    trees = list()
    for i in range(n_trees):
        sample = subsample(train, sample_size)
        tree = build_tree(sample, max_depth, min_size, n_features)
        trees.append(tree)

    predictions['y_pred'] = [bagging_predict(trees, row, to_prob=to_prob) for row in test]

    if tEvaluate:
        # "evaluate" how well the model fit the training data
        y_train = list(np.array(train)[:, -1])  # assuming that class label is at the last column
        y_pred_train = [bagging_predict(trees, row, to_prob=to_prob) for row in train] 
        predictions['y_pred_train'] = y_pred_train
        predictions['scores_train'] = calculate_metrics(y_train, y_pred_train, plot=False, method='RF')
        
    return predictions

def biased_random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features, **kargs):
    """
    BRAF algorithm with a mixture of RF1 (regular RF) and RF2 (RF that focuses on difficult areas determined by kNN)

    Params
    ------
    n_trees: total size of the tree 
    p: the ratio controlling the proportion of RF1 (regular RF) and RF2 (biased RF) 
       in terms of their underlying decision trees
    K: number of neighbors in kNN that determines Tc (critical dataset)

    """
    msg = ""
    verbose = kargs.get('verbose', 1)
    scaling = kargs.get('scaling', True)
    tEvaluate = kargs.get("evaluate", False)

    n_neighbors = kargs.get('K', 10)
    p = kargs.get('p', 0.5)
    s = kargs.get('s', n_trees) 
    n_trees_RF1 = int( np.floor( (1-p) * s ) )
    n_trees_RF2 = s - n_trees_RF1

    T = train
    Tc = create_critical_dataset(T, k=n_neighbors, scaling=scaling) 
    # ... set 'scaling' to True to rescale the feature values to be the same scale, under which euclidean distances are computed

    # [test]
    # ------------------------------------------
    msg += "(biased_rf) size(T): {} vs size(Tc): {}\n".format(len(T), len(Tc))
    msg += "...         Algorithm params | max_depth: {}, min_size: {}, n_features: {}, n_trees(rf1): {}, n_trees(rf2): {}\n".format(
        max_depth, min_size, n_features, n_trees_RF1, n_trees_RF2)
    msg += "...         K={}, p={}, s:{} =?= n_trees:{}\n".format(n_neighbors, p, s, n_trees)
    msg += "...         Feature scaling in kNN? {}\n".format('Yes' if scaling else 'No')
    if verbose: print(msg)
    # ------------------------------------------

    predictions = {} # output may contain predictions as well as other meta data e.g. training scores

    # prediction vector from RF1 purely based on boostrapping 
    P = random_forest(T, test, max_depth, min_size, sample_size, n_trees=n_trees_RF1, n_features=n_features, 
            to_prob=True, evaluate=False, verbose=False)
    
    # prediciton vector from RF2 based on oversampling classificaiton ensemble
    Pc = random_forest(Tc, test, max_depth, min_size, sample_size, n_trees=n_trees_RF2, n_features=n_features, 
            to_prob=True, evaluate=False, verbose=False)

    # ensemble prediction: RF1 + RF2 
    # take the average of prediction vectors as the final prediction
    pv1 = P['y_pred']  
    pv2 = Pc['y_pred'] 
    predictions['y_pred'] = np.mean([pv1, pv2], axis=0).tolist()
    
    # evaluation
    if verbose > 1:  
        print("(biased_rf) pv1: {}".format(pv1[:30]))
        print("(biased_rf) pv2: {}".format(pv2[:30]))

    if tEvaluate: # what happens if the RF were to predict the training set itself? 

        # "evaluate" how well the model fit the training data
        y_train = list(np.array(T)[:, -1])  # assuming that class label is at the last column
        P_train = random_forest(T, T, max_depth, min_size, sample_size, n_trees=n_trees_RF1, n_features=n_features, 
                        to_prob=True, evaluate=False, verbose=False)
        Pc_train = random_forest(Tc, T, max_depth, min_size, sample_size, n_trees=n_trees_RF2, n_features=n_features, 
                        to_prob=True, evaluate=False, verbose=False)
        pvt1 = P_train['y_pred']
        pvt2 = Pc_train['y_pred']
        assert len(pvt1) == len(pvt2), "len(pvt1): {}, len(pvt2): {}".format(len(pvt1), len(pvt2))
        y_pred_train = np.mean([pvt1, pvt2], axis=0).tolist()
        predictions['y_pred_train'] = y_pred_train
        predictions['scores_train'] = calculate_metrics(y_train, y_pred_train, plot=False, method='BRAF')   

    return predictions 

######################################################
#
#  kNN
#
######################################################
def euclidean_distance(row1, row2):
    """
    Calculate the Euclidean distance between two vectors
    """
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)

def create_critical_dataset(train, k=10, scaling=False, col_target=-1, verbose=1):
    """
    Assign for each row in Tmin, k nearest neighoring points from Tmaj (while elimiting duplicate rows).

    Params
    ------
    train: training data (X) assuming that the class label is position at the last column
           if not, need more work to remove the label column when scaling

    """ 
    # from data_processor import scale, split_by_class

    N = len(train)
    T = np.array(train)
    if scaling: # put feature values on the same scale 
        X = T[:, :N-1] # assuming that the class label is at the column
        # np.delete(T, col_target, axis=1) # to remove class label at an arbitrary column

        y = T[:, -1][:, None]  # column vector format
        X = scale(X, scaler='standardize')
        T = np.hstack([X, y])
    # print("(create_critical_dataset) T:\n{}\ndim(train): {}, dim(T): {}".format(T[:5], np.array(train).shape, T.shape))

    # separate miniority and majority classes
    Tmin, Tmaj = split_by_class(T) # rescaled data

    majIDs = set()
    test_cases = np.random.randint(0, len(Tmin), 1) # testing

    # foreach miniority class instance, assign kNNs from majority class 
    for i, row in enumerate(Tmin): 
        neighbors = get_neighbors(Tmaj, row, n_neighbors=k, 
            verify=(verbose > 1) and (i in test_cases))
        majIDs.update(neighbors)

    Tmin, Tmaj = split_by_class(train) # use the original majority class 
    Tc = np.array(Tmaj)[list(majIDs)] # select the kNN rows from the original Tmaj
    Tc = np.vstack([Tmin, Tc]) # add the minority class data back
    np.random.shuffle(Tc)

    if verbose: print("(critical_dataset) kNN-identified n={} unqiue instances in Tmaj => size(Tc)={}".format(len(majIDs), Tc.shape[0]))
    # print("(create_critical_dataset) Tc:\n{}\n".format(Tc[:5]))
    if isinstance(train, list): 
        Tc = Tc.tolist() 
    return Tc

def get_neighbors(train, test_row, n_neighbors, verify=False):
    """
    Locate the most similar neighbors. 
    """
    distances = list()

    for i, train_row in enumerate(train):
        dist = euclidean_distance(test_row, train_row)
        # distances.append((train_row, dist))
        distances.append((i, dist))   # store index into the train_row instead

    distances.sort(key=lambda tup: tup[1])
    if verify: print("(get_neighbors) sorted distances:\n{}\n".format(distances[:10]))

    neighbors = list()
    for i in range(n_neighbors):
        neighbors.append(distances[i][0])  # indices of the k nearest neighbors
    return neighbors

#####################################################################
# driver code 

def demo(input_file='diabetes.csv', input_dir=None, col_target='Outcome'): 
    # from evaluate import evaluate_algorithm

    # Initialize random seed
    set_seed(53)

    if input_dir is None: input_dir = os.getcwd() 
    col_target = 'Outcome'
    tShowTrainingScores = True
    
    # Data imputation (see data_processor.pipeline_data())
    ################################################
    imputation_method = 'median' # {'iter', 'mean', 'median', 'all', 'any', None}
    tImpute = imputation_method is not None
    # Note: this demo assumes that the following attributes can be zero (at least still sensible and therefore not imputed) 
    #
    #       - Pregnancies
    #       - Diabetes Pedigree Function ('DiabetesPedigreeFunction')
    # 
    # But For the following attributes, 0s are considered missing values
    nonzeros_vars = [ 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']
    dataset = pipeline_data(input_file=input_file, input_dir=input_dir, col_target=col_target, 
                impute=tImpute, imputer=imputation_method, target_vars=nonzeros_vars,
                scale=False)

    # print(dataset.head(10))
    dataset = dataset.values.tolist()
    Nv = len(dataset[0])-1  # number of explanatory variables 

    # General RF parameters 
    #########################
    max_depth = 10
    min_size = 1
    sample_size = 1.0

    # n_features = int(np.ceil(sqrt(Nv))) if Nv >= 4 else int(sqrt(Nv))
    n_features = int(sqrt(Nv))  # use only this many features as candidate split points
    ###########################

    # BRAF-specific parameters 
    ###########################
    K = 10 # the k in kNN
    p = 0.5   # ratio for RF2 (biased subsampled data)
    s = 100   # size of the combined forest 
    ########################### 
    
    # Evaluation parameters 
    n_folds = 10
    target_metrics = ['precision', 'recall', 'AUPRC', 'AUROC']

    # Run and evaluate algorithm
    for n_trees in [s, ]:
        perf_metrics, perf_metrics_train = \
                evaluate_algorithm(dataset, biased_random_forest, n_folds,  # data and algorithmic settings
                    max_depth, min_size, sample_size, n_trees, n_features,  # RF parameters
                        K=K, p=p, metrics=target_metrics,   # BRAF-specific parameters
                            evaluate=False, method='BRAF', save_plot=True, verobse=1   # optional params
                            ) 
        
        # print("(demo) Trees: %d" % n_trees)
        for metric, scores in perf_metrics.items(): 
            if tShowTrainingScores: 
                scores_train = perf_metrics_train[metric]
                print("[train] metric: {} | mean: {}, median: {}, std: {}".format(metric, np.mean(scores_train), 
                            np.median(scores_train), np.std(scores_train)))
                print("-" * 100)
            print("[test]  metric: {} | mean: {}, median: {}, std: {}".format(metric, np.mean(scores), 
                        np.median(scores), np.std(scores)))
            print()
            
if __name__ == "__main__": 
    demo(input_file='diabetes.csv')

