import os, sys
import random

from pandas import DataFrame

from csv import reader
import pandas as pd
# from tabulate import tabulate
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def impute(df, **kargs):
    """
    Infer missing values in the data. 
   
    Params
    ------
    null_threshold: a threshold for null values that fall within [0, 1]; 
                    only tolerate at most this fraction of variables assuming NA values

                    For instance, s'pose there 100 variables and null_threshold = 0.2, then 
                    we'll only keep the rows with at least 100 - 100 * 0.2 = 80 non-NA variables 

    """
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer, SimpleImputer
    from sklearn.linear_model import BayesianRidge
    from sklearn.ensemble import ExtraTreesRegressor

    verbose = kargs.get('verbose', 1)
    how = kargs.get('how', 'any')
    N, n_vars = df.shape

    # For the following attributes, 0s are considered missing values
    nonzeros_vars = kargs.get("target_vars", [])
    if len(nonzeros_vars) > 0: 
        df[nonzeros_vars] = df[nonzeros_vars].replace(0, np.nan)
    # for var in nonzeros_vars: 
    #     df.loc[df[var] == 0, var] = np.nan  

    hasNullVals = df.isnull().values.any() 

    msg = ""
    if hasNullVals: 
        col_target = kargs.get('col_target', 'Outcome')  # columns of labels 

        if how.startswith(('iter')):
            
            msg += "(impute) Applying iterative data imputation (using ExtraTreesRegressor() by default)\n"
            # imp = IterativeImputer(max_iter=60, random_state=0)

            X, y, cols_x, cols_y = toXY(df, cols_y=col_target) 
            
            # imputer = IterativeImputer(BayesianRidge())
            imputer = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=20, random_state=53), 
                                missing_values=np.nan, sample_posterior=False, 
                                max_iter=30, tol=0.001, 
                                n_nearest_features=4, initial_strategy='median')

            X = imputer.fit_transform(X)

            df = toDF(X, y, cols_x=cols_x, cols_y=cols_y)   
            assert (df.shape[0] == N) and (df.shape[1] == n_vars)    

        elif how in ('mean', 'median'): # simple imputation (e.g. mean, medican) 
            
            msg += "(impute) Applying simple imputation (taking {})\n".format(how) 
            X, y, cols_x, cols_y = toXY(df, cols_y=col_target) 
            imputer = SimpleImputer(missing_values=np.nan, strategy=how)

            X = imputer.fit_transform(X)
            df = toDF(X, y, cols_x=cols_x, cols_y=cols_y)   

        elif how.startswith('threh'): 
            null_threshold = kargs.get('null_threshold', 0.2) # only tolerate at most this fraction of variables assuming NA values
            # ... e.g. s'pose there 100 variables and null_threshold = 0.2, then 
            # ...      we'll only keep the rows with at least 100 - 100 * 0.2 = 80 non-NA variables 

            n_nonNA = ncols - int(ncols * null_threshold)  # keep only the rows with at least n non-NA values.
            msg += "(impute) keeping only rows with >= {} non-NA values\n".format(n_nonNA)
            df.dropna(thresh=n_nonNA, axis=0, inplace=True) # keep 
            
            msg += "... after dropping rows with predominently null, n_rows: {} -> {} | dim(df): {}\n".format(N, df.shape[0], df.shape) 
        else: 
            msg += "(impute) method: {}\n".format(how)
            df.dropna(how=how, inplace=True) 
            msg += "... after dropping rows with predominently null, n_rows: {} -> {} | dim(df): {}\n".format(how, 
                N, df.shape[0], df.shape)

    else: 
        # noop
        msg += "(impute) No NA found.\n"

    if verbose: print(msg)

    nulls = df.isnull().sum()
    assert np.all(nulls == 0), "(impute) Expecting no NAs after imputation but found: {}".format(np.sum(nulls))

    return df

def scale(X, scaler=None, **kargs):
    from sklearn import preprocessing
    if scaler is None: 
        return X 

    if isinstance(scaler, str): 
        if scaler.startswith(('stand', 'z')): # standardize, z-score
            std_scale = preprocessing.StandardScaler().fit(X)
            X = std_scale.transform(X)
        elif scaler.startswith('minmax'): 
            minmax_scale = preprocessing.MinMaxScaler().fit(X)
            X = minmax_scale.transform(X)
        elif scaler.startswith("norm"): # normalize
            norm = kargs.get('norm', 'l2')
            copy = kargs.get('copy', False)
            X = preprocessing.Normalizer(norm=norm, copy=copy).fit_transform(X)
    else: 
        try: 
            X = scaler.transform(X)
        except Exception as e: 
            msg = "(scale) Invalid scaler: {}".format(e)
            raise ValueError(msg)
    return X

def toDF(X, y, cols_x, cols_y):
    import pandas as pd
    dfX = DataFrame(X, columns=cols_x)
    dfY = DataFrame(y, columns=cols_y)
    return pd.concat([dfX, dfY], axis=1)

def toXY(df, cols_x=[], cols_y=[], untracked=[], **kargs): 
    """
    Convert a dataframe in to the (X, y)-format, where 
       X is an n x m numpy array with n instances and m variables
       y is an n x 1 numpy array, representing class labels

    Params
    ------
    cols_x: explanatory variables

    """
    verbose = kargs.get('verbose', 1)

    # optional operations
    scaler = kargs.pop('scaler', None) # used when scaler is not None (e.g. "standardize")
    
    X = y = None
    if len(untracked) > 0: # untracked variables
        df = df.drop(untracked, axis=1)
    
    if isinstance(cols_y, str): cols_y = [cols_y, ]
    if len(cols_x) > 0:  
        X = df[cols_x].values
        
        cols_y = list(df.drop(cols_x, axis=1).columns)
        y = df[cols_y].values

    else: 
        if len(cols_y) > 0:
            cols_x = list(df.drop(cols_y, axis=1).columns)
            X = df[cols_x].values
            y = df[cols_y].values
        else: 
            if verbose: 
                print("(toXY) Both cols_x and cols_y are empty => Assuming all attributes are variables (n={})".format(df.shape[1]))
            X = df.values
            y = None

    if scaler is not None:
        if verbose: print("(toXY) Scaling X using method:\"{}\"".format(scaler))
        X = scale(X, scaler=scaler, **kargs)
    
    return (X, y, cols_x, cols_y)

def class_prior(L, labels=[0, 1], ratio_ref=0.1, verify=True, verbose=0):  # assuming binary class
    """
    Gather useful summary statistics for class labels.

    Params
    ------
    L: a list or array of class labels 
    ratio_ref: the threshold of imbalanced class dataset. If the size of the minority class falls under this threshold
               (say 0.1), then the dataset is considered as "imbalanced"

    """
    import collections 
    if not labels: labels = np.unique(L)
    lstats = collections.Counter(L)
    ret = {} 
    if len(labels) == 2: # binary class 
        neg_label, pos_label = labels

        ret['n_pos'] = nPos = lstats[pos_label] # np.sum(L==pos_label)
        ret['n_neg'] = nNeg = lstats[neg_label] # np.sum(L==neg_label)
        ret['n_min_class'] = nPos if nPos <= nNeg else nNeg
        ret['n_maj_class'] = nNeg if nPos <= nNeg else nPos
        ret['n'] = nPos + nNeg
        if verify: assert len(L) == ret['n'], "n(labels) do not summed to total (not binary class?)"

        ret['r_pos'] = ret[pos_label] = rPos = nPos/(len(L)+0.0)
        # nNeg = len(L) - nPos 
        ret['r_neg'] = ret[neg_label] = rNeg = nNeg/(len(L)+0.0) # rNeg = 1.0 - rPos
        ret['r_min'] = ret['r_minority'] =  min(rPos, rNeg)
        ret['r_maj'] = ret['r_majority'] = 1. - ret['r_min']
        ret['min_class'] = ret['minority_class'] = pos_label if rPos < rNeg else neg_label
        ret['maj_class'] = ret['majority_class'] = neg_label if rPos < rNeg else pos_label

        ret['r_maj_to_min'] = ret['multiple'] = ret['n_maj_class']/(ret['n_min_class']+0.0)
        
        if min(rPos, rNeg) < ratio_ref:  # assuming pos labels are the minority
            if verbose: print('(class_prior) Imbalanced class distribution: n(+):{0}, n(-):{1}, r(+):{2}, r(-):{3}\n'.format(nPos, nNeg, rPos, rNeg))
            ret['is_balanced'] = False

    else: # single class or multiclass problems
        raise NotImplementedError
    return ret  # keys: n_pos, n_neg, r_pos, r_neg, 0/neg_label, 1/pos_label, is_balanced, r_minority, min_class, maj_class

def load_data0(input_file, **kargs):
    """
    A simpler version of load_data() without using Pandas. 
    """
    input_dir = kargs.get("input_dir", os.getcwd() )
    input_path = os.path.join(input_dir, input_file)
    to_df = kargs.get("to_df", False)

    dataset = list()
    with open(input_path, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)

    # sometimes, it's easier to use dataframe (e.g. data imputation)
    if to_df: dataset = DataFrame(dataset) 
    return dataset

def load_data(input_file, **kargs): 
    import collections

    # I/O paramters
    verbose = kargs.get("verbose", 1)
    input_dir = kargs.get("input_dir", os.getcwd() )
    input_path = os.path.join(input_dir, input_file)
    assert os.path.exists(input_path), "Invalid input path: {}".format(input_path)

    # data parameters
    col_target = kargs.get('col_target', 'Outcome')
    warn_bad_lines = kargs.get('warn_bad_lines', True)
    header = kargs.get('header', 0)
    columns = kargs.get('columns', [])  # only take these columns
    # tImpute = kargs.get('impute', False)
    # impute_method = kargs.get('how', 'any')
    #####################################

    df = pd.read_csv(input_path, sep=',', header=header, index_col=None, error_bad_lines=False, warn_bad_lines=warn_bad_lines)
    if len(columns) > 0: df = df[columns]
    if verbose: 
        print("[load] Loading dataset: {} | dim(df)".format(input_file, df.shape))

    if verbose: 
        X, y, features, _ = toXY(df, cols_y=col_target)
        ret = class_prior(y.flatten(), labels=[0, 1], ratio_ref=0.1, verify=True, verbose=1)

        print("[load] Class distribution ...")
        print("...    dim(X): {}, features (n={}): {}".format(X.shape, len(features), features))
        print("...    n(pos): {}, n(neg): {}, min_class: {}, ratio_min_class: {}\n".format(
            ret['n_pos'], ret['n_neg'], ret['min_class'], ret['r_minority']))

    # if tImpute: 
    #     df = impute(df, how=impute_method, verbose=verbose)

    return df

def pipeline_data(input_file, **kargs): 
    # params 
    col_target = kargs.get('col_target', 'Outcome')
    tImpute = kargs.get('impute', True)
    tScale = kargs.get('scale', False)  # feature scaling is unnecessary (and not recommended) for RF and DT

    # load
    df = load_data(input_file, **kargs)

    # impute 
    if tImpute: 
        imputer = kargs.get('imputer', 'any')
        target_vars = kargs.get("target_vars", []) # only impute these variables
        df = impute(df, how=imputer, target_vars=target_vars)
    
    # scale
    if tScale: 
        scaler = kargs.get('scaler', "standardize")
        X, y, cols_x, cols_y = toXY(df, cols_y=col_target) 
        X = scale(X, scaler=scaler)
        df = toDF(X, y, cols_x, cols_y)
    
    return df

def split_by_class(dataset, col_target=-1):

    N = 0
    if isinstance(dataset, DataFrame):
        df = dataset
        N = df.shape[0]
        assert col_target != -1, "target column not specified"
        y = df[col_target].values
        ret = class_prior(y)

        min_class = ret['min_class']
        maj_class = ret['maj_class']
        Tmin = df.loc[df[col_target == min_class]]
        Tmaj = df.loc[df[col_target == maj_class]]
        assert Tmin.shape[0]+Tmaj.shape[0] == N
    else: # list of lists or numpy array
        X = np.array(dataset)
        N = X.shape[0]
        y = X[:, col_target]  # last column as class label by default
        # print("... y: {}".format(y))
        ret = class_prior(y)

        Tmin = X[y == ret['min_class']]
        Tmaj = X[y == ret['maj_class']]
        assert Tmin.shape[0]+Tmaj.shape[0] == N

        if isinstance(dataset, list): 
            Tmin = Tmin.tolist()
            Tmaj = Tmaj.tolist()
    
    return Tmin, Tmaj

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

def test(): 
    set_seed(53)
    input_dir = os.path.join(os.getcwd(), 'data')
    input_file = 'diabetes.csv'
    col_target = "Outcome"
    imputation_method = 'median' # {'iter', 'mean', 'median', 'all', 'any', None}
    tImpute = imputation_method is not None

    # For the following attributes, 0s are considered missing values
    nonzeros_vars = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age'] # 'Pregnancies', 'DiabetesPedigreeFunction',

    df = load_data(input_file=input_file, input_dir=input_dir, verbose=1)
    print("(test) No data imputation:\n{}\n".format(df.head(100)))

    df = pipeline_data(input_file=input_file, input_dir=input_dir, col_target=col_target, 
            impute=tImpute, imputer=imputation_method, target_vars=nonzeros_vars,
            scale=False)
    print("(test) Data imputation applied:\n{}\n".format(df.head(100)))

    hasNullVals = df.isnull().values.any()
    assert not hasNullVals

    return

if __name__ == "__main__":
    test()
