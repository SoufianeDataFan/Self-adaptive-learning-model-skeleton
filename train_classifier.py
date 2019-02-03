
# -----------------------------------------------------------------------------------------
    # Created by Soufiane CHAMI      # Network Disruptions 
# ---------------------------------------------------------------------------------------



import os 
import pickle
from contextlib import contextmanager
import pandas as pd 
import numpy as np
import datetime
import gc
import time

# Machine learning

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from scipy.stats import uniform as sp_rand
from scipy.stats import randint as sp_randint

import warnings
warnings.filterwarnings("ignore")
    
    
    
# ---------------------------------------------------------------
# Data_Preparation 
#----------------------------------------------------------------

print(os.getcwd())  # check current working directory for debug 





# --------------------
# Useful functions 
# --------------------


@contextmanager


def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))
    

def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df  #, new_columns



# -------------------
# Data Prep
# -------------------

def get_resource_type(encode=False):
    resource_type= pd.read_csv( 'resource_type.csv', error_bad_lines=False, warn_bad_lines=False)
    resource_type.id=resource_type.id.astype(object)
    
    if(encode):
        resource_type['resource_type']= resource_type['resource_type'].str.replace('resource_type ','')
        cat_cols=['resource_type']
        non_cat_cols = [f for f in resource_type.columns.values if f not in cat_cols]
        resource_type.drop(columns=cat_cols)
        
        df = one_hot_encoder(resource_type[cat_cols])
        resource_type = pd.concat([resource_type[non_cat_cols],df], axis=1, sort=False)
    
    
    return resource_type



def get_severity_type(encode=False):

    severity_type=pd.read_csv( 'severity_type.csv', error_bad_lines=False, warn_bad_lines=False)
    severity_type.id=severity_type.id.astype(object)
    if(encode):
        severity_type['severity_type']= severity_type['severity_type'].str.replace('severity_type ','')

        cat_cols=['severity_type']
        non_cat_cols = [f for f in severity_type.columns.values if f not in cat_cols]

        df = one_hot_encoder(severity_type[cat_cols])
        severity_type = pd.concat([severity_type[non_cat_cols],df], axis=1, sort=False)

    return severity_type



def get_train(encode_Target=False):
    train = pd.read_csv('train.csv', error_bad_lines=False, warn_bad_lines=False)
    train.id=train.id.astype(object)
#     train.fault_severity=train.fault_severity.astype('int')
    train['location']= train['location'].str.replace('location ','')
    train.location=train.location.astype(int)
    train.columns.values[train.columns=='fault_severity']='Target_class'
    if(encode_Target):
        cat_cols=['Target_class']
        non_cat_cols = [f for f in train.columns.values if f not in cat_cols]
        train.drop(columns=cat_cols)
        df = one_hot_encoder(train[cat_cols])
        train = pd.concat([train[non_cat_cols],df], axis=1, sort=False)
        return train
    
    else: # means dont touch the fault_severity and location columns, Just return well formated train file
        return train
        
        

def get_test(encode=False):
    test = pd.read_csv('test.csv', error_bad_lines=False, warn_bad_lines=False)
    test.id=test.id.astype(object)
    test['location']= test['location'].str.replace('location ','')
    test.location=test.location.astype(int)
    test['Target_class'] = None
    return test




def get_event_type(encode =False):
    event_type= pd.read_csv('event_type.csv', error_bad_lines=False, warn_bad_lines=False)
    event_type.id=event_type.id.astype(object)
    
    if(encode):
        event_type['event_type']= event_type['event_type'].str.replace('event_type ','')

        cat_cols=['event_type']
        non_cat_cols = [f for f in event_type.columns.values if f not in cat_cols]

        df = one_hot_encoder(event_type[cat_cols])
        event_type = pd.concat([event_type[non_cat_cols],df], axis=1, sort=False)

    return event_type


# ------------------------------------------------------
#    Data binning on the log_features cat features 
# ------------------------------------------------------


def get_log_feature(encode =False):
    log_feature = pd.read_csv('log_feature.csv', error_bad_lines=False, warn_bad_lines=False)
    log_feature.id=log_feature.id.astype(object)
    log_feature['log_feature']= log_feature['log_feature'].str.replace('feature ','')
    log_feature['log_feature']= log_feature['log_feature'].astype(int)
    
    log_feature.reset_index(inplace=True)
    log_feature.rename(columns={'index':'count_of_log_feature_seen'},inplace=True)
    log_feature_value_counts = log_feature.log_feature.value_counts().to_dict()
    log_feature['count_log_feature_per_ids'] = log_feature['log_feature'].map(lambda x: log_feature_value_counts[x])
    
    len_col=len(set(log_feature['count_log_feature_per_ids']))
    max_col=max(log_feature['count_log_feature_per_ids'] )
    bin_max = int(max_col/len_col)+1
    bins= [f*10 for f in range(-1, 41)]
    log_feature['binned_log_feature'] = np.digitize(log_feature['log_feature'], bins, right=True)
    
    bins_offset = list(map(lambda x:x+5, bins))
    log_feature['binned_offset_log_feature'] = np.digitize(log_feature['log_feature'], bins_offset, right=True)
    
    log_feature['position_of_log_feature'] = 1
    log_feature['position_of_log_feature'] = log_feature.groupby(['id'])['position_of_log_feature'].cumsum()
    log_feature['log_feature'] = log_feature['log_feature'].astype(int)

    return log_feature

# same thing can be applied on the other tables ¯\(ツ)/¯



# -----------------------------------------------------------------------------
# returns the training, validation  and testing datasets
# -----------------------------------------------------------------------------



def get_data(encode=False):
    event_type    = get_event_type()
    resource_type = get_resource_type(encode=encode)
    severity_type = get_severity_type(encode=encode)
    log_feature   = get_log_feature()
    train         = get_train()
    test          = get_test()
    temp_combined = pd.concat([train, test], axis=0,ignore_index=True)
    log_combined = pd.merge(temp_combined,log_feature,left_on = ['id'], right_on = ['id'],how='left')
    log_combined = pd.merge(log_combined,resource_type,left_on = ['id'], right_on = ['id'],how='left')
    log_combined = pd.merge(log_combined,severity_type,left_on = ['id'], right_on = ['id'],how='left')
    
    test = log_combined.loc[log_combined.Target_class.isin(list(set(log_combined.Target_class))[-1:])] # contain None
    train= log_combined.loc[log_combined.Target_class.isin(list(set(log_combined.Target_class))[:-1])] # not None 
    train.Target_class= train.Target_class.astype('int')
    y_train= train['Target_class']


    # split data 
    print("Spliting the train/val ..")
    
    SEED = 1235
    VALID_SIZE = 0.30
    
    x_train= train.loc[:, ~train.columns.isin(['Target_class', 'id'])]
    
    print('Data shapes ...') # control shapes 
    print(x_train.shape)
    print(y_train.shape)
    
    # split data into train and test sets
    X_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size=VALID_SIZE, random_state=SEED)
    return X_train, X_val, y_train, y_val , test




# -----------------------------------------------------------------------------
# returns xgboost model with parameters as inputs 
# -----------------------------------------------------------------------------



def score(params):
    print("Training with params: ")
    print(params)
    num_round = int(params['n_estimators'])
    del params['n_estimators']
    X_train, X_val, y_train, y_val , test= get_data(encode=True)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_val, label=y_val)
    watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    gbm_model = xgb.train(params, dtrain, num_round,
                          evals=watchlist,
                          verbose_eval=True)
    predictions = gbm_model.predict(dvalid,
                                    ntree_limit=gbm_model.best_iteration + 1)
    loss = log_loss(y_val, predictions)
    # TODO: Add the importance for the selected features
    print("\tScore {0}\n\n".format(loss))
    # The score function should return the loss (1-score)
    # since the optimize function looks for the minimum
    
    return {'loss': loss, 'status': STATUS_OK}




# -----------------------------------------------------------------------------
# Sefl tuning: get the best model parameters within the search space 
# -----------------------------------------------------------------------------


def optimize(random_state=1235):
    """
    Hyperparams tuning with Random Optimization 
    """
    
    space = {    # space of params tuning 
        'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
        'learning_rate':  hp.choice('learning_rate', 0.001*np.arange(5, 100, dtype=int)),
        'eta': hp.quniform('eta', 0.025, 0.5,0.025),
        'max_depth':  hp.choice('max_depth', np.arange(1, 14, dtype=int)),
        'min_child_weight': hp.quniform('min_child_weight', 1, 20, 1),
        'subsample': hp.quniform('subsample', 0.1, 1, 0.05),
        'max_delta_step': hp.choice('max_delta_step', np.arange(0,5, dtype=int)),
        'gamma': hp.choice('gamma', 0.1*np.arange(0,25, dtype=int)),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1, 0.05),
        'eval_metric': 'mlogloss',
        'objective': 'multi:softprob',
        'booster': 'gbtree',
        'tree_method':'gpu_hist', 
        'predictor':'gpu_predictor',
        'silent': 1,
        'num_class': 3,
        'seed': 1235}
     
    
    # Use the fmin function from Hyperopt to find the best hyperparameters
    best = fmin(score, space, algo=tpe.suggest, max_evals=250)
    
    space.update(best)
    
    return space   # return parameters only 



# ---------------------------------------------------------------------------------------------------------------------
# train xgboost model with tuning parameters or with default ones (chosen by me after the first tuning iteration)
# ---------------------------------------------------------------------------------------------------------------------


def train_single_classifier(params):   # train "optimal" model after hyperparams tuning 
    
    print("Training with params: ")
    print(params)
    num_round = int(params['n_estimators'])
    del params['n_estimators']
    X_train, X_val, y_train, y_val , test = get_data(INPUT_DIR)
    
    #convert to format to inject it to the xgboost
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_val, label=y_val)
    watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    gbm_model = xgb.train(params, dtrain, num_round,
                          evals=watchlist,
                          verbose_eval=True)
    #predict 
    predictions = gbm_model.predict(dvalid,
                                    ntree_limit=gbm_model.best_iteration + 1)
    
    #score 
    
    score = log_loss(y_val, predictions)
    print("Model \tScore {0}\n\n".format(score))
    
    # save model 
    today= datetime.datetime.now()
    model_file_name ='signle_xgboost_'+today.strftime('__%H:%M:%S_%d_%b_%Y_')+"Score__" +str(float("%0.4f"%score))+'.pickle.dat'
    pickle.dump(gbm_model, open(model_file_name, "wb"))
    print("\n")
    print("Model is save in this directory : " + os.getcwd())
    print("\n")
    print("Model file_name : " + model_file_name)

    # make _sbumission 
    
    
    # prepare test dataset
    
    x_test= test.loc[:, ~test.columns.isin(['Target_class', 'id'])]
    x_test =xgb.DMatrix(x_test)
    predictions_test = gbm_model.predict(x_test)
    cols= ['predict_0', 'predict_1', 'predict_2']
    predictions_test= pd.DataFrame(predictions_test, columns=cols)
    predictions_test['id']= test.id.values
    predictions_test= predictions_test.groupby(['id'], axis=0).mean().reset_index()
    
    
    # save submission 
    print("\n")
    submission_file = 'submission'+today.strftime('__%H:%M:%S_%d_%b_%Y_')+'_score_'+str(float("%0.4f"%score))+'.csv'
    predictions_test.to_csv(submission_file, index=False)
    print("submission is made in this directory : " + os.getcwd() + "\n")
    print("submission file_name : " + submission_file+ "\n")
    
    
    return None        #gbm_model, os.getcwd()





# -----------------------------------------------------------------------------
# Run full model, with prams_tuning (True) or with defined params  
# -----------------------------------------------------------------------------


def main(self_tuning = False):
    X_train, X_val, y_train, y_val, test= get_data(True)
    if(self_tuning):
        with timer("XGboost: Paramters tuning with Random optimization"):
            best_hyperparams = optimize(random_state=1235)
            print("The best hyperparameters are: ", "\n")
            print(best_hyperparams)
    else: 
        print("Run XGboost without hyper_params search")

    with timer("train single Model and make test submission"):
        
        try:
            best_hyperparams # will be defined if yo run params tuning ! 
        except NameError:
            
            # I added those params as trial, I got them from one of the tuning iterations ietrations I had 
            best_hyperparams= {'booster': 'gbtree',
                               'colsample_bytree': 0.95,
                               'eta': 0.225,
                               'gamma': 0.85,
                               'max_depth': 6,
                               'min_child_weight': 5.0,
                               'n_estimators': 112.0,
                               'subsample': 0.95,
                               'eval_metric': 'mlogloss',
                               'num_class': 3,
                               'objective': 'multi:softprob',
                               'seed': 1235,
                               'silent': 1}

            
        # train classifier with best_Hyperparamerters 
        train_single_classifier(params= best_hyperparams)
        
        
        gc.collect()


if __name__ == "__main__":
    INPUT_DIR='/path/to/directory/' # directory of the inputs data 
    os.chdir(INPUT_DIR)
    with timer("Full model run"):
        main(self_tuning= True)
        
        
