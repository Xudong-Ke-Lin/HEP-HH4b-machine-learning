# imports
import uproot
import pandas as pd
import tensorflow as tf
from sklearn import tree
from hh4b_utils.nnt_tools import load_nnt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def get_data(file_path_data_16,file_path_data_17,file_path_data_18,region,mc=False):
    '''Open the files and import the data from years 2016, 2017 and 2018
    Calculates the NN_weights using the get_mu() function
    Inputs:
            file_path_data16,17,18: path of the data
            region: sig, validation or control
                    if None, import all three
            mc: if these are mc files, default False
    Outputs:
            data16,17,18: dataset of a specific year
    '''
    # if mc file, flatten is false
    if mc==True:
        flatten=False
    else:
        flatten=True

    # if None, import all three regions 
    if region==None:
        data16 = load_nnt(file_path_data_16, flatten=flatten)
        data17 = load_nnt(file_path_data_17, flatten=flatten)
        data18 = load_nnt(file_path_data_18, flatten=flatten)
    # specific region
    else:
        data16 = load_nnt(file_path_data_16, trees=[region], flatten=flatten)
        data17 = load_nnt(file_path_data_17, trees= [region], flatten=flatten)
        data18 = load_nnt(file_path_data_18, trees = [region], flatten=flatten)

    f_data16 = uproot.open(file_path_data_16)
    f_data17 = uproot.open(file_path_data_17)
    f_data18 = uproot.open(file_path_data_18)
    
    # add year column
    data16['year'] = 16
    data17['year'] = 17
    data18['year'] = 18
    
    # initial maks
    data16 = data16.loc[data16['X_wt_tag']>=1.5].reset_index(drop=True)
    data17 = data17.loc[data17['X_wt_tag']>=1.5].reset_index(drop=True)
    data18 = data18.loc[data18['X_wt_tag']>=1.5].reset_index(drop=True)

    data16 = data16[~data16['pass_vbf_sel']].reset_index(drop=True)
    data17 = data17[~data17['pass_vbf_sel']].reset_index(drop=True)
    data18 = data18[~data18['pass_vbf_sel']].reset_index(drop=True)
    
    # add NN_weights column if mc=False
    if mc==False:
        # calculate norm
        norm_16 = get_mu(f_data16, 16)
        norm_17 = get_mu(f_data17, 17)
        norm_18 = get_mu(f_data18, 18)

        data16['NN_weights'] = norm_16 * data16['NN_d24_weight_bstrap_med_16']
        data17['NN_weights'] = norm_17 * data17['NN_d24_weight_bstrap_med_17']
        data18['NN_weights'] = norm_18 * data18['NN_d24_weight_bstrap_med_18']
    
    return data16,data17,data18

def get_data_mask(data16,data17,data18,mask='2bRW'):
    '''apply mask to the data
    also add sample_weight and class columns
    note: this function could be appended to get_data(), but we may want to have 
    different masks for the same files and not load the files everytime,
    e.g. 2bRW and 4b masks in control region
    inputs:
            data16,17,18: outputs of the get_data() function
            mask: 2bRW or 4b, default=2bRW
    outputs:
            df: dataset after specific masks and concatenate all three years data
    '''
    # concatenate data
    data_all = pd.concat([data16,data17,data18], ignore_index=True)
    if mask=='2bRW':
        df = data_all.loc[(data_all["ntag"] == 2) & (data_all["rw_to_4b"] == True)].reset_index(drop=True) 
        # background weights and class
        df['sample_weight'] = df['NN_weights']
        df['class'] = 0
    if mask=='4b':
        df = data_all.loc[data_all['ntag']>=4].reset_index(drop=True)
        # signal weights and class
        df['sample_weight'] = 1
        df['class'] = 1
        
    return df

def get_mu(file, year: int = 16, vr: bool = False) -> float:
    """get nominal norm value from NNT"""
    vr_fix = "_VRderiv" if vr else ""
    return file[f"NN_norm{vr_fix}_bstrap_med_{year}"].member("fVal")

def build_model(hp):
    '''Deep neural network model used as input in the KerasTuner
    used for hyperparameter tuning
    input: hp is the hyperparameter variable in KerasTuner
    output: turnable model
    '''
    # sequential model
    model = Sequential()
    # number of layers is tunable 
    # each layer contains a dense and a dropout
    for i in range(hp.Int("num_layers", 2, 5, default=3)):
        # add dense layer
        model.add(
            Dense(
                # number of units is tunable , from 50 to 500
                units=hp.Int("units_" + str(i), min_value=50, max_value=500, step=50),
                # activation function is tunable , default relu
                activation=hp.Choice('act_' + str(i), ['relu', 'tanh'], default='relu')
            )
        )
        # add dropout layer
        model.add(
            Dropout(
                # rate is tunable, 0.0 meaning no dropout
                hp.Choice("rate_" + str(i), [0.0, 0.1, 0.2, 0.4])
            )
        )
    # add output dense layer; activation function is tunable, default is softmax
    model.add(Dense(2, activation=hp.Choice('act_output', ['sigmoid', 'softmax'], default='softmax')))
    model.compile(
        # learning rate is tunable, default 0.001
        optimizer=tf.keras.optimizers.Adam(hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4], default=1e-3)),
        loss="categorical_crossentropy",
        # use F1_Score() class as metric
        metrics=[F1_Score()],
    )
    return model

class F1_Score(tf.keras.metrics.Metric):
    '''f1 score metric used in TensorFlow or KerasTuner
    '''
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.f1 = self.add_weight(name='f1', initializer='zeros')
        self.precision_fn = tf.keras.metrics.Precision(thresholds=0.5)
        self.recall_fn = tf.keras.metrics.Recall(thresholds=0.5)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # precision and recall
        p = self.precision_fn(y_true[:,1], y_pred[:,1])
        r = self.recall_fn(y_true[:,1], y_pred[:,1])
        # since f1 is a variable, we use assign
        self.f1.assign(2 * ((p * r) / (p + r + 1e-6)))

    def result(self):
        return self.f1

    def reset_state(self):
        # we also need to reset the state of the precision and recall objects
        self.precision_fn.reset_state()
        self.recall_fn.reset_state()
        self.f1.assign(0)