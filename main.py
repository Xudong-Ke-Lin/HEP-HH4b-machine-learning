################################################################################
################################################################################
# main script for hyperparameter tuning
# this script may take some hours, use linux screen
# uses functions from utils.py, standard libraries and some hep libraries
# may want to change the project_name in tuner and the best model name
################################################################################
################################################################################

# each tuner should have its onw project directory
project_directory = "/mnt/storage/zcapxke/keras_tuner/this_is_final_project"
best_model_name = 'this_is_final_best_model'

# avoids running on GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# hep imports
import mplhep as hep
hep.style.use('ATLAS')
from hh4b_utils.nnt_tools import load_nnt

# standard libraries imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# tensorflow imports
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner import BayesianOptimization, Objective

# import utils.py
import utils

# mc files
file_path_mc_16 = "/mnt/storage/zcapxke/data/sm_hh_pythia_mc16a.root"
file_path_mc_17 = "/mnt/storage/zcapxke/data/sm_hh_pythia_mc16d.root"
file_path_mc_18 = "/mnt/storage/zcapxke/data/sm_hh_pythia_mc16e.root"
mc16,mc17,mc18=utils.get_data(file_path_mc_16,file_path_mc_17,file_path_mc_18,region='sig',mc=True)

# data files
file_path_data_16 = "/mnt/storage/zcapxke/data/data16_Xhh_45_NN_100_bootstraps.root"
file_path_data_17 = "/mnt/storage/zcapxke/data/data17_Xhh_45_NN_100_bootstraps.root"
file_path_data_18 = "/mnt/storage/zcapxke/data/data18_Xhh_45_NN_100_bootstraps.root"
data16,data17,data18=utils.get_data(file_path_data_16,file_path_data_17,file_path_data_18,region='sig')

# apply masks
signal_df = utils.get_data_mask(mc16,mc17,mc18,mask='4b')
bkg_df = utils.get_data_mask(data16,data17,data18,mask='2bRW')

# excluding some features that did not agree well in CR
features = ['m_hh','X_hh','dEta_hh','njets','X_wt_tag','cos_theta_star',
           'pt_hh','pT_2','pT_4','eta_i','dRjj_1','dRjj_2','m_min_dj','m_max_dj',
          'pairing_score_1','pairing_score_2','bkt_lead_jet_pt','bkt_third_lead_jet_pt',
          'm_h1','E_h1','pT_h1','eta_h1','phi_h1','m_h2','E_h2','pT_h2','eta_h2','phi_h2',
          'm_h1_j1','E_h1_j1','eta_h1_j1','phi_h1_j1',
           'm_h1_j2','E_h1_j2','eta_h1_j2','phi_h1_j2',
          'm_h2_j1','E_h2_j1','eta_h2_j1','phi_h2_j1',
           'm_h2_j2','E_h2_j2','eta_h2_j2','phi_h2_j2','year'] 

# final dataset
df_data = pd.concat([signal_df, bkg_df], ignore_index=True)
X = df_data[features]
y = df_data['class']
idx = df_data.index
weights= df_data['sample_weight']

# train 70% and test 30% of the dataset
(
    X_train,
    X_val,
    y_train,
    y_val,
    weights_train,
    weights_val,
    idx_train,
    idx_val,
) = train_test_split(X, y, weights, list(idx), test_size=0.3)

# scale X
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc = scaler.transform(X_val)
# convert y to binary class matrix
y_train_hot = to_categorical(y_train)
y_val_hot = to_categorical(y_val)

# weights for classes
N_bkg_train = weights_train[y_train == 0].sum()
N_sig_train = weights_train[y_train==1].sum()
# ratio of the weights
R = N_bkg_train / N_sig_train
# use this ratio for signal events
weights_train_R = np.copy(weights_train)
weights_train_R[y_train==1] = R

# use Bayesian Optimization from KerasTuner
# information on the parameters https://keras.io/api/keras_tuner/tuners/bayesian/
tuner = BayesianOptimization(
    utils.build_model,
    objective=Objective('val_f1_score', direction='max'),
    metrics=[utils.F1_Score()],
    max_trials=100,
    executions_per_trial=3,
    alpha=0.0001,
    beta=4,
    # overwrite the project folder
    overwrite=True,
    # project folder name
    project_name=project_directory
)

# search for optimal hyperparameters
tuner.search(
    X_train_sc,
    y_train_hot,
    sample_weight=weights_train_R,
    epochs=100,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=15),
    ],
    batch_size=1000,
    validation_data=(X_val_sc, y_val_hot, weights_val),
)

# save best model
best_model = tuner.get_best_models()[0]
best_model.build(X_train_sc.shape)
best_model.save(best_model_name)