#################################################################################
################################################################################
# script for plotting NN socres in shifted (control) regions
# includes comparison of 2bRW, 2bRW with weights and 4b
# this script may take some time, use linux screen
# uses functions from utils.py, standard libraries and some hep libraries
################################################################################
################################################################################ 

#  avoids running on GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# hep imports
import mplhep as hep
hep.style.use('ATLAS')
import matplotlib.pyplot as plt 

# standard libraries imports
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# import utils.py
import utils

# load best model
best_model = tf.keras.models.load_model('this_is_final_best_model', custom_objects={'F1_Score':utils.F1_Score})

# load scales to re-weight 2bRW
scales = np.loadtxt('weights.txt')

# excluding some features that did not agree well in CR
features = ['m_hh','X_hh','dEta_hh','njets','X_wt_tag','cos_theta_star',
           'pt_hh','pT_2','pT_4','eta_i','dRjj_1','dRjj_2','m_min_dj','m_max_dj',
          'pairing_score_1','pairing_score_2','bkt_lead_jet_pt','bkt_third_lead_jet_pt',
          'm_h1','E_h1','pT_h1','eta_h1','phi_h1','m_h2','E_h2','pT_h2','eta_h2','phi_h2',
          'm_h1_j1','E_h1_j1','eta_h1_j1','phi_h1_j1',
           'm_h1_j2','E_h1_j2','eta_h1_j2','phi_h1_j2',
          'm_h2_j1','E_h2_j1','eta_h2_j1','phi_h2_j1',
           'm_h2_j2','E_h2_j2','eta_h2_j2','phi_h2_j2','year'] 

# load scaler
scaler = StandardScaler()

# calculate bins
bins = np.linspace(0.0,1.0,21)

# create figures: fig1 normal scale, fig2 log scale
fig1 = plt.figure(figsize=(15,8))
fig2 = plt.figure(figsize=(15,8))

# for interested regions
for i, region in enumerate(['center_right','lower_right','upper_center','upper_left','upper_right']):
    # file paths
    file_path_data_16 = f"/mnt/storage/zcapxke/data/shifted/{region}/data16_NN_100_bootstraps.root"
    file_path_data_17 = f"/mnt/storage/zcapxke/data/shifted/{region}/data17_NN_100_bootstraps.root"
    file_path_data_18 = f"/mnt/storage/zcapxke/data/shifted/{region}/data18_NN_100_bootstraps.root"
    
    # use control region
    data16,data17,data18=utils.get_data(file_path_data_16,file_path_data_17,file_path_data_18,'control')
    # apply masks
    bkg_2b_df=utils.get_data_mask(data16,data17,data18,mask='2bRW')
    bkg_4b_df=utils.get_data_mask(data16,data17,data18,mask='4b')
    # get inputs for the model
    X_2b = bkg_2b_df[features]
    X_4b = bkg_4b_df[features]
    # get weights for 2bRW
    weights_2b = bkg_2b_df['sample_weight']
    
    # scale data
    X_2b_sc = scaler.fit_transform(X_2b)
    X_4b_sc = scaler.fit_transform(X_4b)

    # calculate predictions 
    pred_test_2b = best_model.predict(X_2b_sc)[:,1]
    pred_test_4b = best_model.predict(X_4b_sc)[:,1]
    
    # normal scale plot
    ax1 = fig1.add_subplot(2, 3, i+1)
    # 2b RW
    h1,_,_ = ax1.hist(pred_test_2b, bins = bins, histtype='step',label='2b RW', density= True,
                   weights=weights_2b, color ='r')
    # 2b Rw with scales
    _=ax1.hist(bins[:-1],bins=bins,weights=h1*scales, histtype='step',density=True,color='m',label='2b RW scaled',
        linestyle='dashed')
    # 4b
    h2,_,_ = ax1.hist(pred_test_4b, bins = bins, histtype='step', label = '4b', density = True, color='b')
    ax1.set_title(region, size=10)
    ax1.tick_params(axis='x', labelsize=10)
    ax1.tick_params(axis='y', labelsize=10)
    ax1.yaxis.offsetText.set_fontsize(10)
    ax1.legend(loc='best', prop={'size': 10})
    
    # log scale plot
    ax2 = fig2.add_subplot(2, 3, i+1)
    # 2b RW
    h1,_,_ = ax2.hist(pred_test_2b, bins = bins, histtype='step',label='2b RW', density= True,
                   weights=weights_2b, color='r', log=True)
    # 2b Rw with scales
    _=ax2.hist(bins[:-1],bins=bins,weights=h1*scales, histtype='step',density=True,color='m',label='2b RW scaled',
         linestyle='dashed', log=True)
    # 4b
    h2,_,_ = ax2.hist(pred_test_4b, bins = bins, histtype='step', label = '4b', density = True, color='b', log=True)
    ax2.set_title(region, size=10)
    ax2.tick_params(axis='x', labelsize=10)
    ax2.tick_params(axis='y', labelsize=10)
    ax2.yaxis.offsetText.set_fontsize(10)
    ax2.legend(loc='best', prop={'size': 10})

# save figures
fig1.savefig('shifted_plots.png')
fig2.savefig('shifted_plots_log.png')