from __future__ import print_function

import shap
import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Reshape, Dropout, ZeroPadding1D
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
import phenograph
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import statistics
import SimpleITK as sitk
from myshow import myshow
from scipy.stats import norm
from PIL import Image
import os 
import glob
from keras import models
from keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
import pandas as pd
import re
import string
import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
import nltk
import seaborn as sns
import scipy
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import random
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

def dataLoad(rawData):                      
  # # Remove cells that are unknown
  rawData = rawData[rawData['Cell Type'] != "Unknown"]

  # Normalize data between 0 and 1
  normRows = rawData.columns[0:-6]
  for i in normRows:
      max_value = rawData[i].max()
      min_value = rawData[i].min()
      rawData[i] = (rawData[i] - min_value) / (max_value - min_value)

  max_value = rawData["New Cluster Label"].max()
  min_value = rawData["New Cluster Label"].min()
  rawData["New Cluster Label"] = (rawData["New Cluster Label"] - min_value) / (max_value - min_value)

  rawData["Patient"] = ""
  
  # ROI 3
  rawData['Patient'][rawData['ROI'] == 'ROI005_ROI_005_3']= "64"
  rawData['Patient'][rawData['ROI'] == 'ROI006_ROI_006_3']= "64"
  rawData['Patient'][rawData['ROI'] == 'ROI007_ROI_007_3']= "61"
  rawData['Patient'][rawData['ROI'] == 'ROI008_ROI_008_3']= "61"
  rawData['Patient'][rawData['ROI'] == 'ROI009_ROI_009_3']= "60"
  rawData['Patient'][rawData['ROI'] == 'ROI010_ROI_010_3']= "60"
  rawData['Patient'][rawData['ROI'] == 'ROI011_ROI_011_3']= "59"
  rawData['Patient'][rawData['ROI'] == 'ROI012_ROI_012_3']= "59"
  rawData['Patient'][rawData['ROI'] == 'ROI013_ROI_013_3']= "56"
  rawData['Patient'][rawData['ROI'] == 'ROI014_ROI_014_3']= "56"
  rawData['Patient'][rawData['ROI'] == 'ROI015_ROI_015_3']= "55"
  rawData['Patient'][rawData['ROI'] == 'ROI016_ROI_016_3']= "55"
  rawData['Patient'][rawData['ROI'] == 'ROI017_ROI_017_3']= "54"
  rawData['Patient'][rawData['ROI'] == 'ROI023_ROI_023_3']= "45"
  rawData['Patient'][rawData['ROI'] == 'ROI024_ROI_024_3']= "45"
  rawData['Patient'][rawData['ROI'] == 'ROI027_ROI_027_3']= "33"
  rawData['Patient'][rawData['ROI'] == 'ROI028_ROI_028_3']= "33"
  rawData['Patient'][rawData['ROI'] == 'ROI029_ROI_029_3']= "32"
  rawData['Patient'][rawData['ROI'] == 'ROI030_ROI_030_3']= "32"
  rawData['Patient'][rawData['ROI'] == 'ROI033_ROI_033_3']= "23"
  rawData['Patient'][rawData['ROI'] == 'ROI034_ROI_034_3']= "23"
  rawData['Patient'][rawData['ROI'] == 'ROI035_ROI_035_3']= "22"
  rawData['Patient'][rawData['ROI'] == 'ROI036_ROI_036_3']= "22"
  rawData['Patient'][rawData['ROI'] == 'ROI037_ROI_037_3']= "19"
  rawData['Patient'][rawData['ROI'] == 'ROI038_ROI_038_3']= "19"
  rawData['Patient'][rawData['ROI'] == 'ROI041_ROI_041_3']= "8"
  rawData['Patient'][rawData['ROI'] == 'ROI042_ROI_042_3']= "8"
  rawData['Patient'][rawData['ROI'] == 'ROI043_ROI_043_3']= "6"
  rawData['Patient'][rawData['ROI'] == 'ROI044_ROI_044_3']= "6"

  # # Combining into groups identified by Tiziana
  rawData.loc[rawData["Cell Type"].isin(["CD163 Macrophage", "B Cells", "CD4 T cell", "CD8 T cell", "Dendritic Cell", 
                                         "Granzyme B Immune Cell", "Granzyme B+ Macrophage", "Immune cell", "Macrophage", 
                                         "T regulatory Cell"]), "Cell Type"] = "Immune"
  rawData.loc[rawData["Cell Type"].isin(["Actin+"]), "Cell Type"] = "Stroma"
  rawData.loc[rawData["Cell Type"].isin(["Granzyme B Tumour", "Proliferating Tumour"]), "Cell Type"] = "Tumour"
  rawData.loc[rawData["Cell Type"].isin(["Granzyme B+", "Proliferating Cell"]), "Cell Type"] = "Other"

  # Encode cell types into integers
  cellTypes = list(np.unique(rawData["Cell Type"]))

  # Encode cell type strings to integer
  from sklearn.preprocessing import LabelEncoder
  le = LabelEncoder()
  rawData["Cell Type"] = le.fit_transform(rawData["Cell Type"])

  rawData = rawData[['141Pr_141Pr_alpha-actin.ome.tiff',
                     '142Nd_142Nd_CD66b.ome.tiff', '143Nd_143Nd-Vimentin.ome.tiff',
                     '144Nd_144Nd-CD14.ome.tiff', '147Sm_147Sm-CD163.ome.tiff', '148Nd_148Nd-PANCK.ome.tiff',
                     '149Sm_149Sm-CD11b.ome.tiff', '151Eu_151Eu-GATA3.ome.tiff',
                     '152Sm_152Sm-CD45.ome.tiff', '154Sm_154SmCD366TIM3.ome.tiff',
                     '155Gd_155Gd-FOXP3.ome.tiff',
                     '156Gd_156Gd-CD4.ome.tiff', '158Gd_158Gd-CD11c.ome.tiff',
                     '159Tb_159Tb-CD68.ome.tiff', '161Dy_161Dy-CD20.ome.tiff',
                     '162Dy_162Dy-CD8a.ome.tiff', 
                     '165Ho_165Ho-PD1.ome.tiff', 
                     '167Er_167Er-GRANZB.ome.tiff', '168Er_168Er-KI67.ome.tiff',
                     '169Tm_169Tm-DCLAMP.ome.tiff',
                     '170Er_170Er-CD3.ome.tiff', 
                     '174Yb_174Yb-HLA-DR.ome.tiff', 
                     "CYSvTUR", "Cell Type", "ROI", "Patient", "Cell Label"]] #'173Yb_173Yb-CD45RO.ome.tiff', 

  # Add zero padding
  rawData["Zeros1"] = 0
  rawData["Zeros2"] = 0

  return rawData

def trainTestSplit(rawData):
  labelData = rawData["Cell Type"]
  x_trainDf, x_testDf, y_train, y_test = train_test_split(rawData, labelData, test_size=0.33, random_state=42, stratify=labelData)
  x_trainDf = x_trainDf.drop(["ROI", "Cell Type", "CYSvTUR", "Patient", "Cell Label"],1)
  # Balance datasets
  x_trainDf, y_train = SMOTE().fit_resample(x_trainDf, y_train)
  # Make validation set
  x_trainDf, x_validDf, y_train, y_valid = train_test_split(x_trainDf, y_train, test_size=0.2, random_state=42, stratify=y_train)
  # Convert to arrays
  x_train = np.asarray(x_trainDf).reshape(x_trainDf.shape[0], x_trainDf.shape[1], 1)
  x_test = x_testDf.drop(["ROI", "Cell Type", "CYSvTUR", "Patient", "Cell Label"],1)
  x_test = np.asarray(x_test).reshape(x_test.shape[0], x_test.shape[1], 1)
  x_valid = np.asarray(x_validDf).reshape(x_validDf.shape[0], x_validDf.shape[1], 1)
  
  return x_trainDf, x_testDf, x_validDf, x_train, x_test, x_valid, y_train, y_test, y_valid

def cellCounts(data):
  cellts = le.inverse_transform(data["Cell Type"])
  unique, counts = np.unique(cellts, return_counts=True)
  cellcounts = dict(zip(unique, counts))
  return cellcounts

def convaeClassifier(x_train, x_test, x_valid, y_train, y_test, y_valid):
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior() 
  
  from tensorflow.keras.callbacks import EarlyStopping

  adam = tf.keras.optimizers.Adam(learning_rate=0.00005)
  adamLow = tf.keras.optimizers.Adam(learning_rate=0.00001)
  bottleneck_size = 5 # number of dimensions to view latent space in; same as "code size"
  nFeatures = x_train.shape[1]
  input_ae = Input(shape=(nFeatures, 1))

  model = None

  x = Conv1D(16, 3, activation="relu", padding="same")(input_ae)
  x = Conv1D(16, 3, activation="relu", padding="same")(x)
  x = MaxPooling1D()(x)
  x = Conv1D(32, 3, activation="relu", padding="same")(x)
  x = Conv1D(32, 3, activation="relu", padding="same")(x)
  x = Dropout(0.7)(x)
  encoded = MaxPooling1D()(x)

  x = Conv1D(32, 3, activation="relu", padding="same")(x)
  x = Conv1D(32, 3, activation="relu", padding="same")(x)
  x = UpSampling1D()(x)
  x = Conv1D(16, 3, activation="relu", padding="same")(x)
  x = Conv1D(16, 3, activation="relu", padding="same")(x)
  decoded = Conv1D(1, 3, activation='sigmoid', padding='same', name='ae_out')(x)

  y = Flatten()(encoded)
  y = Dense(64, activation = 'relu')(y)
  y = Dense(24, activation = 'relu')(y)
  output_class = Dense(nClasses, activation = 'softmax', name = "class_out")(y)

  ae_alone = Model(input_ae, decoded)
  ae_joint = Model(inputs=[input_ae], outputs=[decoded, output_class])
  ae_alone.compile(optimizer=adamLow, loss='mean_squared_error', metrics=['mse'])
  


def main():
  rawData = pd.read_csv("C:\\Users\\sindhura\\OneDrive - Queen's University\\Queens\\Research MSc\\Bladder Data\\newPhenographData.csv")
  rawData = dataLoad(rawData)
  nClasses = len(np.unique(rawData["Cell Type"]))
  x_trainDf, x_testDf, x_validDf, x_train, x_test, x_valid, y_train, y_test, y_valid = trainTestSplit(rawData)
  
  
  














