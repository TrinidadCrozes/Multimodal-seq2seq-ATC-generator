import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../..')))
from seq2seq import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns

from sklearn import preprocessing
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split 
from sklearn.metrics import precision_recall_curve, average_precision_score, multilabel_confusion_matrix, classification_report

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

import random
import itertools

def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)                
    os.environ["PYTHONHASHSEED"] = str(seed) 
    os.environ["TF_DETERMINISTIC_OPS"] = "1" 
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1" 
# Convert a string that simulates a list to a real list
def convert_string_list(element):
    # Delete [] of the string
    element = element[1:len(element)-1]
    # Create a list that contains each code as e.g. 'A'
    ATC_list = list(element.split(', '))
    for index, code in enumerate(ATC_list):
        # Delete '' of the code
        ATC_list[index] = code[1:len(code)-1]
    return ATC_list

def train_test_1(seed):
    train_set = pd.read_csv(f'../Datasets/SplitATC_Rep_train_set{seed}.csv')
    val_set = pd.read_csv(f'../Datasets/SplitATC_Rep_val_set{seed}.csv')
    # test_set = pd.read_csv(f'../Datasets/SplitATC_Rep_test_set.csv')
    # train_set = pd.concat([train_set, val_set], ignore_index=True)
    # Delete unnecessary columns from train set
    train_set.drop('Neutralized SMILES', axis = 1, inplace = True)
    train_set.drop('ATC Codes', axis = 1, inplace = True)
    train_set.drop('ATC_level2', axis = 1, inplace = True)
    train_set.drop('ATC_level3', axis = 1, inplace = True)
    train_set.drop('ATC_level4', axis = 1, inplace = True)
    train_set.drop('multiple_ATC', axis = 1, inplace = True)
    train_set = train_set.reset_index(drop=True)
    # Delete unnecessary columns from test set
    val_set.drop('Neutralized SMILES', axis = 1, inplace = True)
    val_set.drop('ATC Codes', axis = 1, inplace = True)
    val_set.drop('ATC_level2', axis = 1, inplace = True)
    val_set.drop('ATC_level3', axis = 1, inplace = True)
    val_set.drop('ATC_level4', axis = 1, inplace = True)
    val_set.drop('multiple_ATC', axis = 1, inplace = True)
    test_set = val_set.reset_index(drop=True)
    # Divide in X and y
    X_train = train_set.drop('ATC_level1', axis = 1)
    y_train = train_set['ATC_level1']
    X_test = test_set.drop('ATC_level1', axis = 1)
    y_test = test_set['ATC_level1']
    return X_train, y_train, X_test, y_test
def train_test_2(seed):
    train_set = pd.read_csv(f'../Datasets/SplitATC_Rep_train_set{seed}.csv')
    val_set = pd.read_csv(f'../Datasets/SplitATC_Rep_val_set{seed}.csv')
    # test_set = pd.read_csv(f'../Datasets/SplitATC_Rep_test_set.csv')
    # train_set = pd.concat([train_set, val_set], ignore_index=True)
    # Delete unnecessary columns 
    train_set.drop('Neutralized SMILES', axis = 1, inplace = True)
    train_set.drop('ATC Codes', axis = 1, inplace = True)
    train_set.drop('ATC_level3', axis = 1, inplace = True)
    train_set.drop('ATC_level4', axis = 1, inplace = True)
    train_set.drop('multiple_ATC', axis = 1, inplace = True)
    train_set = train_set.reset_index(drop=True)
    # Delete unnecessary columns 
    val_set.drop('Neutralized SMILES', axis = 1, inplace = True)
    val_set.drop('ATC Codes', axis = 1, inplace = True)
    val_set.drop('ATC_level3', axis = 1, inplace = True)
    val_set.drop('ATC_level4', axis = 1, inplace = True)
    val_set.drop('multiple_ATC', axis = 1, inplace = True)
    test_set = val_set.reset_index(drop=True)
    # Replicate compounds that have more than 1 ATC level 1 code
    new_rows = []

    for _, row in train_set.iterrows():
        ATC_level1_list = convert_string_list(row['ATC_level1'])
        for code in ATC_level1_list:
            new_row = row.copy()
            new_row['ATC_level1'] = code
            new_rows.append(new_row)

    new_train_set = pd.DataFrame(new_rows)
    new_train_set = new_train_set.reset_index(drop=True)
    # Delete level 1 letter from ATC_level2
    new_rows = [] 
    for _, row in new_train_set.iterrows():
        ATC_level2_list = convert_string_list(row['ATC_level2'])
        # Split ATC code if they have more than 1 code at level 2
        for code in ATC_level2_list:
            if code[0] == row['ATC_level1']:
                new_row = row.copy()
                new_row['ATC_level2'] = code[1:len(code)]
                new_rows.append(new_row)

    new_train_set2 = pd.DataFrame(new_rows)
    new_train_set2 = new_train_set2.reset_index(drop=True)
    
    new_test_set2 = test_set
    
    X_train = new_train_set2.drop('ATC_level2', axis = 1)
    y_train = new_train_set2['ATC_level2']
    X_test = new_test_set2.drop('ATC_level2', axis = 1)
    y_test = new_test_set2['ATC_level2']
    
    return X_train, y_train, X_test, y_test
def train_test_3(seed):
    train_set = pd.read_csv(f'../Datasets/SplitATC_Rep_train_set{seed}.csv')
    val_set = pd.read_csv(f'../Datasets/SplitATC_Rep_val_set{seed}.csv')
    # test_set = pd.read_csv(f'../Datasets/SplitATC_Rep_test_set.csv')
    # train_set = pd.concat([train_set, val_set], ignore_index=True)
    # Delete unnecessary columns 
    train_set.drop('Neutralized SMILES', axis = 1, inplace = True)
    train_set.drop('ATC Codes', axis = 1, inplace = True)
    train_set.drop('ATC_level1', axis = 1, inplace = True)
    train_set.drop('ATC_level4', axis = 1, inplace = True)
    train_set.drop('multiple_ATC', axis = 1, inplace = True)
    train_set = train_set.reset_index(drop=True)
    # Delete unnecessary columns 
    val_set.drop('Neutralized SMILES', axis = 1, inplace = True)
    val_set.drop('ATC Codes', axis = 1, inplace = True)
    val_set.drop('ATC_level1', axis = 1, inplace = True)
    val_set.drop('ATC_level4', axis = 1, inplace = True)
    val_set.drop('multiple_ATC', axis = 1, inplace = True)
    test_set = val_set.reset_index(drop=True)
    # Replicate compounds that have more than 1 ATC code
    new_rows = []

    for _, row in train_set.iterrows():
        ATC_level2_list = convert_string_list(row['ATC_level2'])
        for code in ATC_level2_list:
            new_row = row.copy()
            new_row['ATC_level2'] = code
            new_rows.append(new_row)

    new_train_set = pd.DataFrame(new_rows)
    new_train_set = new_train_set.reset_index(drop=True)
    # Delete level 1 letter from ATC_level2
    new_rows = [] 
    for _, row in new_train_set.iterrows():
        ATC_level3_list = convert_string_list(row['ATC_level3'])
        # Split ATC code if they have more than 1 code at level 2
        for code in ATC_level3_list:
            if code[0:3] == row['ATC_level2']:
                new_row = row.copy()
                new_row['ATC_level3'] = code[3:len(code)]
                new_rows.append(new_row)

    new_train_set2 = pd.DataFrame(new_rows)
    new_train_set2 = new_train_set2.reset_index(drop=True)

    new_test_set2 = test_set

    X_train = new_train_set2.drop('ATC_level3', axis = 1)
    y_train = new_train_set2['ATC_level3']
    X_test = new_test_set2.drop('ATC_level3', axis = 1)
    y_test = new_test_set2['ATC_level3']
    
    return X_train, y_train, X_test, y_test
def train_test_4(seed):
    train_set = pd.read_csv(f'../Datasets/SplitATC_Rep_train_set{seed}.csv')
    val_set = pd.read_csv(f'../Datasets/SplitATC_Rep_val_set{seed}.csv')
    # test_set = pd.read_csv(f'../Datasets/SplitATC_Rep_test_set.csv')
    # train_set = pd.concat([train_set, val_set], ignore_index=True)
    # Delete unnecessary columns 
    train_set.drop('Neutralized SMILES', axis = 1, inplace = True)
    train_set.drop('ATC Codes', axis = 1, inplace = True)
    train_set.drop('ATC_level1', axis = 1, inplace = True)
    train_set.drop('ATC_level2', axis = 1, inplace = True)
    train_set.drop('multiple_ATC', axis = 1, inplace = True)
    train_set = train_set.reset_index(drop=True)
    # Delete unnecessary columns 
    val_set.drop('Neutralized SMILES', axis = 1, inplace = True)
    val_set.drop('ATC Codes', axis = 1, inplace = True)
    val_set.drop('ATC_level1', axis = 1, inplace = True)
    val_set.drop('ATC_level2', axis = 1, inplace = True)
    val_set.drop('multiple_ATC', axis = 1, inplace = True)
    test_set = val_set.reset_index(drop=True)
    # Replicate compounds that have more than 1 ATC code
    new_rows = []

    for _, row in train_set.iterrows():
        ATC_level3_list = convert_string_list(row['ATC_level3'])
        for code in ATC_level3_list:
            new_row = row.copy()
            new_row['ATC_level3'] = code
            new_rows.append(new_row)

    new_train_set = pd.DataFrame(new_rows)
    new_train_set = new_train_set.reset_index(drop=True)
    # Delete level 1 letter from ATC_level2
    new_rows = [] 
    for _, row in new_train_set.iterrows():
        ATC_level4_list = convert_string_list(row['ATC_level4'])
        # Split ATC code if they have more than 1 code at level 2
        for code in ATC_level4_list:
            if code[0:4] == row['ATC_level3']:
                if code[4:len(code)] != '':
                    new_row = row.copy()
                    new_row['ATC_level4'] = code[4:len(code)]
                    new_rows.append(new_row)

    new_train_set2 = pd.DataFrame(new_rows)
    new_train_set2 = new_train_set2.reset_index(drop=True)

    new_test_set2 = test_set

    X_train = new_train_set2.drop('ATC_level4', axis = 1)
    y_train = new_train_set2['ATC_level4']
    X_test = new_test_set2.drop('ATC_level4', axis = 1)
    y_test = new_test_set2['ATC_level4']
    
    return X_train, y_train, X_test, y_test

def get_X_train_level2(X_train2, y_train2):
    # Get all available labels describing the level 1 ATC code
    labels2 = set()
    for code in y_train2:
        labels2.add(code)
            
    labels2 = sorted(list(labels2))
    y_labels_encoder2 = MultiLabelBinarizer()
    y_labels_encoder2.fit([labels2])
    encoded_y_train2 = y_labels_encoder2.transform(y_train2.values.reshape(-1, 1))
    
    atc_level1_labels2 = set()
    for _, row in X_train2.iterrows():
        atc_level1_labels2.add(row['ATC_level1'])
            
    atc_level1_labels2 = sorted(list(atc_level1_labels2))
    atc_level1_labels_encoder2 = MultiLabelBinarizer()
    atc_level1_labels_encoder2.fit([atc_level1_labels2])
    ATC_level11 = X_train2['ATC_level1']
    ATC_level1_2 = ATC_level11.copy()
    for index, lista in enumerate(ATC_level11):
        ATC_level1_2[index] = []
        ATC_level1_2[index].append(lista)
    categorical_atc2 = atc_level1_labels_encoder2.transform(ATC_level1_2)
    X_train2.drop(labels=['ATC_level1'], axis="columns", inplace=True)
    df_level1_2 = pd.DataFrame(categorical_atc2, columns=atc_level1_labels2)
    X_train2 = pd.concat([X_train2, df_level1_2], axis = 1)
    X_train2 = np.asarray(X_train2).astype(np.float32)
    encoded_y_train2 = np.asarray(encoded_y_train2).astype(np.float32)
    # Complete NaN values in each column with the median
    X_train2[pd.isna(X_train2)] = np.nanmedian(X_train2)
    # Define an instance of the MinMaxScaler
    scaler2 = MinMaxScaler()
    # Fit the scaler to the data and transform it
    X_train2 = scaler2.fit_transform(X_train2)
    return X_train2, encoded_y_train2, labels2, y_labels_encoder2, atc_level1_labels_encoder2, scaler2

def get_X_train_level3(X_train3, y_train3, X_test3):
    # Get all available labels describing the level 1 ATC code
    labels3 = set()
    for code in y_train3:
        labels3.add(code)
            
    labels3 = sorted(list(labels3))
    y_labels_encoder3 = MultiLabelBinarizer()
    y_labels_encoder3.fit([labels3])
    encoded_y_train3 = y_labels_encoder3.transform(y_train3.values.reshape(-1, 1))
    
    atc_level1_labels3 = set()
    atc_level2_labels3 = set()
    for _, row in X_train3.iterrows():
        atc_level1_labels3.add(row['ATC_level2'][0:1])
        atc_level2_labels3.add(row['ATC_level2'][1:3])
    for _, row in X_test3.iterrows():
        lista = convert_string_list(row['ATC_level2'])
        for code in lista:
            atc_level1_labels3.add(code[0:1])
            atc_level2_labels3.add(code[1:3])
            
    atc_level1_labels3 = sorted(list(atc_level1_labels3))
    atc_level2_labels3 = sorted(list(atc_level2_labels3))
    atc_level1_labels_encoder3 = MultiLabelBinarizer()
    atc_level1_labels_encoder3.fit([atc_level1_labels3])
    atc_level2_labels_encoder3 = MultiLabelBinarizer()
    atc_level2_labels_encoder3.fit([atc_level2_labels3])
    ATC_level22 = X_train3['ATC_level2']
    ATC_level1_3 = ATC_level22.copy()
    ATC_level2_3 = ATC_level22.copy()
    for index, lista in enumerate(ATC_level22):
        ATC_level1_3[index] = []
        ATC_level1_3[index].append(lista[0:1])
    for index, lista in enumerate(ATC_level22):
        ATC_level2_3[index] = []
        ATC_level2_3[index].append(lista[1:3])
    X_train3.drop(labels=['ATC_level2'], axis="columns", inplace=True)
    categorical_atc1_3 = atc_level1_labels_encoder3.transform(ATC_level1_3)
    categorical_atc2_3 = atc_level2_labels_encoder3.transform(ATC_level2_3)
    df_level1_3 = pd.DataFrame(categorical_atc1_3, columns=atc_level1_labels3)
    df_level2_3 = pd.DataFrame(categorical_atc2_3, columns=atc_level2_labels3)
    X_train3 = pd.concat([X_train3, df_level1_3, df_level2_3], axis = 1)
    X_train3 = np.asarray(X_train3).astype(np.float32)
    encoded_y_train3 = np.asarray(encoded_y_train3).astype(np.float32)
    # Complete NaN values in each column with the median
    X_train3[pd.isna(X_train3)] = np.nanmedian(X_train3)
    # Define an instance of the MinMaxScaler
    scaler3 = MinMaxScaler()
    # Fit the scaler to the data and transform it
    X_train3 = scaler3.fit_transform(X_train3)
    return X_train3, encoded_y_train3, labels3, y_labels_encoder3, atc_level1_labels_encoder3, atc_level2_labels_encoder3, scaler3

def get_X_train_level4(X_train4, y_train4, X_test4):
    # Get all available labels describing the level 1 ATC code
    labels4 = set()
    for code in y_train4:
        labels4.add(code)
            
    labels4 = sorted(list(labels4))
    y_labels_encoder4 = MultiLabelBinarizer()
    y_labels_encoder4.fit([labels4])
    encoded_y_train4 = y_labels_encoder4.transform(y_train4.values.reshape(-1, 1))
    
    atc_level1_labels4 = set()
    atc_level2_labels4 = set()
    atc_level3_labels4 = set()
    for _, row in X_train4.iterrows():
        atc_level1_labels4.add(row['ATC_level3'][0:1])
        atc_level2_labels4.add(row['ATC_level3'][1:3])
        atc_level3_labels4.add(row['ATC_level3'][3:4])
    for _, row in X_test4.iterrows():
        lista = convert_string_list(row['ATC_level3'])
        for code in lista:
            atc_level1_labels4.add(code[0:1])
            atc_level2_labels4.add(code[1:3])
            atc_level3_labels4.add(code[3:4])
    
    atc_level1_labels4 = sorted(list(atc_level1_labels4))
    atc_level2_labels4 = sorted(list(atc_level2_labels4))
    atc_level3_labels4 = sorted(list(atc_level3_labels4))
    atc_level1_labels_encoder4 = MultiLabelBinarizer()
    atc_level1_labels_encoder4.fit([atc_level1_labels4])
    atc_level2_labels_encoder4 = MultiLabelBinarizer()
    atc_level2_labels_encoder4.fit([atc_level2_labels4])
    atc_level3_labels_encoder4 = MultiLabelBinarizer()
    atc_level3_labels_encoder4.fit([atc_level3_labels4])
    ATC_level33 = X_train4['ATC_level3']
    ATC_level1_4 = ATC_level33.copy()
    for index, lista in enumerate(ATC_level33):
        ATC_level1_4[index] = []
        ATC_level1_4[index].append(lista[0:1])
    ATC_level2_4 = ATC_level33.copy()
    for index, lista in enumerate(ATC_level33):
        ATC_level2_4[index] = []
        ATC_level2_4[index].append(lista[1:3])
    ATC_level3_4 = ATC_level33.copy()
    for index, lista in enumerate(ATC_level33):
        ATC_level3_4[index] = []
        ATC_level3_4[index].append(lista[3:4])
    X_train4.drop(labels=['ATC_level3'], axis="columns", inplace=True)
    categorical_atc1_4 = atc_level1_labels_encoder4.transform(ATC_level1_4)
    categorical_atc2_4 = atc_level2_labels_encoder4.transform(ATC_level2_4)
    categorical_atc3_4 = atc_level3_labels_encoder4.transform(ATC_level3_4)
    df_level1_4 = pd.DataFrame(categorical_atc1_4, columns=atc_level1_labels4)
    df_level2_4 = pd.DataFrame(categorical_atc2_4, columns=atc_level2_labels4)
    df_level3_4 = pd.DataFrame(categorical_atc3_4, columns=atc_level3_labels4)
    X_train4 = pd.concat([X_train4, df_level1_4, df_level2_4, df_level3_4], axis = 1)
    X_train4 = np.asarray(X_train4).astype(np.float32)
    encoded_y_train4 = np.asarray(encoded_y_train4).astype(np.float32)
    # Complete NaN values in each column with the median
    X_train4[pd.isna(X_train4)] = np.nanmedian(X_train4)
    # Define an instance of the MinMaxScaler
    scaler4 = MinMaxScaler()
    # Fit the scaler to the data and transform it
    X_train4 = scaler4.fit_transform(X_train4)

    return X_train4, encoded_y_train4, labels4, y_labels_encoder4, atc_level1_labels_encoder4, atc_level2_labels_encoder4, atc_level3_labels_encoder4, scaler4

def test_set_level1(scaler1, X_test):
    X_test = np.asarray(X_test).astype(np.float32)
    X_test[pd.isna(X_test)] = np.nanmedian(X_test)
    X_test = scaler1.transform(X_test)
    return X_test
def test_set_level2(scaler2, X_test1, pred_df_level1, atc_level1_labels_encoder2):
    X_test1.drop(labels=['ATC_level1'], axis="columns", inplace=True)
    X_test1['ATC_level1'] = pred_df_level1['pred_1']
    ATC_level1 = X_test1['ATC_level1']
    categorical_atc = atc_level1_labels_encoder2.transform(ATC_level1)
    df_level1 = pd.DataFrame(categorical_atc, columns=atc_level1_labels_encoder2.classes_)
    X_test1 = pd.concat([X_test1, df_level1], axis = 1)
    X_test1.drop(labels=['ATC_level1'], axis="columns", inplace=True)

    X_test1 = np.asarray(X_test1).astype(np.float32)
    X_test1[pd.isna(X_test1)] = np.nanmedian(X_test1)
    X_test1 = scaler2.transform(X_test1)

    return X_test1
def test_set_level3(scaler3, X_test1, pred_df_level1, pred_df_level2, atc_level1_labels_encoder3, atc_level2_labels_encoder3):
    predicted_codes = []
    
    for index, pred1 in enumerate(pred_df_level1['pred_1']):
        pred2 = str(pred_df_level2.at[pred_df_level2.index[index], 'pred_2']).zfill(2)
        prediction = pred1 + '' + pred2
        predicted_codes.append(prediction)
        
    X_test1['ATC_level2'] = predicted_codes
    
    ATC_level22 = X_test1['ATC_level2']
    ATC_level1_3 = ATC_level22.copy()
    ATC_level2_3 = ATC_level22.copy()
    for index, atc in enumerate(ATC_level22):
        ATC_level1_3[index] = []
        ATC_level1_3[index].append(atc[0:1])
        ATC_level2_3[index] = []
        ATC_level2_3[index].append(atc[1:3])

    X_test1 = X_test1.drop(labels=['ATC_level2'], axis="columns")
    
    categorical_atc1_3 = atc_level1_labels_encoder3.transform(ATC_level1_3)
    categorical_atc2_3 = atc_level2_labels_encoder3.transform(ATC_level2_3)
    df_level1_3 = pd.DataFrame(categorical_atc1_3, columns=atc_level1_labels_encoder3.classes_)
    df_level2_3 = pd.DataFrame(categorical_atc2_3, columns=atc_level2_labels_encoder3.classes_)
    X_test1 = pd.concat([X_test1, df_level1_3, df_level2_3], axis = 1)                         
    
    X_test1 = np.asarray(X_test1).astype(np.float32)
    X_test1[pd.isna(X_test1)] = np.nanmedian(X_test1)
    X_test1 = scaler3.transform(X_test1)

    return X_test1
def test_set_level4(scaler4, X_test1, pred_df_level1, pred_df_level2, pred_df_level3, atc_level1_labels_encoder4, atc_level2_labels_encoder4, atc_level3_labels_encoder4):
    predicted_codes = []
    
    for index, pred1 in enumerate(pred_df_level1['pred_1']):
        pred2 = str(pred_df_level2.at[pred_df_level2.index[index], 'pred_2']).zfill(2)
        pred3 = pred_df_level3.at[pred_df_level3.index[index], 'pred_3']
        prediction = pred1 + '' + pred2 + '' + pred3
        predicted_codes.append(prediction)
        
    X_test1['ATC_level3'] = predicted_codes
    
    ATC_level33 = X_test1['ATC_level3']
    ATC_level1_4 = ATC_level33.copy()
    ATC_level2_4 = ATC_level33.copy()
    ATC_level3_4 = ATC_level33.copy()
    for index, atc in enumerate(ATC_level33):
        ATC_level1_4[index] = []
        ATC_level1_4[index].append(atc[0:1])
        ATC_level2_4[index] = []
        ATC_level2_4[index].append(atc[1:3])
        ATC_level3_4[index] = []
        ATC_level3_4[index].append(atc[3:4])

    X_test1 = X_test1.drop(labels=['ATC_level3'], axis="columns")

    categorical_atc1_4 = atc_level1_labels_encoder4.transform(ATC_level1_4)
    categorical_atc2_4 = atc_level2_labels_encoder4.transform(ATC_level2_4)
    categorical_atc3_4 = atc_level3_labels_encoder4.transform(ATC_level3_4)
    df_level1_4 = pd.DataFrame(categorical_atc1_4, columns=atc_level1_labels_encoder4.classes_)
    df_level2_4 = pd.DataFrame(categorical_atc2_4, columns=atc_level2_labels_encoder4.classes_)
    df_level3_4 = pd.DataFrame(categorical_atc3_4, columns=atc_level3_labels_encoder4.classes_)
    X_test1 = pd.concat([X_test1, df_level1_4, df_level2_4, df_level3_4], axis = 1)                         
    X_test1 = np.asarray(X_test1).astype(np.float32)
    X_test1[pd.isna(X_test1)] = np.nanmedian(X_test1)
    X_test1 = scaler4.transform(X_test1)

    return X_test1
    
def generate_predictions(scaler1, scaler2, scaler3, scaler4, model1, X_test1, model2, X_test2, model3, X_test3, model4, X_test4, mlb, y_labels_encoder2, y_labels_encoder3, y_labels_encoder4, atc_level1_labels_encoder2, atc_level1_labels_encoder3, atc_level2_labels_encoder3, atc_level1_labels_encoder4, atc_level2_labels_encoder4, atc_level3_labels_encoder4, previous_predictions = None, index = None):
    """Genera predicciones para todos los niveles, o solo para un índice si se repite"""
    
    def sample_prediction(model, X_test, encoder, previous_predictions = None, index = None):
        """Calcula la predicción con probabilidad ponderada para un índice o todos"""
        if previous_predictions is None:
            # Primera vez: calcular todas las predicciones
            y_prob = model.predict(X_test, verbose=0)
            predictions = [
                random.choices(encoder.classes_, weights=row, k=1)[0] for row in y_prob
            ]
        else:
            # Mantener las predicciones anteriores y cambiar solo la del índice dado
            predictions = previous_predictions.copy()
            if index is not None:
                y_prob = model.predict(X_test, verbose=0)
                row = np.array(y_prob)[index]
                predictions[index] = random.choices(encoder.classes_, weights=row, k=1)[0]
    
        return predictions
    
    # Nivel 1
    X_test1 = test_set_level1(scaler1, X_test1)
    pred_1 = sample_prediction(model1, X_test1, mlb, previous_predictions['pred_1'] if previous_predictions is not None else None, index)
    pred_df_level1 = pd.DataFrame(pred_1, columns=['pred_1'])

    # Nivel 2
    X_test2 = test_set_level2(scaler2, X_test2, pred_df_level1, atc_level1_labels_encoder2)
    pred_2 = sample_prediction(model2, X_test2, y_labels_encoder2, previous_predictions['pred_2'] if previous_predictions is not None else None, index)
    pred_df_level2 = pd.DataFrame(pred_2, columns=['pred_2'])

    # Nivel 3
    X_test3 = test_set_level3(scaler3, X_test3, pred_df_level1, pred_df_level2, atc_level1_labels_encoder3, atc_level2_labels_encoder3)
    pred_3 = sample_prediction(model3, X_test3, y_labels_encoder3, previous_predictions['pred_3'] if previous_predictions is not None else None, index)
    pred_df_level3 = pd.DataFrame(pred_3, columns=['pred_3'])

    # Nivel 4
    X_test4 = test_set_level4(scaler4, X_test4, pred_df_level1, pred_df_level2, pred_df_level3, atc_level1_labels_encoder4, atc_level2_labels_encoder4, atc_level3_labels_encoder4)
    pred_4 = sample_prediction(model4, X_test4, y_labels_encoder4, previous_predictions['pred_4'] if previous_predictions is not None else None, index)
    pred_df_level4 = pd.DataFrame(pred_4, columns=['pred_4'])

    return pred_df_level1, pred_df_level2, pred_df_level3, pred_df_level4
# Predict probabilities
def random_predictions(scaler1, scaler2, scaler3, scaler4, model1, X_test1, model2, X_test2, model3, X_test3, model4, X_test4, mlb, y_labels_encoder2, y_labels_encoder3, y_labels_encoder4, atc_level1_labels_encoder2, atc_level1_labels_encoder3, atc_level2_labels_encoder3, atc_level1_labels_encoder4, atc_level2_labels_encoder4, atc_level3_labels_encoder4):
    final_predictions = [[] for _ in range(len(X_test1))]
    max_attempts = 30
    for i in range(3):
        pred_df_level1, pred_df_level2, pred_df_level3, pred_df_level4 = generate_predictions(scaler1, scaler2, scaler3, scaler4, model1, X_test1, model2, X_test2, model3, X_test3, model4, X_test4, mlb, y_labels_encoder2, y_labels_encoder3, y_labels_encoder4,  atc_level1_labels_encoder2, atc_level1_labels_encoder3, atc_level2_labels_encoder3, atc_level1_labels_encoder4, atc_level2_labels_encoder4, atc_level3_labels_encoder4)

        for index in range(len(pred_df_level1)):
            attempts = 0
            while attempts < max_attempts:
                pred1 = pred_df_level1.at[index, 'pred_1']
                pred2 = str(pred_df_level2.at[index, "pred_2"]).zfill(2)
                pred3 = pred_df_level3.at[index, "pred_3"]
                pred4 = pred_df_level4.at[index, "pred_4"]
                prediction = pred1 + pred2 + pred3 + pred4

                if prediction not in final_predictions[index]:
                    final_predictions[index].append(prediction)
                    break  # Salimos del bucle cuando obtenemos una predicción nueva

                print(f"Prediction {prediction} already found in {final_predictions[index]}, reclassifying index {index}...")
                previous_predictions = pd.concat([pred_df_level1, pred_df_level2, pred_df_level3, pred_df_level4], axis = 1)
                # Recalcular la clasificación completa solo para este índice
                pred_df_level1, pred_df_level2, pred_df_level3, pred_df_level4 = generate_predictions(scaler1, scaler2, scaler3, scaler4, model1, X_test1, model2, X_test2, model3, X_test3, model4, X_test4, mlb, y_labels_encoder2, y_labels_encoder3, y_labels_encoder4, atc_level1_labels_encoder2, atc_level1_labels_encoder3, atc_level2_labels_encoder3, atc_level1_labels_encoder4, atc_level2_labels_encoder4, atc_level3_labels_encoder4, previous_predictions, index)
                attempts += 1
    return final_predictions
    
def hyperparametersselection(seed):
    set_seeds(seed)
    hyperparameters_grid = { 
        'learning_rate': [0.001, 0.0001, 5e-3],
        'batch_size': [16, 32, 64],
        'kernel_initializer': ['glorot_uniform', 'he_uniform'],
    }
    df_tests = random_search(50, seed, hyperparameters_grid)

    df_tests = df_tests.sort_values(by = "Precision nivel1")
    df_tests = pd.read_csv(f'neuralnetwork_results{seed}.csv', keep_default_na=False)
    df_tests['F1 nivel1'] = 2*((df_tests['Precision nivel1'] * df_tests['Recall nivel1'])/(df_tests['Precision nivel1'] + df_tests['Recall nivel1']))
    df_tests = df_tests.sort_values(by = "F1 nivel1", ascending=False)
    df_tests.to_csv(f'sortedneuralnetwork_results{seed}.csv', index = False)
    best_combination = (pd.read_csv(f"sortedneuralnetwork_results{seed}.csv", keep_default_na=False)).loc[0]
    return best_combination
    
def random_search(max_evals, seed, hyperparameters_grid):
    tested_params = set()
    stdout_jupyter = sys.stdout
    combinations = list(itertools.product(*hyperparameters_grid.values()))
    if len(combinations)>max_evals:
        max_evals = max_evals
    else:
        max_evals = len(combinations)
    tested_params = set()
    df_tests = pd.DataFrame(columns = ['learning_rate', 'batch_size', 'kernel_initializer', 'Precision nivel1', 'Precision nivel2', 'Precision nivel3', 'Precision nivel4', 'Recall nivel1', 'Recall nivel2', 'Recall nivel3', 'Recall nivel4', 'Drugs that have at least one match'], index = list(range(max_evals)))
    sys.stdout = open(f'log{seed}.txt', 'w')
    for comb in range(max_evals):
        while True:
            random_params = {k: random.sample(v, 1)[0] for k, v in hyperparameters_grid.items()}
            params_tuple = tuple(random_params.values())
            if params_tuple not in tested_params:
                tested_params.add(params_tuple)
                break   
        X_train1, y_train1, X_test1, y_test1 = train_test_1(seed)
        X_train2, y_train2, X_test2, y_test2 = train_test_2(seed)
        X_train3, y_train3, X_test3, y_test3 = train_test_3(seed)
        X_train4, y_train4, X_test4, y_test4 = train_test_4(seed)
        # LEVEL1
        # Get all available labels describing the level 1 ATC code
        labels1 = set()
        for lista in y_train1:
            lista = convert_string_list(lista)
            for code in lista:
                labels1.add(code)
        for lista in y_test1:
            lista = convert_string_list(lista)
            for code in lista:
                labels1.add(code)
                
        labels1 = sorted(list(labels1))
        mlb = MultiLabelBinarizer()
        mlb.fit([labels1])
        y_new = y_train1.copy()
        for index, lista in enumerate(y_train1):
            y_new[index] = []
            lista = convert_string_list(lista)
            for i, label in enumerate(lista):
                y_new[index].append(lista[i])
        y_categorical1 = mlb.transform(y_new)
        
        X_train1 = np.asarray(X_train1).astype(np.float32)
        y_categorical1 = np.asarray(y_categorical1).astype(np.float32)
        # Complete NaN values in each column with the median
        X_train1[pd.isna(X_train1)] = np.nanmedian(X_train1)
        # Define an instance of the MinMaxScaler
        scaler1 = MinMaxScaler()
        # Fit the scaler to the data and transform it
        X_train1 = scaler1.fit_transform(X_train1)
        
        model1 = Sequential([
            (Dense(len(labels1), input_dim = X_train1.shape[1], activation = 'sigmoid', kernel_initializer=random_params['kernel_initializer']))
        ])
        
        # Compile the model
        model1.compile(
            loss = keras.losses.BinaryCrossentropy(),
            optimizer = keras.optimizers.Adam(learning_rate=random_params['learning_rate']),
            metrics = ['binary_accuracy', 'binary_crossentropy']
        )
        
        # Define early stopping
        callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience = 5, verbose = 1)
        
        # Train the model
        history = model1.fit(
            X_train1, 
            y_categorical1, 
            epochs = 500, 
            batch_size = random_params['batch_size'],
            validation_split = 0.15,
            callbacks = [callback], 
            verbose = 0
        )
        #LEVEL 2
        X_train2, encoded_y_train2, labels2, y_labels_encoder2, atc_level1_labels_encoder2, scaler2 = get_X_train_level2(X_train2, y_train2)
        
        model2 = Sequential([
            (Dense(len(labels2), input_dim = X_train2.shape[1], activation = 'sigmoid', kernel_initializer=random_params['kernel_initializer']))
        ])
        
        model2.compile(
            loss = keras.losses.BinaryCrossentropy(),
            optimizer = keras.optimizers.Adam(learning_rate=random_params['learning_rate']),
            metrics = ['binary_accuracy', 'binary_crossentropy']
        )
        
        callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience = 5, verbose = 1)
        
        history = model2.fit(
            X_train2, 
            encoded_y_train2, 
            epochs = 500, 
            batch_size = random_params['batch_size'],
            validation_split = 0.15,
            callbacks = [callback], 
            verbose = 0
        )
        
        #LEVEL 3
        X_train3, encoded_y_train3, labels3, y_labels_encoder3, atc_level1_labels_encoder3, atc_level2_labels_encoder3, scaler3 = get_X_train_level3(X_train3, y_train3, X_test3)
        
        model3 = Sequential([
            (Dense(len(labels3), input_dim = X_train3.shape[1], activation = 'sigmoid', kernel_initializer=random_params['kernel_initializer']))
        ])
        
        model3.compile(
            loss = keras.losses.BinaryCrossentropy(), 
            optimizer = keras.optimizers.Adam(learning_rate=random_params['learning_rate']),
            metrics = ['binary_accuracy', 'binary_crossentropy']
        )
        
        callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience = 5, verbose = 1)
        
        history = model3.fit(
            X_train3, 
            encoded_y_train3, 
            epochs = 500, 
            batch_size = random_params['batch_size'],
            validation_split = 0.15,
            callbacks = [callback], 
            verbose = 0
        )
        
        #LEVEL 4
        X_train4, encoded_y_train4, labels4, y_labels_encoder4, atc_level1_labels_encoder4, atc_level2_labels_encoder4, atc_level3_labels_encoder4, scaler4 = get_X_train_level4(X_train4, y_train4, X_test4)
        
        model4 = Sequential([
            (Dense(len(labels4), input_dim = X_train4.shape[1], activation = 'sigmoid', kernel_initializer=random_params['kernel_initializer']))
        ])
        
        model4.compile(
            loss = keras.losses.BinaryCrossentropy(), 
            optimizer = keras.optimizers.Adam(learning_rate=random_params['learning_rate']),
            metrics = ['binary_accuracy', 'binary_crossentropy']
        )
        
        callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience = 5, verbose = 1)
        
        history = model4.fit(
            X_train4, 
            encoded_y_train4, 
            epochs = 500, 
            batch_size = random_params['batch_size'],
            validation_split = 0.15,
            callbacks = [callback], 
            verbose = 0
        )
        
        #TEST
        X_test3.drop(labels=['ATC_level2'], axis="columns", inplace=True)
        X_test4.drop(labels=['ATC_level3'], axis="columns", inplace = True)
        output = random_predictions(scaler1, scaler2, scaler3, scaler4, model1, X_test1, model2, X_test2, model3, X_test3, model4, X_test4, mlb, y_labels_encoder2, y_labels_encoder3, y_labels_encoder4, atc_level1_labels_encoder2, atc_level1_labels_encoder3, atc_level2_labels_encoder3, atc_level1_labels_encoder4, atc_level2_labels_encoder4, atc_level3_labels_encoder4)

        predictions = []
        for preds in output:
            interm = []
            for pred in preds:
                clean_pred = pred.replace('<START>', '').replace('<END>', '')
                if len(clean_pred) == 5:
                    interm.append(clean_pred)
            predictions.append(interm)
                
        precision_1, precision_2, precision_3, precision_4 = defined_metrics.precision(predictions, f"../Datasets/Rep_val_set{seed}.csv", 'ATC Codes')
        recall_1, recall_2, recall_3, recall_4, comp = defined_metrics.recall(predictions, f"../Datasets/Rep_val_set{seed}.csv", 'ATC Codes')
        df_tests.iloc[comb, :] = [f"{random_params['learning_rate']}", f"{random_params['batch_size']}", f"{random_params['kernel_initializer']}", f"{precision_1}", f"{precision_2}", f"{precision_3}", f"{precision_4}", f"{recall_1}", f"{recall_2}", f"{recall_3}", f"{recall_4}", f"{comp}"]
        df_tests.to_csv(f"neuralnetwork_results{seed}.csv", index = False)
    sys.stdout = stdout_jupyter
    return df_tests
