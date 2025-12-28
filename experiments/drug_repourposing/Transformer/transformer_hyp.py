import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../..')))
from seq2seq import *
import pandas as pd
import numpy as np
import torch
import random
import itertools
from sklearn.preprocessing import StandardScaler

def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def convert_string_list(element):
    # Delete [] of the string
    element = element[0:len(element)]
    # Create a list that contains each code as e.g. 'A'
    ATC_list = list(element.split('; '))
    for index, code in enumerate(ATC_list):
        # Delete '' of the code
        ATC_list[index] = code[0:len(code)]
    return ATC_list

def hyperparametersselection(seed, source_index, target_index, X_train, X_val, X_val2, y_train, y_val):
    set_seeds(seed)
    hyperparameters_grid = { 
        'embedding_dim': [64, 128],
        'feedforward_dim': [64, 128, 256],
        'enc_layers': [2, 4],
        'dec_layers': [2, 3, 4],
        'attention_heads': [2, 4],
        'dropout': [0.0, 0.1, 0.2],
        'weight_decays': [10**-4, 10**-5],
        'learning_rates': [10**-3, 10**-4]
    }
    df_tests = random_search(50, seed, source_index, target_index, X_train, y_train, X_val, X_val2, y_val, hyperparameters_grid)

    df_tests = df_tests.sort_values(by = "Precision nivel1")
    df_tests = pd.read_csv(f'transformer_results{seed}.csv')
    df_tests['F1 nivel1'] = 2*((df_tests['Precision nivel1'] * df_tests['Recall nivel1'])/(df_tests['Precision nivel1'] + df_tests['Recall nivel1']))
    df_tests = df_tests.sort_values(by = "F1 nivel1", ascending=False)
    df_tests.to_csv(f'sortedtransformer_results{seed}.csv', index = False)
    return df_tests.loc[0]

def random_search(max_evals, seed, source_index, target_index, X_train, y_train, X_val, X_val2, y_val, hyperparameters_grid):
    tested_params = set()
    stdout_jupyter = sys.stdout
    df_tests = pd.DataFrame(columns = ['#epochs', 'embedding_dim', 'feedforward_dim', 'enc_layers', 'dec_layers', 'attention_heads', 'dropout', 'weight_decay', 'learning_rate', 'Precision nivel1', 'Precision nivel2', 'Precision nivel3', 'Precision nivel4', 'Recall nivel1', 'Recall nivel2', 'Recall nivel3', 'Recall nivel4', 'Drugs that have at least one match'], index = list(range(max_evals)))
    sys.stdout = open(f'log{seed}.txt', 'w')
    for i in range(max_evals):
        while True:
            random_params = {k: random.sample(v, 1)[0] for k, v in hyperparameters_grid.items()}
            params_tuple = tuple(random_params.values())
            if params_tuple not in tested_params:
                tested_params.add(params_tuple)
                break   
        model = models.Transformer(
                 source_index, 
                 target_index,
                 max_sequence_length = 800,
                 embedding_dimension = random_params['embedding_dim'],
                 feedforward_dimension = random_params['feedforward_dim'],
                 encoder_layers = random_params['enc_layers'],
                 decoder_layers = random_params['dec_layers'],
                 attention_heads = random_params['attention_heads'],
                 activation = "relu",
                 dropout = random_params['dropout'])   
        model.to("cuda")
        model.fit(
                X_train,
                y_train,
                X_val, 
                y_val, 
                batch_size = 32, 
                epochs = 500, 
                learning_rate = random_params['learning_rates'], 
                weight_decay = random_params['weight_decays'],
                progress_bar = 0, 
                save_path = None
        ) 
        model.load_state_dict(torch.load("best_model.pth", weights_only=True))
        ep = model.early_stopping.best_epoch
        loss, error_rate = model.evaluate(X_val, y_val)    

        predictions2, log_probabilities2 = search_algorithms.beam_search(
            model, 
            X_val2,
            predictions = 6, # max length of the predicted sequence
            beam_width = 3,
            batch_size = 32, 
            progress_bar = 0
        )
        output_beam2 = [target_index.tensor2text(p) for p in predictions2]
                
        predictions = []
        for preds in output_beam2:
            interm = []
            for pred in preds:
                clean_pred = pred.replace('<START>', '').replace('<END>', '')
                if len(clean_pred) == 5:
                    interm.append(clean_pred)
            predictions.append(interm)
                
        precision_1, precision_2, precision_3, precision_4 = defined_metrics.precision(predictions, f"../Datasets/Rep_val_set{seed}.csv", 'ATC Codes')
        recall_1, recall_2, recall_3, recall_4, comp = defined_metrics.recall(predictions, f"../Datasets/Rep_val_set{seed}.csv", 'ATC Codes')
        df_tests.iloc[i, :] = [f"{ep}", f"{random_params['embedding_dim']}", f"{random_params['feedforward_dim']}", f"{random_params['enc_layers']}", f"{random_params['dec_layers']}", f"{random_params['attention_heads']}", f"{random_params['dropout']}", f"{random_params['weight_decays']}", f"{random_params['learning_rates']}", f"{precision_1}", f"{precision_2}", f"{precision_3}", f"{precision_4}", f"{recall_1}", f"{recall_2}", f"{recall_3}", f"{recall_4}", f"{comp}"]
        df_tests.to_csv(f"transformer_results{seed}.csv", index = False)
    sys.stdout = stdout_jupyter
    return df_tests

