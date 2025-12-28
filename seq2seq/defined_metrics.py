import pandas as pd
from sklearn import metrics

# Converts a string that simulates a list to a real list
def convert_string_list(element):
    # Delete [] of the string
    element = element[0:len(element)]
    # Create a list that contains each code as e.g. 'A'
    ATC_list = list(element.split('; '))
    for index, code in enumerate(ATC_list):
        # Delete '' of the code
        ATC_list[index] = code[0:len(code)]
    return ATC_list


def precisionatn(
    predictions, 
    name_true_dataset_onecodeperdrug, 
    name_column
):
    # Load validation dataset and initialize counters
    df = pd.read_csv(name_true_dataset_onecodeperdrug)
    total_samples = [len(predictions)] * 4
    # Initialize counters for total matches  at each level
    total_matches = [0, 0, 0, 0]

    # Iterate through the predictions for each compound
    for i, list_preds in enumerate(predictions):
        true_code = df[name_column].iloc[i]
        match_found = [False, False, False, False]
        for pred in list_preds:
            # Compare each level
            if pred[0] == true_code[0]:
                match_found[0] = True
                if pred[1:3] == true_code[1:3]:
                    match_found[1] = True
                    if pred[3] == true_code[3]:
                        match_found[2] = True
                        if pred[4] == true_code[4]:
                            match_found[3] = True
        if match_found[0]:
            total_matches[0] += 1
        else:
            total_samples[1] -= 1
        if match_found[1]:
            total_matches[1] += 1
        else:
            total_samples[2] -= 1
        if match_found[2]:
            total_matches[2] += 1
        else:
            total_samples[3] -= 1
        if match_found[3]:
            total_matches[3] += 1
    precisionatn = [total_matches[i] / total_samples[i] if total_samples[i] > 0 else 0 for i in range(4)]
    
    precisionatn_1 = precisionatn[0] * 100
    precisionatn_2 = precisionatn[1] * 100
    precisionatn_3 = precisionatn[2] * 100
    precisionatn_4 = precisionatn[3] * 100
    return precisionatn_1, precisionatn_2, precisionatn_3, precisionatn_4

def precision(
    predictions, 
    name_true_dataset, 
    name_column
):
    # Load validation dataset and initialize counters
    df = pd.read_csv(name_true_dataset)
    total_compounds = len(df)
    # Initialize counters for total matches and valid comparisons at each level
    total_matches = [0, 0, 0, 0]
    valid_comparisons = [total_compounds] * 4 # Start with all compounds being valid for comparison at every level
    # Iterate through the predictions for each compound
    for i, list_preds in enumerate(predictions):
        true_codes = df[name_column].iloc[i]
        true_codes = convert_string_list(true_codes)
        num_preds = [len(list_preds)]*4
        level_matches = [0, 0, 0, 0]
        not_compared = [False, True, True, True]
        # Compare predicted codes with true codes
        for pred in list_preds:
            match_found = [False, False, False, False]
            for true_code in true_codes:
                # Compare each level
                if pred[0] == true_code[0]:
                    match_found[0] = True
                    if pred[1:3] == true_code[1:3]:
                        match_found[1] = True
                        if pred[3] == true_code[3]:
                            match_found[2] = True
                            if pred[4] == true_code[4]:
                                match_found[3] = True
            for level in range(4):
                if match_found[level]:
                    # If at least one time it can compare with level y+1 so it converts to False
                    # Level y has matched so it compares to level y+1
                    level_matches[level] += 1
                    if level+1 < 4:
                        not_compared[level+1] = False
                else:
                     if level+1 < 4:
                        num_preds[level+1] -= 1
        valid_comparisons = [valid_comparisons[level] - int(not_compared[level]) for level in range(4)]
        total_matches = [total_matches[level] + ((level_matches[level]/num_preds[level]) if num_preds[level] > 0 else 0) for level in range(4)]

    precisions = [total_matches[level] / valid_comparisons[level] if valid_comparisons[level] > 0 else 0 for level in range(4)]
    
    precision_1 = precisions[0] 
    precision_2 = precisions[1] 
    precision_3 = precisions[2] 
    precision_4 = precisions[3] 
    return precision_1, precision_2, precision_3, precision_4

def recall(
    predictions, 
    name_true_dataset, 
    name_column
):
    # Load test dataset and initialize counters
    df = pd.read_csv(name_true_dataset)
    total_compounds = len(df)
    # Initialize counters for total matches and valid comparisons at each level
    total_matches = [0, 0, 0, 0]
    valid_comparisons = [total_compounds] * 4 # Start with all compounds being valid for comparison at every level
    compounds_with_match = [0, 0, 0, 0]
    # Iterate through the predictions for each compound
    for i, list_preds in enumerate(predictions):
        true_codes = df[name_column].iloc[i]
        true_codes = convert_string_list(true_codes)
        num_true_codes = [len(true_codes)]*4
        level_matches = [0, 0, 0, 0]
        not_compared = [False, True, True, True]
        compound_match = [False, False, False, False]
        # Compare true codes with predicted codes
        for true_code in true_codes:
            match_found = [False, False, False, False]
            for pred in list_preds:
                # Compare each level
                if pred[0] == true_code[0]:
                    match_found[0] = True
                    compound_match[0] = True
                    if pred[1:3] == true_code[1:3]:
                        match_found[1] = True
                        compound_match[1] = True
                        if pred[3] == true_code[3]:
                            match_found[2] = True
                            compound_match[2] = True
                            if pred[4] == true_code[4]:
                                match_found[3] = True
                                compound_match[3] = True
            for level in range(4):
                if match_found[level]:
                    # If at least one time it can compare with level y+1 then it converts to False
                    # Level y has matched so it compares to level y+1
                    level_matches[level] += 1
                    if level+1 < 4:
                        not_compared[level+1] = False
                else:
                    if level+1 < 4:
                        num_true_codes[level+1] -= 1
        # For compound x it couldn't compare at level i so it has to decrease the amount of comparisons at level i for compound x
        valid_comparisons = [valid_comparisons[level] - int(not_compared[level]) for level in range(4)]
        total_matches = [total_matches[level] + ((level_matches[level]/num_true_codes[level]) if num_true_codes[level] > 0 else 0) for level in range(4)]
        for level in range(4):
            if all(compound_match[:level+1]):
                compounds_with_match[level] += 1
        
    recalls = [(total_matches[level] / valid_comparisons[level]) if valid_comparisons[level] > 0 else 0 for level in range(4)]
    recall_1 = recalls[0]
    recall_2 = recalls[1]
    recall_3 = recalls[2]
    recall_4 = recalls[3]
    return recall_1, recall_2, recall_3, recall_4, compounds_with_match

def complete_metrics(output_beam, 
                     name_true_dataset, 
                     name_column, 
                     k
):
    precisions = []
    recalls = []
    f1s = []
    df = pd.read_csv(name_true_dataset)
    for i, preds in enumerate(output_beam):
        ground_truth = convert_string_list(df[name_column][i])
        binary_predictions = []
        binary_ground_truth = []
        clean_preds = []
        for pred in preds[0:k]:
            clean_preds.append(pred)
        set_pred_gt = list(set(clean_preds + ground_truth))
        for code in set_pred_gt:
            if code in clean_preds:
                binary_predictions.append(1)
            else:
                binary_predictions.append(0)
            if code in ground_truth:
                binary_ground_truth.append(1)
            else:
                binary_ground_truth.append(0)    
        precisions.append(metrics.precision_score(binary_ground_truth, binary_predictions, zero_division = 0.0))
        recalls.append(metrics.recall_score(binary_ground_truth, binary_predictions, zero_division = 0.0))
        f1s.append(metrics.f1_score(binary_ground_truth, binary_predictions, zero_division = 0.0))
    return precisions, recalls, f1s

def complete_metrics_level3(output_beam, 
                     name_true_dataset, 
                     name_column, 
                     k
):
    precisions = []
    recalls = []
    f1s = []
    df = pd.read_csv(name_true_dataset)
    for i, preds in enumerate(output_beam):
        ground_truth = convert_string_list(df[name_column][i])
        binary_predictions = []
        binary_ground_truth = []
        clean_preds = []
        clean_gt = []
        for pred in preds[0:k]:
            clean_preds.append(pred)
        for code in ground_truth:
            clean_gt.append(code[0:4])
        set_pred_gt = list(set(clean_preds + clean_gt))
        for code in set_pred_gt:
            if code in clean_preds:
                binary_predictions.append(1)
            else:
                binary_predictions.append(0)
            if code in clean_gt:
                binary_ground_truth.append(1)
            else:
                binary_ground_truth.append(0)    
        precisions.append(metrics.precision_score(binary_ground_truth, binary_predictions, zero_division = 0.0))
        recalls.append(metrics.recall_score(binary_ground_truth, binary_predictions, zero_division = 0.0))
        f1s.append(metrics.f1_score(binary_ground_truth, binary_predictions, zero_division = 0.0))
    return precisions, recalls, f1s

def complete_metrics_level2(output_beam, 
                     name_true_dataset, 
                     name_column, 
                     k
):
    precisions = []
    recalls = []
    f1s = []
    df = pd.read_csv(name_true_dataset)
    for i, preds in enumerate(output_beam):
        ground_truth = convert_string_list(df[name_column][i])
        binary_predictions = []
        binary_ground_truth = []
        clean_preds = []
        clean_gt = []
        for pred in preds[0:k]:
            clean_preds.append(pred)
        for code in ground_truth:
            clean_gt.append(code[0:3])
        set_pred_gt = list(set(clean_preds + clean_gt))
        for code in set_pred_gt:
            if code in clean_preds:
                binary_predictions.append(1)
            else:
                binary_predictions.append(0)
            if code in clean_gt:
                binary_ground_truth.append(1)
            else:
                binary_ground_truth.append(0)    
        precisions.append(metrics.precision_score(binary_ground_truth, binary_predictions, zero_division = 0.0))
        recalls.append(metrics.recall_score(binary_ground_truth, binary_predictions, zero_division = 0.0))
        f1s.append(metrics.f1_score(binary_ground_truth, binary_predictions, zero_division = 0.0))
    return precisions, recalls, f1s