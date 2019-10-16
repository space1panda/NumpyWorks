def precision(result, ground_truth):
    return len(result[result==ground_truth]
               [result[result==ground_truth]==True])/ len(result[result==True])

def recall(result, ground_truth):
    return len(result[result==ground_truth]
               [result[result==ground_truth]==True])/ len(ground_truth[ground_truth==True])

def f1(precision, recall):
    return 2 * (precision*recall) / (precision+recall)