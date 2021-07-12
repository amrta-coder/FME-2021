import numpy as np
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve


def get_optimal_precision_recall(y_true, y_score):
    """Find precision and recall values that maximize f1 score."""
    # Get precision-recall curve
    precision, recall, threshold = precision_recall_curve(y_true, y_score)
    # Compute f1 score for each point (use nan_to_num to avoid nans messing up the results)
    f1_score = np.nan_to_num(2 * precision * recall / (precision + recall))
    # Select threshold that maximize f1 score
    index = np.argmax(f1_score)
    threshold = threshold[index-1] if index != 0 else threshold[0]-1e-10
    return threshold, f1_score[index], precision[index], recall[index]

def pred_modify(pred_arr):
    for index in range(len(pred_arr)):
        # if index==0:
        #     if pred_arr[index] ==1 and pred_arr[index+1] ==1:
        #         pred_arr[index] = 1
        #     else:
        #         pred_arr[index] = 0
        if index>0 and index<len(pred_arr)-1:
            if pred_arr[index-1] ==1 and pred_arr[index] ==1:
                pred_arr[index] = 1
            elif pred_arr[index+1] ==1 and pred_arr[index] ==1:
                pred_arr[index] = 1
            else:
                pred_arr[index] = 0
        # elif index==len(pred_arr)-1:
        #     if pred_arr[index] ==1 and pred_arr[index-1] ==1:
        #         pred_arr[index] = 1
        #     else:
        #         pred_arr[index] = 0
    return pred_arr