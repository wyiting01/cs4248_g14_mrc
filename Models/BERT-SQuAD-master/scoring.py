"""
Script to get F1 score for BERT model

Inputs will require the answer provided and / or span provided. This is accompanied by the answers and spans produced by the bert model
"""

def f1_score(real_start, real_end, pred_start, pred_end):
    intersection = max(0,min(real_end, pred_end) - max(real_start, pred_start))
    union = (pred_end - pred_start + 1) + (real_end - real_start + 1) - intersection

    precision = intersection / (pred_end - pred_start + 1)
    recall = intersection / (real_end - real_start + 1)
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    return f1

def batch_f1(ans, pred):
    total_f1 = 0
    for i in range(ans):
        ans_span = ans[i]
        pred_span = pred[i]

        total_f1 += f1_score(ans_span[0], ans_span[0], pred_span[1], pred_span[1])
    return total_f1 / len(ans)