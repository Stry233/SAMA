"""Metrics and evaluation utilities for the MIA framework."""

import logging
import numpy as np
from sklearn.metrics import auc, roc_curve
from typing import List, Tuple, Any


def results_with_bootstrapping(y_true: List, y_pred: List, fpr_thresholds: List[float],
                               n_bootstraps: int = 1000) -> List[str]:
    """Compute bootstrapped AUC and TPR at given FPR thresholds."""
    if not y_true or not y_pred:
        logging.warning("Empty y_true or y_pred in results_with_bootstrapping")
        na_result = "N/A"
        return [na_result] + [na_result for _ in fpr_thresholds]

    n = len(y_true)
    if n == 0:
        logging.warning("Zero length y_true in results_with_bootstrapping")
        na_result = "N/A"
        return [na_result] + [na_result for _ in fpr_thresholds]

    aucs = []
    tprs_at_fprs = {fpr_val: [] for fpr_val in fpr_thresholds}

    for _ in range(n_bootstraps):
        if n == 1:
            idx = [0] * n
        else:
            idx = np.random.choice(n, n, replace=True)

        y_true_sample = np.array(y_true)[idx]
        y_pred_sample = np.array(y_pred)[idx]

        if len(np.unique(y_true_sample)) < 2:
            aucs.append(np.nan)
            for fpr_val in fpr_thresholds:
                tprs_at_fprs[fpr_val].append(np.nan)
            continue

        fpr_bs, tpr_bs, _ = roc_curve(y_true_sample, y_pred_sample)
        aucs.append(auc(fpr_bs, tpr_bs))

        for fpr_val in fpr_thresholds:
            if not fpr_bs.size:
                tprs_at_fprs[fpr_val].append(np.nan)
                continue
            tpr_at_fpr = tpr_bs[np.argmin(np.abs(fpr_bs - fpr_val))]
            tprs_at_fprs[fpr_val].append(tpr_at_fpr)

    # Calculate mean and std
    mean_auc = np.nanmean(aucs)
    std_auc = np.nanstd(aucs)
    results = [f"{mean_auc:.4f} ± {std_auc:.4f}"]

    for fpr_val in fpr_thresholds:
        mean_tpr = np.nanmean(tprs_at_fprs[fpr_val])
        std_tpr = np.nanstd(tprs_at_fprs[fpr_val])
        results.append(f"{mean_tpr:.4f} ± {std_tpr:.4f}")

    return results