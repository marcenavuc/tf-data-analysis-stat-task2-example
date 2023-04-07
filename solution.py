import pandas as pd
import numpy as np

from scipy.stats import norm, chi2


chat_id = 351730666 # Ваш chat ID, не меняйте название переменной


def solution(p: float, x: np.array) -> tuple:
    alpha = 1 - p
    return max(max(x), 2 * (x.mean() - np.sqrt(np.var(x)) * norm.ppf(1 - alpha / 2) / np.sqrt(len(x)))), \
           max(max(x), 2 * (x.mean() - np.sqrt(np.var(x)) * norm.ppf(alpha / 2) / np.sqrt(len(x))))
