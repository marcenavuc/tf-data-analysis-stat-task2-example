import pandas as pd
import numpy as np

from scipy.stats import norm, chi2


chat_id = 351730666 # Ваш chat ID, не меняйте название переменной


def solution(p: float, x: np.array) -> tuple:
    n = len(x)
    s2 = np.var(x, ddof=1)
    left = np.sqrt(n - 1) * s2 / chi2.ppf((1 - p) / 2, df=n - 1)
    right = np.sqrt(n - 1) * s2 / chi2.ppf((1 + p) / 2, df=n - 1)
    return left, right
