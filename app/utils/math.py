# -*- coding: utf-8 -*-
from typing import Union

def safe_division(numerator: Union[int, float], denominator: Union[int, float]) -> float:
    """
    Performs division safely, returning 0.0 if the denominator is zero.
    """
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)