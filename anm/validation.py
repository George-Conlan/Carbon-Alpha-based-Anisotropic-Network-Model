import numpy as np
from scipy.stats import pearsonr as pr, spearmanr as sr
import math
def bfactor_to_msf(bfactors):
    inverteddw = 3/(8*math.pi**2)
    msf = bfactors*inverteddw
    return msf



def fit_scale(pred_msf, exp_msf):
      numerator = np.sum(pred_msf*exp_msf)
      denominator = np.sum(pred_msf**2)
      c = numerator/denominator
      return c
def compute_correlation(pred_msf, exp_msf):
     return pr(pred_msf, exp_msf), sr(pred_msf, exp_msf)
def compute_residuals(pred_msf, exp_msf, scale):
     pmsf = pred_msf*scale-exp_msf
     return pmsf

def validate(pred_msf, bfactors):
     exp_msf = bfactor_to_msf(bfactors)
     scale = fit_scale(pred_msf, exp_msf)
     correlation = compute_correlation(pred_msf, exp_msf)
     residuals = compute_residuals(pred_msf, exp_msf, scale)
     return scale, correlation, residuals
