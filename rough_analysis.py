import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from utils.preprocessing import SimplePreprocessor

sp = SimplePreprocessor()
correct1 = pd.read_csv("output_data/2017-10-15-1226--2017-10-15-1238.csv")

row1 = correct1.iloc[1]
blobvalue = sp.revertgzip(row1.blobvalue)
skew = stats.skew(blobvalue)
kurtosis = stats.kurtosis(blobvalue)
print(blobvalue)
print('cal_avg: ' + str(blobvalue.mean()))
print('cal_skew: ' + str(skew))
print('cal_kurtosis: ' + str(kurtosis))
print("data_skew: " + str(row1.Skewness))
print("data_mean: " + str(row1.avs))
print('data_kurtosis: ' + str(row1.Kurtosis))
