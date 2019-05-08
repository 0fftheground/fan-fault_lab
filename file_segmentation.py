import numpy as np
import pandas as pd
from utils.preprocessing import SimplePreprocessor
import time
import os

outdir = "./output_data/"
if not os.path.exists(outdir):
    os.mkdir(outdir)
# time_span
error_time_point_list = ['2017-09-28 08:30:00.000', '2017-09-28 11:00:00.000', '2017-10-12 08:50:00.000',
                         '2017-10-12 11:00:00.000',
                         '2017-10-12 15:20:00.000', '2017-10-12 17:30:00.000', '2017-10-13 09:02:00.000',
                         '2017-10-13 11:02:00.000',
                         '2017-10-13 13:00:00.000', '2017-10-13 14:48:00.000']

correct_time_point_list = ['2017-10-15 12:26:00.000',
                           '2017-10-15 12:38:00.000',
                           '2017-10-16 09:00:00.000', '2017-10-16 11:00:00.000', '2017-10-16 13:40:00.000',
                           '2017-10-16 14:40:00.000']

fan_fault = pd.read_csv("data/fan_fault.csv")
fan_fault['timestamp'] = fan_fault['testtime'].apply(
    lambda x: int(time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S.000'))))

sp = SimplePreprocessor()
correct_data_frame = sp.filesegmentation(correct_time_point_list, fan_fault,1)
error_data_frame = sp.filesegmentation(error_time_point_list, fan_fault,0)
total_data_frame = pd.concat(correct_data_frame + error_data_frame, axis=0, ignore_index=True)
total_data_frame.to_csv(os.path.join(outdir, 'valid_total.csv'))
for index in range(len(correct_data_frame)):
    correct_data_frame[index].to_csv(os.path.join(outdir, str(time.strftime('%Y-%m-%d-%H%M', time.localtime(correct_time_point_list[index * 2]))) + '--' + str(
        time.strftime('%Y-%m-%d-%H%M', time.localtime(correct_time_point_list[index * 2 + 1]))) + ".csv")
                                     )
for index in range(len(error_data_frame)):
    error_data_frame[index].to_csv(os.path.join(outdir, str(time.strftime('%Y-%m-%d-%H%M',
                                                                          time.localtime(error_time_point_list[index * 2]))) + '--' + str(
        time.strftime('%Y-%m-%d-%H%M', time.localtime(error_time_point_list[index * 2 + 1]))) + ".csv")
                                   )
