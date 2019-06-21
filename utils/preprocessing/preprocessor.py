import time
import binascii
import gzip
import numpy as np
import pandas as pd
from scipy.fftpack import fft
import os


class Preprocessor:
    def __init__(self):
        return

    def filesegmentation(self, file_list, data_frame, label):
        ''' 根据提供的时间段序列，从总样本中提取获得带标签的数据。
        :param file_list (List): 正或负样本的时间段序列，长度为2*N,N为时间段数.i.e：['2017-09-28 08:30:00.000', '2017-09-28 11:00:00.000', '2017-10-12 08:50:00.000','2017-10-12 11:00:00.000']
        :param data_frame(Dataframe): 包含所有样本的dataFrame
        :param label(int): 正负样本标签，取值为0,1.
        :return:根据file_list提取的带标签的子样本。
        '''
        if len(file_list) % 2 != 0:
            raise Exception("Invalid length!", "file_list.length: " + len(file_list))
            return

        for index in range(len(file_list)):
            file_list[index] = int(time.mktime(time.strptime(file_list[index], '%Y-%m-%d %H:%M:%S.000')))

        output_list = []
        count = 0
        while count < len(file_list):
            temp = data_frame[(data_frame['timestamp'] >= file_list[count]) & (
                    data_frame['timestamp'] <= file_list[count + 1])]
            temp['label'] = label
            output_list.append(temp)
            count += 2
        return output_list

    def revertgzip(self, blobvalue):
        ''' 对编码的blobvalue值进行解码
        :param blobvalue(str):
        :return(np array):
        '''
        b = blobvalue[2:]  # 截取掉'0x'
        c = binascii.a2b_hex(b)  # 转换成ASCii编码的字符串
        d = str(gzip.decompress(c), encoding="utf8")
        e = d.split('$')
        for index in range(len(e)):
            if e[index] == '':
                del e[index]
        d = np.array(e)
        return d.astype('float')

    def fft_preprocessing(self):
        ''' 对waveType为2和6的正负样本进行傅里叶变换，并存储得到的频域值。
        '''
        sequential_data = pd.read_csv('./output_data/sequential_data.csv')
        ne_2_sample = sequential_data[(sequential_data.label == 0) & (sequential_data.ml_id == 2)].drop(
            ['label', 'ml_id', 'Unnamed: 0'], axis=1)
        ne_6_sample = sequential_data[(sequential_data.label == 0) & (sequential_data.ml_id == 6)].drop(
            ['label', 'ml_id', 'Unnamed: 0'], axis=1)
        po_2_sample = sequential_data[(sequential_data.label == 1) & (sequential_data.ml_id == 2)].drop(
            ['label', 'ml_id', 'Unnamed: 0'], axis=1)
        po_6_sample = sequential_data[(sequential_data.label == 1) & (sequential_data.ml_id == 6)].drop(
            ['label', 'ml_id', 'Unnamed: 0'], axis=1)
        po_6_f = []
        ne_6_f = []
        ne_2_f = []
        po_2_f = []
        for index in range(len(po_6_sample)):
            item = 2.0 / po_6_sample.shape[1] * np.abs(fft(po_6_sample.iloc[index]))
            po_6_f.append(item[0:len(item)//2])
        for index in range(len(ne_6_sample)):
            item = 2.0 / ne_6_sample.shape[1] * np.abs(fft(ne_6_sample.iloc[index]))
            ne_6_f.append(item[0:len(item)//2])
        for index in range(len(po_2_sample)):
            item = 2.0 / po_2_sample.shape[1] * np.abs(fft(po_2_sample.iloc[index]))
            po_2_f.append(item[0:len(item)//2])
        for index in range(len(ne_2_sample)):
            item = 2.0 / ne_2_sample.shape[1] * np.abs(fft(ne_2_sample.iloc[index]))
            ne_2_f.append(item[0:len(item)//2])
        if not os.path.isdir('output_data/fft'):
            os.mkdir('output_data/fft')
        np.save('output_data/fft/fft_po6_train_data', np.array(po_6_f))
        np.save('output_data/fft/fft_ne6_train_data', np.array(ne_6_f))
        np.save('output_data/fft/fft_ne2_train_data', np.array(ne_2_f))
        np.save('output_data/fft/fft_po2_train_data', np.array(po_2_f))

    def file_segmentation(self):
        '''
        根据仁喜提供的正负样本时间段，从总样本中分别提取正负样本，并将正负样本合并存储到'output_data/valid_total.csv'
        正负样本根据时间段分别存储到output_data/
        '''
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

        correct_data_frame = self.filesegmentation(correct_time_point_list, fan_fault, 1)
        error_data_frame = self.filesegmentation(error_time_point_list, fan_fault, 0)
        total_data_frame = pd.concat(correct_data_frame + error_data_frame, axis=0, ignore_index=True)
        total_data_frame.to_csv(os.path.join(outdir, 'valid_total.csv'))
        for index in range(len(correct_data_frame)):
            correct_data_frame[index].to_csv(os.path.join(outdir, str(
                time.strftime('%Y-%m-%d-%H%M', time.localtime(correct_time_point_list[index * 2]))) + '--' + str(
                time.strftime('%Y-%m-%d-%H%M', time.localtime(correct_time_point_list[index * 2 + 1]))) + ".csv"))
        for index in range(len(error_data_frame)):
            error_data_frame[index].to_csv(os.path.join(outdir, str(
                time.strftime('%Y-%m-%d-%H%M', time.localtime(error_time_point_list[index * 2]))) + '--' + str(
                time.strftime('%Y-%m-%d-%H%M', time.localtime(error_time_point_list[index * 2 + 1]))) + ".csv"))

    def blobvalue_extraction(self):
        '''
        提取waveType为4的样本的blobvalue，进行解码得到原始时间序列并保存到output_data/sequential_data.csv
        '''
        waveType2 = pd.read_csv('output_data/waveType2_clean.csv')

        blobvalue_list = waveType2['blobvalue'].copy()
        label = waveType2['label'].copy()
        sp = Preprocessor()
        output_df = pd.DataFrame()
        for index in range(len(blobvalue_list)):
            temp = sp.revertgzip(blobvalue_list[index])
            output_df = output_df.append(pd.Series(temp), ignore_index=True)

        output_df['label'] = label
        output_df['ml_id'] = waveType2['ml_id'].copy()
        output_df.to_csv('output_data/sequential_data.csv')
