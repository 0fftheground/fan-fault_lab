import time
import binascii
import gzip
import numpy as np


class SimplePreprocessor:
    def __init__(self):
        return

    def filesegmentation(self, file_list, data_frame, label):

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
        b = blobvalue[2:]  # 截取掉'0x'
        c = binascii.a2b_hex(b)  # 转换成ASCii编码的字符串
        d = str(gzip.decompress(c), encoding="utf8")
        e = d.split('$')
        for index in range(len(e)):
            if e[index] == '':
                del e[index]
        d = np.array(e)
        return d.astype('float')
