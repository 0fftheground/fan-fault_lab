from model._globals import Config
import model.helpers as helpers
import model.modeling as models
from datetime import datetime as dt
from keras.utils import plot_model
import os
import numpy as np
import csv
import matplotlib.pyplot as plt
from utils.preprocessing import Preprocessor

if __name__ == "__main__":
    # Preprocessor().fft_preprocessing()
    config = Config("config.yaml")
    # _id = dt.now().strftime("%Y-%m-%d_%H")
    _id = '2019-06-20_09'
    helpers.make_dirs(_id)
    logger = helpers.setup_logging(config, _id)
    model = None
    train_data = np.load('output_data/fft/fft_po2_train_data.npy')
    train_data_scaled = train_data[:, :, np.newaxis]
    model = models.get_gru_vae_model(train_data_scaled, logger, _id)
    # plot_model(model, to_file=os.path.join('result/%s/models' % _id, "gru_vae_model.png"),show_shapes=True)
    rmse_list = []
    temp = model.predict(train_data_scaled)
    np.save(os.path.join('result/%s/models' % _id, "fft_po2_predict"), temp)
    # for i in range(train_data.shape[0]-1):
    #     temp = model.predict(train_data_scaled[i:i+1])
    #     rmse_list.append(np.sqrt(np.mean((train_data_scaled[i:i+1]-temp[0])**2)))
    # np.save(os.path.join('result/%s/models' % _id, "fft_po2_rmse"), np.array(rmse_list))
    # line0_predict = model.predict(train_data_scaled[201:202])
    # line1_predict = model.predict(train_data_scaled[101:102])
    # np.save(os.path.join('result/%s/models' % _id, "fft_ne2_predict_data_201"), line0_predict)
    # np.save(os.path.join('result/%s/models' % _id, "fft_ne2_predict_data_101"), line1_predict)
    # plt.title('comparison diagram(positive_sample_number:201)')
    # plt.ylabel("Magnitude")
    # plt.xlabel("Frequency")
    # plt.plot(train_data_scaled[201:202].reshape(2048), linestyle='dashdot', linewidth=1, color='green', label='origin')
    # plt.plot(line0_predict[0].reshape(2048), linestyle='dotted', linewidth=1, color='red', label='prediction')
    # plt.legend(loc='best')
    # plt.savefig(os.path.join('result/%s/models' % _id, "fft_po2_train_data_201_compare.png"), dpi=300)
