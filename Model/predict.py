import os
import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from metrics import dice_loss, dice_coef, iou
from train import create_dir


H = 512
W = 512

if __name__ == "__main__":
   
    np.random.seed(11)
    tf.random.set_seed(11)

    """ Создать новую папку для сохранения результатов """
    create_dir("test_images/results")

    """ Загрузка модели """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model("model.h5")

    """ Загрузка тестовых изображений """
    data_x = glob("test_images/image/*")

    for path in tqdm(data_x, total=len(data_x)):
       
        #name = path.split("/")[-1].split(".")[0]
        name = path.split("/")[-1].split(".")[0][-1]
       
        """ Чтение изображений """
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        h, w, _ = image.shape
        x = cv2.resize(image, (W, H))
        x = x/255.0
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=0)

        """ Прогноз """
        y = model.predict(x)[0]
        y = cv2.resize(y, (w, h))
        y = np.expand_dims(y, axis=-1)

        """ Сохранение результатов """
        masked_image = image * y
        line = np.ones((h, 10, 3)) * 128
        #cat_images = np.concatenate([image, line, masked_image], axis=1)
        cat_images = np.concatenate([image, line, masked_image], axis=1)
        cv2.imwrite(f"test_images/mask/{name}.png", cat_images)
        # cv2.imwrite(f"test_images/mask/{name}+'new'.png", masked_image)