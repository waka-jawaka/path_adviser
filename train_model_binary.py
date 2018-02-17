import numpy as np
import cv2
import time
import os
import pandas as pd
from collections import deque
from models import inception_v3 as googlenet
from random import shuffle


FILE_I_END = 10

WIDTH = 480
HEIGHT = 270
LR = 1e-3
EPOCHS = 10
BATCH_SIZE = 15

MODEL_NAME = 'Binary Classifier v3'
PREV_MODEL = 'Binary Classifier v3'

LOAD_MODEL = True

obstacle = [1, 0]
no_obstacle = [0, 1]

model = googlenet(WIDTH, HEIGHT, 3, LR, output=2, model_name=MODEL_NAME)

if LOAD_MODEL:
    model.load(PREV_MODEL)
    print('We have loaded a previous model!!!!')
    
for e in range(EPOCHS):
    data_order = [i for i in range(1, FILE_I_END + 1)]
    shuffle(data_order)
    for count, i in enumerate(data_order):
        try:
            file_name = 'processed_dataset/{}.npy'.format(i)
            data = np.load(file_name)
            print('{}.npy'.format(i), len(data))

            np.random.shuffle(data)
            train = data[:-40]
            validation = data[-40:]

            X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 3)
            Y = [i[1] for i in train]

            test_x = np.array([i[0] for i in validation]).reshape(-1, WIDTH, HEIGHT, 3)
            test_y = [i[1] for i in validation]

            model.fit(
				{'input': X}, 
				{'targets': Y}, 
				n_epoch=EPOCHS,
				batch_size=BATCH_SIZE, 
				validation_set=({'input': test_x}, {'targets': test_y}), 
                snapshot_step=2500, 
				show_metric=True, 
				run_id=MODEL_NAME
			)

            if count % 10 == 0:
                print('SAVING MODEL!')
                model.save(MODEL_NAME)
                    
        except Exception as e:
            print(str(e))
