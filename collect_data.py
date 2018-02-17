import numpy as np
from grabscreen import grab_screen
import cv2
import time
from getkeys import key_check
import os

OBSTACLE = [0, 1]
NO_OBSTACLE = [1, 0]

starting_value = 1

while True:
    file_name = 'dataset/training_data-{}.npy'.format(starting_value)

    if os.path.isfile(file_name):
        print('File exists, moving along',starting_value)
        starting_value += 1
    else:
        print('File does not exist, starting fresh!',starting_value)
        break

def keys_to_output(keys):
    output = np.array([0,1], dtype=np.uint8)
	if 'W' in keys:
		output = [1, 0]
    return output

def main(file_name, starting_value):
    file_name = file_name
    starting_value = starting_value
    training_data = []
    for i in list(range(10))[::-1]:
        print(i+1)
        time.sleep(1)

    last_time = time.time()
    paused = False
    print('STARTING!!!')
    while(True):
        
        if not paused:
            screen = grab_screen(region=(0,40,1920,1120))
            last_time = time.time()
            # resize to something a bit more acceptable for a CNN
            screen = cv2.resize(screen, (480,270))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            
            keys = key_check()
            output = keys_to_output(keys)
            training_data.append([screen,output])

            last_time = time.time()
            if len(training_data) == 100:
                print(len(training_data))
                np.save(file_name,training_data)
                print('SAVED')
                training_data = []
                starting_value += 1
                file_name = 'dataset/training_data-{}.npy'.format(starting_value)

                    
        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)

main(file_name, starting_value)
