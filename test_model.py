import numpy as np
from models import inception_v3 as googlenet

WIDTH = 480
HEIGHT = 270
LR = 1e-3
EPOCHS = 1

model = googlenet(WIDTH, HEIGHT, 3, LR, output=2)
MODEL_NAME = 'Binary Classifier v3'
model.load(MODEL_NAME)

def calculate_accuracy(pred, test):
	correct = 0
	for i in range(len(test)):
		if np.argmax(pred[i]) == np.argmax(test[i][1]):
			correct += 1
	return correct	

def main():
	test = np.load('processed_dataset/test.npy')
	predictions = [model.predict([item[0].reshape(WIDTH,HEIGHT,3)])[0]
					for item in test]
	for i in range(len(test)):
		print(predictions[i], '-->', test[i][1])
	correct = calculate_accuracy(predictions, test)
	print("\nCorrect: {} of {} test cases.".format(correct, len(test)))

if __name__ == '__main__':
	main()       
