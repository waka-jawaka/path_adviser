import sys
from os.path import isdir, isfile, join
from os import makedirs, listdir

import numpy as np

folder = 'dataset/'
new_folder = 'processed_dataset/'
index = 5

TEST_DATA = []

def squash_arrays(files, path):
	arrays = [np.load(f) for f in files]
	res = np.concatenate(arrays)
	np.random.shuffle(res)

	TEST_DATA.extend(res[-10:])
	res = res[:-10]
	
	np.save(path, res)
	del arrays
	del res


if __name__ == '__main__':
	if not isdir(new_folder):
		makedirs(new_folder)
	npy_files = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
	while index <= len(npy_files) or index == 5:
		files = npy_files[index - 5: index]
		path = join(new_folder, str(index // 5))
		squash_arrays(files, path)
		index += 5

	np.save(join(new_folder, 'test.npy'), TEST_DATA)
