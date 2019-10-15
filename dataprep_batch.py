#!/usr/bin/env python3
from common import Configuration
from natsort import natsorted
import glob, math, os, shutil, sys
import numpy as np

# The approximate batch size that we will group the sliced images into
BATCH_SIZE = 100

# Retrieve the list of image files
imageDir = Configuration.path('segmentation_data_sliced')
images = natsorted(glob.glob(os.path.join(imageDir, '*.png')))

# Split the list of images into batches
numDirs = math.ceil(len(images) / BATCH_SIZE)
batches = np.array_split(images, numDirs)

# Move each batch into a subdirectory
for index, batch in enumerate(batches):
	
	# Create the subdirectory for the batch
	subdir = os.path.join(imageDir, str(index))
	if os.path.exists(subdir) == False:
		os.makedirs(subdir)
	
	# Move each of the files into the subdirectory
	for file in batch.tolist():
		
		# Progress output
		print('Moving {} to {}...'.format(file, subdir))
		sys.stdout.flush()
		
		# Move the file
		shutil.move(file, os.path.join(subdir, os.path.basename(file)))

# Progress output
print('Done.')
