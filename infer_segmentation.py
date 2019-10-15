#!/usr/bin/env python3
from common import ProgressLogger, SegmentationNetwork
import cv2, os, sys
import numpy as np

# Our progress callback class
class SegmentationProgress(object):
	def __init__(self):
		self.logger = None
	
	def progress(self, current, total):
		
		# Create the progress logger once we know the total number of steps
		if self.logger is None:
			self.logger = ProgressLogger(total)
		
		# Display progress output to the user
		self.logger.progress(current, 'Performing inference...', sameLine = True)


# We require two command-line arguments: the image to classify and the output mask filename
if len(sys.argv) > 2:
	
	# Perform inference using the segmentation neural network
	progress = SegmentationProgress()
	data = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
	probabilities = SegmentationNetwork.loadAndInfer(data, lambda current, total: progress.progress(current, total))
	
	# Generate the output mask by performing an argmax operation on the raw probabilities
	mask = np.argmax(probabilities, axis=2)
	
	# Save the mask to the output file
	cv2.imwrite(sys.argv[2], mask * 255)
	print('\nDone.')
	
else:
	script = os.path.basename(__file__)
	print(script + ': segments an image using the segmentation neural network')
	print()
	print('Usage:')
	print(script + ' INFILE OUTFILE')
