#!/usr/bin/env python3
from common import ClassificationNetwork
import os, sys

# We require one command-line argument: the image to classify
if len(sys.argv) > 1:
	
	# Perform inference using the morphotype classification neural network
	data = ClassificationNetwork.readImage(sys.argv[1])
	probabilities, labels = ClassificationNetwork.loadAndInfer(data)
	
	# Build a mapping from labels to probabilities
	probabilityMap = {}
	for label, index in labels.items():
		probabilityMap[label] = probabilities[0][index]
	print(probabilityMap)
	
else:
	script = os.path.basename(__file__)
	print(script + ': classifies an image using the morphotype classification neural network')
	print()
	print('Usage:')
	print(script + ' INFILE')
