#!/usr/bin/env python3
from common import Configuration, Logger, TimeLogger, ProgressLogger, SegmentationNetwork
import cv2, glob, mergetiff, os, subprocess, tempfile, time
from natsort import natsorted
import numpy as np

# The layer numbers for the RGB channel data and the binary mask
# (The defaults specified here are correct for the TIFF files Emma prepared in Photoshop)
CHANNELS_LAYER = 1
MASK_LAYER     = 0

# Retrieve the list of input files
inputDir = Configuration.path('segmentation_data_raw')
outputDir = Configuration.path('segmentation_data_preprocessed')
inputFiles = natsorted(glob.glob(os.path.join(inputDir, '**', '*.tif')))

# Progress output
print('Preprocessing raw data for the segmentation network ({} files)...'.format(len(inputFiles)))

# Keep track of processing progress and timing
numFiles = len(inputFiles)
timer = TimeLogger(numFiles, 'file')
progress = ProgressLogger(numFiles)

# Process each input file
for filenum, infile in enumerate(inputFiles):
	
	# Progress output
	progress.progress(filenum, 'Preprocessing input file "{}"...'.format(infile))
	
	# Create a temporary directory to hold our intermediate files
	with tempfile.TemporaryDirectory() as tempDir:
		
		# Create the filenames for our intermediate files and output file
		channelsFile = os.path.join(tempDir, 'channels.tif')
		maskFile = os.path.join(tempDir, 'mask.tif')
		outfile = os.path.join(outputDir, infile.replace(inputDir + '/', '').replace('/', '-').replace(' ', '-').replace('.tif', '.png'))
		
		# Use ImageMagick to extract the layers containing the RGB channel data and the binary mask
		subprocess.call(['magick', 'convert', '-quiet', '{}[{}]'.format(infile, CHANNELS_LAYER), channelsFile])
		subprocess.call(['magick', 'convert', '-quiet', '{}[{}]'.format(infile, MASK_LAYER), maskFile])
		
		# Read the mask in and discard any redundant colour channels
		mask = mergetiff.rasterFromFile(maskFile)
		if len(mask.shape) > 2:
			mask = mask[:,:,0]
		
		# Perform binary thresholding on the mask to remove any anti-aliasing
		ret, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
		
		# Denoise the mask
		mask = SegmentationNetwork.denoise(mask)
		
		# Invert the mask, since Emma painted the particles black rather than white in the mask layer,
		# whereas the neural network uses classification label 0 for "not particle" and 1 for "particle"
		mask = 255 - mask
		
		# Open the file containing the RGB channel data
		channels = mergetiff.rasterFromFile(channelsFile)
		if channels.shape[2] > 3:
			channels = channels[:,:,0:3]
		elif channels.shape[2] < 3:
			Logger.error('could not extract RGB channel data!')
		
		# Merge the RGB channels with the modified mask
		shape = (channels.shape[0], channels.shape[1], 4)
		merged = np.zeros(shape, dtype=channels.dtype)
		merged[:,:,0:3] = channels
		merged[:,:,3] = mask
		
		# Write the output file (convert RGB to BGR for OpenCV)
		if os.path.exists(outfile) == True:
			Logger.warning('overwriting existing file {}'.format(outfile))
		cv2.imwrite(outfile, merged[...,[2,1,0,3]])

# Progress output
timer.end()
Logger.success('preprocessing complete ({}).'.format(timer.report()))
