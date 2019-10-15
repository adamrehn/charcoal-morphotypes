#!/usr/bin/env python3
from common import Configuration, Logger, TimeLogger, ProgressLogger, SegmentationNetwork
from natsort import natsorted
import glob, os

# Retrieve the list of input files
inputDir = Configuration.path('segmentation_data_preprocessed')
outputDir = Configuration.path('segmentation_data_sliced')
inputFiles = natsorted(glob.glob(os.path.join(inputDir, '*.png')))

# Progress output
print('Slicing preprocessed data for the segmentation network ({} files)...'.format(len(inputFiles)))

# Keep track of processing progress and timing
numFiles = len(inputFiles)
timer = TimeLogger(numFiles, 'file')
progress = ProgressLogger(numFiles)

# Process each input file
for filenum, infile in enumerate(inputFiles):
	
	# Progress output
	progress.progress(filenum, 'Slicing input file "{}"...'.format(infile))
	
	# Slice the file
	SegmentationNetwork.sliceToDir(infile, outputDir, includeMask=True, warnOnOverwrite=True)

# Progress output
timer.end()
Logger.success('slicing complete ({}).'.format(timer.report()))
