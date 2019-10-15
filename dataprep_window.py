#!/usr/bin/env python3
from common import Configuration, Logger, TimeLogger, ProgressLogger, SegmentationNetwork
from natsort import natsorted
import glob, os

# Retrieve the list of input files
inputDir = Configuration.path('segmentation_data_sliced')
outputDir = Configuration.path('segmentation_data_windowed')
inputFiles = natsorted(glob.glob(os.path.join(inputDir, '*.png')))

# Progress output
print('Windowing sliced data for the segmentation network ({} files)...'.format(len(inputFiles)))

# Keep track of processing progress and timing
numFiles = len(inputFiles)
timer = TimeLogger(numFiles, 'file')
progress = ProgressLogger(numFiles)

# Process each input file
for filenum, infile in enumerate(inputFiles):
	
	# Progress output
	progress.progress(filenum, 'Windowing input file "{}"...'.format(infile))
	
	# Slice the file
	SegmentationNetwork.windowToDir(infile, outputDir, warnOnOverwrite=True)

# Progress output
timer.end()
Logger.success('windowing complete ({}).'.format(timer.report()))
