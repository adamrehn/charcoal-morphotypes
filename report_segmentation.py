#!/usr/bin/env python3
from common import Logger, ProgressLogger, TimeLogger, Configuration, Utility, ValidationUtils, SegmentationNetwork
import cv2, glob, maskutils, math, os, re, sys
import numpy as np

# Our report HTML template
REPORT_TEMPLATE = '''
<!doctype html>
<html>
	<head>
		<title>Validation Report: Segmentation Neural Network</title>
		
		<style type="text/css">
			h1
			{
				text-align: center;
				padding-bottom: 1em;
			}
			
			table
			{
				table-layout: fixed;
				margin: 0 auto;
			}
			
			table td
			{
				text-align: center;
				width: 19%;
			}
			
			table img {
				max-width: 99%;
			}
		</style>
	</head>
	
	<body>
		<h1>Overall Validation Accuracy:<br>$$__ACCURACY__$$</h1>
		<table>
			<thead>
				<tr>
					<th>Input Image</th>
					<th>Ground Truth</th>
					<th>Prediction</th>
					<th>Identified Particles</th>
					<th>Particle Details</th>
				</tr>
			</thead>
			<tbody>
				$$__ROWS__$$
			</tbody>
		</table>
	</body>
</html>
'''

# Retrieve the path to the report output directory and ensure it is empty
outputDir = Configuration.path('segmentation_reports')
for file in glob.glob(os.path.join(outputDir, '*.png')) + glob.glob(os.path.join(outputDir, '*.html')):
	os.unlink(file)

# Retrieve the list of pre-windowed validation images for the segmentation neural network
validationData = SegmentationNetwork.getValidationData()
preWindowed = validationData.images

# Strip the window-related suffixes to determine the original (non-windowed) image filenames
stripRegex = re.compile('\\-[0-9]+\\-window[0-9]+')
origDir = Configuration.path('segmentation_data_preprocessed')
originalImages = set([os.path.join(origDir, stripRegex.sub('', os.path.basename(p))) for p in preWindowed])

# The first image may have had some of its windows in the training dataset,
# so we remove it from our list to ensure all data is completely unseen
originalImages = sorted(originalImages)[1:]

# Load the network from the last saved checkpoint and compute our overall validation accuracy
model, metadata = SegmentationNetwork.load()
accuracy = ValidationUtils.computeValidationAccuracy(model, validationData)

# Progress output
numImages = len(originalImages)
print('Generating validation report for the segmentation network ({} images)...'.format(numImages))

# Keep track of processing progress and timing
timer = TimeLogger(numImages, 'image')
progress = ProgressLogger(numImages)

# Process each input file
tableRows = []
for filenum, infile in enumerate(originalImages):
	
	# Progress output
	progress.progress(filenum, 'Processing validation image "{}"...'.format(infile))
	
	# Split the ground-truth binary mask from the other image channels
	raster = cv2.imread(infile, cv2.IMREAD_UNCHANGED)
	channels, groundTruth = maskutils.splitAlphaMask(raster)
	
	# Perform inference on the image data
	probabilities = SegmentationNetwork.infer(model, metadata, channels)
	prediction = np.argmax(probabilities, axis=2).astype(np.uint8)
	
	# Denoise the predicted mask and retrieve the exact bounding box of each individual particle
	denoised = SegmentationNetwork.denoise(prediction)
	sliced = cv2.cvtColor(denoised * 255, cv2.COLOR_GRAY2RGB)
	particleDetails = []
	borderColour = (0,0,255)
	borderThickness = math.floor(channels.shape[0] * 0.01)
	for instanceNum, instance in enumerate(SegmentationNetwork.sliceExact(channels, denoised)):
		
		# Draw a border around the bounding box
		box = cv2.boxPoints(instance)
		box = np.int0(box)
		cv2.drawContours(sliced, [box], 0, borderColour, borderThickness)
		
		# Store the size and length/width ratio of the particle
		centre, size, rotation = instance
		particleDetails.append({
			'length': np.max(size),
			'width':  np.min(size),
			'ratio':  SegmentationNetwork.particleElongationRatio(size)
		})
	
	# Save the images to the output directory
	imageBase = os.path.join(outputDir, os.path.basename(infile).replace('.png', ''))
	imageChannels = '{}.channels.png'.format(imageBase)
	imageGroundTruth = '{}.groundtruth.png'.format(imageBase)
	imagePrediction = '{}.prediction.png'.format(imageBase)
	imageSliced = '{}.sliced.png'.format(imageBase)
	cv2.imwrite(imageChannels, channels)
	cv2.imwrite(imageGroundTruth, groundTruth)
	cv2.imwrite(imagePrediction, prediction * 255)
	cv2.imwrite(imageSliced, sliced)
	
	# Generate the table row for our HTML report
	tableRows.append('<tr><td><img src="{}"></td><td><img src="{}"></td><td><img src="{}"></td><td><img src="{}"></td><td>{}</td></tr>'.format(
		os.path.basename(imageChannels),
		os.path.basename(imageGroundTruth),
		os.path.basename(imagePrediction),
		os.path.basename(imageSliced),
		''.join(['<li><strong>Size:</strong> {:.0f} x {:.0f}, <strong>Ratio:</strong> {:.2f}</li>'.format(particle['length'], particle['width'], particle['ratio']) for particle in particleDetails])
	))

# Save the HTML report
html = REPORT_TEMPLATE.replace('$$__ACCURACY__$$', '{:.2f}%'.format(accuracy * 100.0))
html = html.replace('$$__ROWS__$$', '\n'.join(tableRows))
Utility.writeFile(os.path.join(outputDir, '_report.html'), html)

# Progress output
timer.end()
Logger.success('report generation complete ({}).'.format(timer.report()))
