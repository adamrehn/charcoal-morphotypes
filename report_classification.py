#!/usr/bin/env python3
from common import Logger, ProgressLogger, TimeLogger, Configuration, Utility, ValidationUtils, ClassificationNetwork
import cv2, glob, os, re, shutil, sys
import numpy as np

# Our report HTML template
REPORT_TEMPLATE = '''
<!doctype html>
<html>
	<head>
		<title>Validation Report: Morphotype Classification Neural Network</title>
		
		<style type="text/css">
			h1
			{
				text-align: center;
				padding-bottom: 1em;
			}
			
			body > table
			{
				table-layout: fixed;
				margin: 0 auto;
			}
			
			body > table td
			{
				text-align: center;
				width: 30%;
			}
			
			table img
			{
				max-width: 99%;
				max-height: 10em;
			}
			
			tr.predicted td {
				font-weight: bold;
			}
			
			tr.predicted.correct td {
				color: forestgreen;
			}
			
			tr.predicted.incorrect td {
				color: red;
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
outputDir = Configuration.path('classification_reports')
for file in glob.glob(os.path.join(outputDir, '*.png')) + glob.glob(os.path.join(outputDir, '*.html')):
	os.unlink(file)

# Retrieve the list of validation images for the morphotype classification neural network
validationData = ClassificationNetwork.getValidationData()
validationImages = sorted([os.path.join(validationData.directory, p) for p in validationData.filenames])

# Load the network from the last saved checkpoint and compute our overall validation accuracy
model, metadata = ClassificationNetwork.load()
accuracy = ValidationUtils.computeValidationAccuracy(model, validationData)

# Compute our overall validation accuracy
result = model.evaluate_generator(validationData)
accuracyIndex = model.metrics_names.index('acc')
accuracy = result[accuracyIndex]

# Progress output
numImages = len(validationImages)
print('Generating validation report for the morphotype classification network ({} images)...'.format(numImages))

# Keep track of processing progress and timing
timer = TimeLogger(numImages, 'image')
progress = ProgressLogger(numImages)

# Process each input file
tableRows = []
for filenum, infile in enumerate(validationImages):
	
	# Progress output
	progress.progress(filenum, 'Processing validation image "{}"...'.format(infile))
	
	# Copy the input image to the output directory
	imageName = os.path.basename(infile)
	raster = cv2.imread(infile, cv2.IMREAD_COLOR)
	cv2.imwrite(os.path.join(outputDir, imageName), raster)
	
	# Perform inference and determine which label has the highest probability
	data = ClassificationNetwork.readImage(infile)
	probabilities, labels = ClassificationNetwork.infer(model, metadata, data)
	predictedIndex = np.argmax(probabilities[0], axis=0).astype(np.uint8)
	predictedLabel = [k for k in labels.keys() if labels[k] == predictedIndex][0]
	
	# Build a mapping from labels to probabilities
	probabilityMap = {}
	for label, index in labels.items():
		probabilityMap[label] = probabilities[0][index]
	
	# Generate the table row for our HTML report
	groundTruth = os.path.basename(os.path.dirname(infile))
	isCorrect = predictedLabel == groundTruth
	row = '<tr><td><img src="{}"></td>'.format(imageName)
	row += '<td>{}</td>'.format(groundTruth)
	row += '<td><table><tbody>'
	for label, probability in probabilityMap.items():
		row += '<tr class="{} {}"><td>{}</td><td>{:.10f}</td></tr>'.format(
			'predicted' if label == predictedLabel else '',
			'correct' if isCorrect == True else 'incorrect',
			label,
			probability
		)
	row += '</tbody></table></td>'
	tableRows.append(row)

# Save the HTML report
html = REPORT_TEMPLATE.replace('$$__ACCURACY__$$', '{:.2f}%'.format(accuracy * 100.0))
html = html.replace('$$__ROWS__$$', '\n'.join(tableRows))
Utility.writeFile(os.path.join(outputDir, '_report.html'), html)

# Progress output
timer.end()
Logger.success('report generation complete ({}).'.format(timer.report()))
