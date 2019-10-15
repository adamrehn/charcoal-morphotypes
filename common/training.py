from .configuration import Configuration
from .logging import Logger, TimeLogger
import glob, json, keras, os
import numpy as np

class TrainingUtils:
	
	@staticmethod
	def computeWeights(trainingData):
		'''
		Computes the loss weightings for a set of classification labels, based on the number of training
		samples present for each class. This function expects a Keras Sequence with a precomputed `classes`
		attribute (e.g. the output of an ImageDataGenerator flow_from_directory() call.)
		
		The weights returned by this function can be used as the `classWeights` argument to the
		TrainingUtils.trainWithCheckpoints() function.
		'''
		
		# Determine how many training samples are present for each classification label
		classes = np.unique(trainingData.classes).tolist()
		samples = [np.nonzero(trainingData.classes == c)[0].shape[0] for c in classes]
		
		# Compute the weights for each classification label
		total = np.sum(samples)
		weights = {}
		for c in classes:
			weights[c] = 1.0 - (samples[c] / total)
		return weights
	
	
	@staticmethod
	def checkpointPath(name, extension):
		'''
		Returns the full path to the checkpoint file for the specified model name
		'''
		return os.path.join(Configuration.path('checkpoints'), '{}-checkpoint.{}'.format(name, extension))
	
	
	@staticmethod
	def loadCheckpoint(name, customObjects = None):
		'''
		Loads a previously-saved checkpoint of a Keras model, along with its saved metadata
		
		Returns a tuple of (model, metadata)
		'''
		
		# Read the metadata JSON file
		metadata = None
		with open(TrainingUtils.checkpointPath(name, 'json'), 'r') as jsonFile:
			metadata = json.load(jsonFile)
		
		# Load the model checkpoint
		model = keras.models.load_model(TrainingUtils.checkpointPath(name, 'h5'), customObjects)
		return (model, metadata)
	
	
	@staticmethod
	def trainWithCheckpoints(model, name, metadata, trainingData, validationData, batchSize, epochs, accuracyMetric, earlyStoppingPatience = 100, classWeights = None):
		'''
		Takes a compiled Keras model and performs training, checkpointing the trained network
		'''
		
		# Save the metadata to disk in JSON format
		with open(TrainingUtils.checkpointPath(name, 'json'), 'w') as jsonFile:
			json.dump(metadata, jsonFile)
		
		# Checkpoint the model after each epoch, only keeping the checkpoint with the best accuracy
		checkpoint = keras.callbacks.ModelCheckpoint(
			TrainingUtils.checkpointPath(name, 'h5'),
			save_best_only = True,
			monitor = accuracyMetric,
			mode = 'max',
			verbose = 1
		)
		
		# Stop training if the validation accuracy has not increased during the specified number of epochs
		# (This typically indicates that the model has started over-fitting and the accuracy will not recover)
		earlyStop = keras.callbacks.EarlyStopping(
			monitor = accuracyMetric,
			patience = earlyStoppingPatience,
			mode = 'max',
			verbose = 1
		)
		
		# Enable visualisation of the training results in TensorBoard
		tensorboard = keras.callbacks.TensorBoard(
			log_dir = Configuration.path('logs'),
			batch_size = batchSize
		)
		
		# Remove any existing logs from previous training runs
		logs = glob.glob(os.path.join(Configuration.path('logs'), 'events.*'))
		for log in logs:
			os.unlink(log)
		
		# Keep track of the time taken for training to complete
		timer = TimeLogger()
		
		# Perform training for the requested number of epochs
		model.fit_generator(
			trainingData,
			epochs = epochs,
			class_weight = classWeights,
			validation_data = validationData,
			callbacks = [checkpoint, earlyStop, tensorboard]
		)
		
		# Report the total training time, as well as the average time per epoch
		timer.end()
		Logger.success('training complete ({}).'.format(timer.report()))
