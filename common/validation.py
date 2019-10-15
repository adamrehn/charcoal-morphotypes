class ValidationUtils:
	
	@staticmethod
	def computeValidationAccuracy(model, validationData):
		'''
		Determines the validation accuracy of the supplied neural network
		'''
		result = model.evaluate_generator(validationData)
		accuracyMetric = [m for m in model.metrics_names if 'acc' in m][0]
		accuracyIndex = model.metrics_names.index(accuracyMetric)
		return result[accuracyIndex]
