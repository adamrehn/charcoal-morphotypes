import humanfriendly, sys, time
from termcolor import colored

class Logger:
	'''
	Provides general logging functionality
	'''
	
	@staticmethod
	def success(message, bold=True):
		'''
		Reports a successful operation
		'''
		Logger._print('green', 'Success: ' + message, bold)
	
	@staticmethod
	def warning(message, bold=True):
		'''
		Emits a warning
		'''
		Logger._print('yellow', 'Warning: ' + message, bold)
	
	@staticmethod
	def error(message, bold=True):
		'''
		Emits an error and raises an exception
		'''
		Logger._print('red', 'Error: ' + message, bold)
		raise RuntimeError(message)
	
	@staticmethod
	def _print(colour, message, bold):
		attributes = ['bold'] if bold == True else []
		print(colored(message, color=colour, attrs=attributes), file=sys.stderr)


class TimeLogger(object):
	'''
	Provides functionality for timing processing operations and reporting the results
	'''
	
	def __init__(self, count = 0, unit = None):
		'''
		Creates a new logger for measuring the processing of the specified number of inputs
		(or simply total processing time if the specified number of inputs is zero)
		'''
		self._count = count
		self._unit = unit
		self.start()
		self.end()
	
	def start(self):
		'''
		Sets the starting time to the current time
		'''
		self._start = time.time()
	
	def end(self):
		'''
		Sets the completion time to the current time
		'''
		self._end = time.time()
	
	def report(self):
		'''
		Generates a report string summarising the processing time statistics
		'''
		total = self._end - self._start
		if self._count > 0:
			average = total / self._count
			return '{} {}(s), {} total, {} average per {}'.format(
				self._count,
				self._unit,
				humanfriendly.format_timespan(total),
				humanfriendly.format_timespan(average),
				self._unit
			)
		else:
			return '{} total'.format(humanfriendly.format_timespan(total))


class ProgressLogger(object):
	'''
	Provides functionality for determining and reporting progress of processing tasks
	'''
	
	def __init__(self, count):
		'''
		Creates a new logger for logging the processing of the specified number of inputs
		'''
		self._count = count
	
	def progress(self, current, message, sameLine = False):
		'''
		Determines the current percentage of progress and logs it
		'''
		percent = ((current+1) / self._count) * 100.0
		print(colored('[{:.1f}%] '.format(percent), color='cyan', attrs=['bold']) + message, end = '\r' if sameLine == True else '\n')
