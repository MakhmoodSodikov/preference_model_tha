import logging


ENV_LEVEL_NUM = 21
AGENT_LEVEL_NUM = 22
SETUP_LEVEL_NUM = 23
DEBUG = False

logging.addLevelName(ENV_LEVEL_NUM, "ENV")
logging.addLevelName(AGENT_LEVEL_NUM, "AGENT")
logging.addLevelName(SETUP_LEVEL_NUM, "SETUP")


class CustomLogger(logging.Logger):
	def setup(self, message, *args, **kws):
		if self.isEnabledFor(SETUP_LEVEL_NUM):
			self._log(SETUP_LEVEL_NUM, message, args, **kws)

	def env(self, message, *args, **kws):
		if self.isEnabledFor(ENV_LEVEL_NUM):
			self._log(ENV_LEVEL_NUM, message, args, **kws)

	def agent(self, message, *args, **kws):
		if self.isEnabledFor(AGENT_LEVEL_NUM):
			self._log(AGENT_LEVEL_NUM, message, args, **kws)


logging.setLoggerClass(CustomLogger)


def get_logger(name: str) -> CustomLogger:
	return logging.getLogger(name)  # type: ignore


def setup_logging(debug: bool = False, log_dir: str = "run_logs"):
	level = logging.DEBUG if debug else logging.INFO
	logging.basicConfig(level=level, format='[%(levelname)s]: %(message)s' + '\n' + "="*30)
