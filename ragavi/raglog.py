import logging
import os
import sys
import textwrap
import warnings

LOG_FILE = "ragavi.log"
ROOT_LOGGER_LEVEL = "DEBUG"
ROOT_HANDLER_LEVEL = "INFO"


def wrap_warning_text(message, category, filename, lineno, file=None,
                      line=None):
    wrapper = textwrap.TextWrapper(initial_indent=''.rjust(51),
                                   break_long_words=True,
                                   subsequent_indent=''.rjust(49),
                                   width=160)
    message = wrapper.fill(str(message))
    return "%s:%s:\n%s:\n%s" % (filename, lineno,
                                category.__name__.rjust(64), message)


def __handle_uncaught_exceptions(extype, exval, extraceback):
    """Function to Capture all uncaught exceptions into the log file

       Parameters to this function are acquired from sys.excepthook. This
       is because this function overrides :obj:`sys.excepthook`
      `Sys module excepthook <https://docs.python.org/3/library/sys.html#sys.excepthook>`_

    """
    message = "Oops ... !"
    logger.error(message, exc_info=(extype, exval, extraceback))


# capture only a single instance of a matching repeated warning
warnings.filterwarnings("module")

# capture warnings from all modules
logging.captureWarnings(True)

try:
    cols, rows = os.get_terminal_size(0)
except:
    # for python2
    cols, rows = (100, 100)

# only get data from ragavi modules
logger_filter = logging.Filter("ragavi")

# setup root logger and defaults
logger = logging.getLogger()
logger.setLevel(ROOT_LOGGER_LEVEL)

# warnings logger
w_logger = logging.getLogger("py.warnings")
w_logger.setLevel("WARNING")

# console handler
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler(LOG_FILE)

c_handler.setLevel(ROOT_HANDLER_LEVEL)
f_handler.setLevel(ROOT_HANDLER_LEVEL)

f_formatter = logging.Formatter(
    "%(asctime)s - %(name)-20s - %(levelname)-10s - %(message)s",
    datefmt="%d.%m.%Y@%H:%M:%S")

c_handler.setFormatter(f_formatter)
c_handler.addFilter(logger_filter)

f_handler.setFormatter(f_formatter)
f_handler.addFilter(logger_filter)

logger.addFilter(logger_filter)
logger.addHandler(c_handler)
logger.addHandler(f_handler)

w_logger.addHandler(f_handler)


sys.excepthook = __handle_uncaught_exceptions
warnings.formatwarning = wrap_warning_text
