import logging
import os
import sys
import textwrap
import traceback as tb
import warnings

snitch = logging.getLogger(__name__)

LOG_FILE = "ragavi.log"
ROOT_LOGGER_LEVEL = "INFO"
ROOT_HANDLER_LEVEL = "INFO"

# set up the general logger
# capture only a single instance of a matching repeated warning
warnings.filterwarnings("module")
# capture warnings from all modules
logging.captureWarnings(True)

# only get data from ragavi modules
logger_filter = logging.Filter("chaos")
f_formatter = logging.Formatter(
    "%(asctime)s - %(name)-20s - %(levelname)-10s - %(message)s",
    datefmt="%d.%m.%Y@%H:%M:%S")

# console handler
c_handler = logging.StreamHandler()
c_handler.setLevel(ROOT_HANDLER_LEVEL)
c_handler.setFormatter(f_formatter)
c_handler.addFilter(logger_filter)

# warnings logger
w_logger = logging.getLogger("py.warnings")
w_logger.setLevel("CRITICAL")
w_logger.addFilter(logger_filter)

# setup root logger and defaults
root_logger = logging.getLogger()
root_logger.setLevel(ROOT_LOGGER_LEVEL)
root_logger.addFilter(logger_filter)
root_logger.addHandler(c_handler)

def get_logger(log):
    # log.filters = log.parent.filters
    # log.handlers = log.parent.handlers
    return log


def update_log_levels(root_logger, level):
    snitch.info("Testing Log level now at info")
    snitch.debug("Testing log level")
    if str(level).isnumeric():
        level = logging._levelToName[int(level)]
    else:
        level = logging._nameToLevel[level.upper()]
    root_logger.setLevel(level)
    # change the logging level for the handlers
    if root_logger.hasHandlers():
        for handler in root_logger.handlers:
            handler.setLevel(level)
    snitch.info(f"Logging level is now: {level}")
    snitch.debug("Debugging mode")

def update_logfile_name(root_logger, fname="ragavi.log"):
    f_handler = logging.FileHandler(fname)
    f_handler.setLevel(root_logger.level)
    f_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)-20s - %(levelname)-10s - %(message)s",
        datefmt="%d.%m.%Y@%H:%M:%S"))
    f_handler.addFilter(root_logger.filters[0])
    # w_logger.addHandler(f_handler)

    # append an extension if none is provided
    fname += ".log" if os.path.splitext(fname)[-1] == "" else ""

    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            root_logger.handlers.remove(handler)
    
    root_logger.addHandler(f_handler)
    snitch.debug(f"Logfile is at: {fname}")
        

def wrap_warning_text(message, category, filename, lineno, file=None,
                      line=None):
    wrapper = textwrap.TextWrapper(initial_indent=''.rjust(51),
                                   break_long_words=True,
                                   subsequent_indent=''.rjust(49),
                                   width=160)
    message = wrapper.fill(str(message))
    return "%s:%s:\n%s:\n%s" % (filename, lineno,
                                category.__name__.rjust(64), message)


def _handle_uncaught_exceptions(extype, exval, extraceback):
    """
    Function to Capture all uncaught exceptions into the log file
    Parameters to this function are acquired from sys.excepthook. This
    is because this function overrides :obj:`sys.excepthook`
    `Sys module excepthook <https://docs.python.org/3/library/sys.html#sys.excepthook>`_
    """
    trace = tb.format_exception(extype, exval, extraceback)
    trace = " ".join(trace).split("\n")
    snitch.error("Oops ... uncaught exception occurred!")
    _ = [snitch.error(_) for _ in trace if _ != ""]

sys.excepthook = _handle_uncaught_exceptions
warnings.formatwarning = wrap_warning_text
