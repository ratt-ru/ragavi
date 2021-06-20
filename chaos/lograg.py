import logging
import sys
import textwrap
import traceback as tb
import warnings

LOG_FILE = "ragavi.log"
ROOT_LOGGER_LEVEL = "INFO"
ROOT_HANDLER_LEVEL = "INFO"

# set up the general logger
# capture only a single instance of a matching repeated warning
warnings.filterwarnings("module")
# capture warnings from all modules
logging.captureWarnings(True)

# only get data from ragavi modules
logger_filter = logging.Filter("")
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
    snitch = get_logger(logging.getLogger(__name__))
    snitch.error("Oops ... uncaught exception occurred!")
    _ = [snitch.error(_) for _ in trace if _ != ""]

sys.excepthook = _handle_uncaught_exceptions
warnings.formatwarning = wrap_warning_text