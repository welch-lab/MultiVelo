from multivelo import settings
import os

msg_codes = {1: "update", 2: "warning", 3: "error"}


# msg = the message to print
# code = what message code this runs under
def _msg(msg, code):

    if code == 2 or code == 3:
        msg = msg_codes[code] + ": " + msg

    if settings.GENE is not None:
        msg = str(settings.GENE) + " - " + msg

    msg = str(msg) + "\n"

    return msg


# msg: the message to output
# v: at what minimum verbosity level do we output this?
# filename: the filename to output the log to
def _log(msg, code, v=0):

    # if the current verbosity is less than the minimum
    # verbosity needed for this message to print,
    # don't bother printing it
    if settings.VERBOSITY < v:
        return

    msg = _msg(msg, code=code)

    print(msg)

    if settings.LOG_FILENAME is not None:
        log_path = settings.LOG_FOLDER + "/" + settings.LOG_FILENAME

        if not os.path.isdir(settings.LOG_FOLDER):
            os.mkdir(settings.LOG_FOLDER)

        with open(log_path, "a") as logfile:
            logfile.write(msg)

            logfile.close()


def update(msg, v):
    _log(msg, code=1, v=v)


def warn(msg, v):
    _log(msg, code=2, v=v)


def error(msg, v):
    _log(msg, code=3, v=v)
