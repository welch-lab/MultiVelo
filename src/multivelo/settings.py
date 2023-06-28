import os

"""Settings
"""

global VERBOSITY

# cwd: The current working directory
global CWD

global LOG_FOLDER

global LOG_FILENAME

global GENE

# if os.path.isfile("settings.txt"):
#     with open("settings.txt", "r") as sfile:
#         VERBOSITY = int(sfile.readline())

#         # get substrings from the file to avoid including newlines
#         CWD = sfile.readline()[:-1]
#         LOG_FOLDER = sfile.readline()[:-1]
#         LOG_FILENAME = sfile.readline()[:-1]
# else:

VERBOSITY = 1
CWD = os.path.abspath(os.getcwd())
LOG_FOLDER = os.path.join(CWD, "../logs")
LOG_FILENAME = None
GENE = None
