import os


# Get current directory
print os.getcwd()
print os.path.realpath(__file__)

print os.path.split(__file__)