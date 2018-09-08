import yaml
import fileinput
import os


def parse(config_path):
  """Parse a config file into a config object.
  """
  with open(config_path) as file:
    config = yaml.load(file.read())
  return config