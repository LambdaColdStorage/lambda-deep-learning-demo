"""
Copyright 2018 Lambda Labs. All Rights Reserved.
Licensed under
==========================================================================

Read a stream of text and clean it up


"""

import argparse


def main():
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--input_file",
                      help="Path for input text file",
                      default="")
  parser.add_argument("--output_file",
                      help="Path for output text file",
                      default="")


  args = parser.parse_args()

if __name__ == "__main__":
  main()