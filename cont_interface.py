import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import argparse
import json
from experiments import linear_regression, log_regression

print("\n\nWELCOME TO THE NUMPY WORKS LAB (by space1panda)!\n\nPlease, "
      "follow the instructions in the terminal to properly "
      "use this software for R&D purposes\n\n")

while True:
    print("Pick up the experiment you want to run: \n1. polynom linear regression "
          "\n2. binary logistic regression")
    choice = int(input("Pick up the number of experiment "))
    config_path  = input("Provide config file path:   ")
    if __name__=="__main__":
        parser = argparse.ArgumentParser(description='Training configuration:')
        parser.add_argument('-c', '--config', default=config_path, type=str,
                            help='config file (default: ./<script_filename>.json)')

        if choice==1:
            ext_args = parser.parse_args()
            config = json.load(open(ext_args.config))
            linear_regression.main(config)
        elif choice==2:
            ext_args = parser.parse_args()
            config = json.load(open(ext_args.config))
            log_regression.main(config)

    ending = input("\n\nDo you want to run new experiment? Please answer y/n ")
    if ending=='n':
        os._exit(1)


