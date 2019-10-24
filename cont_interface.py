import os
import argparse
import json
from experiments import linear_regression, log_regression

choice = os.environ["EXP"]
plots = os.environ["PLOT"]
config_default_dir = "experiments/"
config_file = os.environ["CONFIG"]
config_path = "".join([config_default_dir,config_file])
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Training configuration:')
    parser.add_argument('-c', '--config', default=config_path, type=str,
                        help='config file (default: ./<script_filename>.json)')
    parser.add_argument('-p', '--noplot', default=plots, type=bool,
                        help='Run Training without pyplots')
    if choice=='linear':
        ext_args = parser.parse_args()
        config = json.load(open(ext_args.config))
        linear_regression.main(config, ext_args)
    elif choice=='logistic':
        ext_args = parser.parse_args()
        config = json.load(open(ext_args.config))
        log_regression.main(config,ext_args)
