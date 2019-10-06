
import argparse
from inference import *


def main():

    parser = argparse.ArgumentParser(description = 'Mode Selection')
    parser.add_argument('--mode_select', type = str, default = 'inference', 
                        help='Mode Selection: train / inference')                   

    args = parser.parse_args()
    print(f'Selected Mode: {args.mode_select}')   

    if args.mode_select == 'inference':
        inference.start_inference()


if __name__ == "__main__":
    main()