
import argparse
from segmentation import *


def main():

    parser = argparse.ArgumentParser(description = 'Image File Name')
    parser.add_argument('--image', type = str, default = 'img.png', 
                        help='Enter the image file name')                   

    args = parser.parse_args()

    Segmentation(fname_img = args.image).start_segmentation()


if __name__ == "__main__":
    main()