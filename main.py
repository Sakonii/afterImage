
import argparse
from segmentation import *


def main():

    parser = argparse.ArgumentParser(description = 'Image File Name')
    parser.add_argument('--image', type = str, default = 'img.png', 
                        help = 'Enter the image file name')                   

    args = parser.parse_args()

    Seg1 = Segmentation(fname_img = args.image)
    Seg1.start_segmentation()
    Seg1.show()

if __name__ == "__main__":
    main()