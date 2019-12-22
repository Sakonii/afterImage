import argparse

from segmentation import Segmentation, acc_camvid
from GUI import ImageObjects


def main():

    parser = argparse.ArgumentParser(description="Image File Name")
    parser.add_argument(
        "--image", type=str, default="img.png", help="Enter the image file name"
    )

    args = parser.parse_args()

    img, mask, classes = Segmentation(fname_img=args.image).start_segmentation()
    ImageObjects(img, mask, classes).inference()


if __name__ == "__main__":
    main()
