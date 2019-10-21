import argparse

from segmentation import Segmentation, acc_camvid
from GUI import ImageObjects


def main():

    parser = argparse.ArgumentParser(description="Image File Name")
    parser.add_argument(
        "--image", type=str, default="img.png", help="Enter the image file name"
    )

    args = parser.parse_args()

    img, mask, num_classes = Segmentation(fname_img=args.image).start_segmentation()

    img_obj = ImageObjects(img, mask, num_classes)
    img_obj.detect_objects()
    img_obj.show()


if __name__ == "__main__":
    main()