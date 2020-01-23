import argparse

from segmentation import Segmentation, acc_camvid
from inference import ImageObjects
from inpainting import FeatureLoss


def main():

    img, mask, classes = Segmentation(
        fname_img=args.image, model=args.model_segmentation
    ).start_segmentation()
    ImageObjects(img, mask, classes, model=args.model_inpainting).inference()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image File Name")
    parser.add_argument(
        "--image", type=str, default="img.png", help="Enter the image file name"
    )
    parser.add_argument(
        "--model_segmentation",
        type=str,
        default="1_segmentation_model.pkl",
        help="Pre-trained Weights for Segmentation",
    )
    parser.add_argument(
        "--model_inpainting",
        type=str,
        default="2_inpainting_perceptual_text.pkl",
        help="Pre-trained Weights for Inpainting",
    )
    args = parser.parse_args()

    main()
