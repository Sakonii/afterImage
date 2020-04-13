import argparse
import torch
from . import opt
import numpy as np

from torchvision import transforms
from cv2 import cv2
from PIL import Image
from .net import PConvUNet
from .util.io import load_ckpt
from .util.image import unnormalize


class Inpainting:
    "A Wrapper for Inpainting related objects"

    def __init__(self, modelPath="./models/1000000.pth", size=256):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.modelPath = modelPath
        self.size = (size, size)
        # Load arch and weights
        self.model = PConvUNet().to(self.device)
        load_ckpt(self.modelPath, [("model", self.model)])
        # Transforms that match the model
        self.img_transform = transforms.Compose(
            [
                transforms.Resize(size=self.size),
                transforms.ToTensor(),
                transforms.Normalize(mean=opt.MEAN, std=opt.STD),
            ]
        )
        self.mask_transform = transforms.Compose(
            [transforms.Resize(size=self.size), transforms.ToTensor()]
        )

    def tensor_to_numpy(self, tensor):
        "Convert Normalised Tensor to De-Normalized Numpy Image"
        return (
            unnormalize(tensor.cpu()).squeeze().permute(1, 2, 0).numpy()
        )

    def predict(self, img, mask):
        "Predictor function"
        self.model.eval()
        with torch.no_grad():
            prediction, _ = self.model(img.to(self.device), mask.to(self.device))
        prediction = prediction.to(torch.device("cpu"))
        prediction_comp = mask * img + (1 - mask) * prediction
        # To OpenCV Compatible Numpy Image
        img = self.tensor_to_numpy(img)
        prediction = self.tensor_to_numpy(prediction)
        prediction_comp = self.tensor_to_numpy(prediction_comp)
        return prediction, prediction_comp, img

    def to_original(self):
            
        "Move the selected object"
        roiSrc = self.origImg
        roiMask = self.origMask

        roiDst = self.origImg

        maskInverted = cv2.bitwise_not(roiMask)
        imgBg = cv2.bitwise_and(roiSrc, roiSrc, mask=roiMask)
        imgFg = cv2.bitwise_and(roiDst, roiDst, mask=maskInverted)
        dst = cv2.add(imgBg, imgFg)
        return dst

    def start_inpainting(self, img, mask):
        "Returns inpainted image based on the mask"
        # Copy Of Original
        self.origImg = np.copy(img)
        self.origMask = np.copy(mask)
        # Input Image Conversions
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # To PIL Image
        img = self.img_transform(img)
        # Mask Conversions
        mask = Image.fromarray(cv2.bitwise_not(mask))  # Inverse The Mask for Model
        mask = self.mask_transform(mask)
        mask[mask != 0] = 1
        # Image / Mask to batch of size 1
        img = img.view(1, 3, self.size[0], self.size[1])
        mask = mask.view(1, 3, self.size[0], self.size[1])
        # Predict
        prediction, prediction_comp, img = self.predict(img, mask)
        # Revert to Original Size
        prediction = cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR)
        prediction = cv2.resize(
            prediction,
            dsize=(self.origImg.shape[1], self.origImg.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        # 
        prediction = cv2.normalize(
            prediction, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        cv2.copyTo(prediction, self.origMask, self.origImg)
        return self.origImg


def main():
    img = cv2.imread(args.image)
    mask = cv2.imread(args.mask)
    prediction = Inpainting(args.model, size=256).start_inpainting(img, mask)
    cv2.imshow("Inpainting", prediction)
    cv2.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image / Mask / Weights")
    parser.add_argument(
        "--image",
        type=str,
        default="input/img.png",
        help="Name of image file name located at ./",
    )
    parser.add_argument(
        "--mask",
        type=str,
        default="input/mask.png",
        help="Name of mask file name located at ./",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./models/1000000.pth",
        help="Name of mask file name located at ./",
    )
    args = parser.parse_args()
    main()
