import numpy as np
from cv2 import cv2

import torch.nn as nn
import torch.nn.functional as F

from fastai.vision import Image, open_image, image2np, pil2tensor
from fastai.callbacks.hooks import load_learner, hook_outputs
from fastai.utils.mem import gc


class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.base_loss = F.l1_loss
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = (
            ["pixel",]
            + [f"feat_{i}" for i in range(len(layer_ids))]
            + [f"gram_{i}" for i in range(len(layer_ids))]
        )

    def gram_matrix(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, -1)
        return (x @ x.transpose(1, 2)) / (c * h * w)

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]

    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [self.base_loss(input, target)]
        self.feat_losses += [
            self.base_loss(f_in, f_out) * w
            for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)
        ]
        self.feat_losses += [
            self.base_loss(self.gram_matrix(f_in), self.gram_matrix(f_out)) * w ** 2 * 5e3
            for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)
        ]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)

    def __del__(self):
        self.hooks.remove()


class Inpainting:
    "A Wrapper for Inpainting related objects"

    def __init__(self, img, model, path_to_learner="./models"):

        self.path_to_learner = path_to_learner
        self.learn = load_learner(path=self.path_to_learner, file=model)

        "Convert np.ndarray to torch style Image"
        self._img = np.copy(img)
        self.img = Image(pil2tensor(img, np.float32).div_(255))

    def start_inpainting(self):
        "Returns inpainted image"

        self.pred = self.learn.predict(self.img)
        self.img = self.pred[0]

        "Convert from torch style Image to OpenCV compatible np.ndarray"
        self.img = image2np(self.img.data * 255).astype(np.uint8)
        cv2.cvtColor(src=self.img, dst=self.img, code=cv2.COLOR_BGR2RGB)

        self.img = cv2.resize(
            self.img,
            dsize=(self._img.shape[1], self._img.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        return self.img
