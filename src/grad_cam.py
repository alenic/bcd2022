import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import matplotlib.cm as cm
import cv2


class GradCAM(object):
    def __init__(self, model, candidate_layers=None, relu=True, in_chans=3, device="cuda"):
        self.device = device
        self.model = model
        # a set of hook function handlers
        self.handlers = []

        self.fmap_pool = {}
        self.grad_pool = {}
        self.candidate_layers = candidate_layers  # list
        self.relu = relu
        self.in_chans = in_chans

        def save_fmaps(key):
            def forward_hook(module, input, output):
                if isinstance(module, torch.nn.LSTM):
                    self.fmap_pool[key] = output[0].detach()
                else:
                    self.fmap_pool[key] = output.detach()

            return forward_hook

        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook

        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    def remove_hook(self):
        for handle in self.handlers:
            handle.remove()

    def forward(self, image):
        self.image_shape = image.shape[2:]
        self.logits = self.model(image)
        self.probs = F.sigmoid(self.logits).squeeze(-1)
        return self.probs

    def backward(self, y_true):
        """
        Class-specific backpropagation
        """
        #one_hot = self._encode_one_hot(ids)
        self.model.zero_grad()
        if y_true:
            self.logits.backward(retain_graph=True)
        else:
            -self.logits.backward(retain_graph=True)

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            for n, p in self.model.named_modules():
                print(n)
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = F.adaptive_avg_pool2d(grads, 1)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        if self.relu:
            gcam = F.relu(gcam)
        gcam = F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )

        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        min_map = gcam.min(dim=1, keepdim=True)[0]
        max_map = gcam.max(dim=1, keepdim=True)[0]
        gcam -= min_map
        gcam /= max_map - min_map + 1e-12
        gcam = gcam.view(B, C, H, W)

        return gcam

    def image_heatmap_mix(self, region, raw_image, paper_cmap=False):
        cmap = (cm.hot(region)[..., :3] * 255.0).astype(np.uint8)
        cmap = cv2.resize(cmap, (raw_image.shape[1], raw_image.shape[0]))
        if paper_cmap:
            alpha = region[..., None]
            region = alpha * cmap + (1 - alpha) * raw_image
        else:
            mixed = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2

        return np.uint8(mixed)
