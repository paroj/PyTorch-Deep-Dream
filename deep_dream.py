import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import numpy as np
import argparse
import os
import tqdm
from utils import deprocess, preprocess, clip

import cv2
import time


def dream(image, model, iterations, lr):
    """ Updates the image to maximize outputs for n iterations """
    image = Variable(image, requires_grad=True)
    for i in range(iterations):
        model.zero_grad()
        out = model(image)
        loss = out.norm()
        loss.backward()
        avg_grad = float(image.grad.data.abs().mean())
        norm_lr = lr / avg_grad
        image.data += norm_lr * image.grad.data
        image.data = clip(image.data)
        image.grad.data.zero_()
    return image


def deep_dream(image, model, iterations, lr, octave_scale, num_octaves):
    """ Main deep dream method """
    image = preprocess(image).unsqueeze(0).cuda()

    # Extract image representations for each octave
    octaves = [image]
    for _ in range(num_octaves - 1):
        image = nn.functional.interpolate(image, scale_factor=1/octave_scale, mode='bilinear', align_corners=False)
        octaves.append(image)

    detail = torch.zeros(octaves[-1].size(), dtype=torch.float).cuda()
    for octave, octave_base in enumerate(tqdm.tqdm(octaves[::-1], desc="Dreaming")):
        if octave > 0:
            # Upsample detail to new octave dimension
            detail = nn.functional.interpolate(detail, size=octave_base.size()[-2:], mode='bilinear', align_corners=False)

        # Add deep dream detail from previous octave to new base
        input_image = octave_base + detail
        # Get new deep dream image
        dreamed_image = dream(input_image.cuda(), model, iterations, lr)
        # Extract deep dream details
        detail = dreamed_image - octave_base

    return deprocess(dreamed_image.cpu().data.numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, default="images/supermarket.jpg", help="path to input image")
    parser.add_argument("--iterations", default=20, type=int, help="number of gradient ascent steps per octave")
    parser.add_argument("--at_layer", default=27, type=int, help="layer at which we modify image to maximize outputs")
    parser.add_argument("--lr", default=0.01, help="learning rate")
    parser.add_argument("--octave_scale", default=1.4, type=float, help="image scale between octaves")
    parser.add_argument("--num_octaves", default=10, type=int, help="number of octaves")
    args = parser.parse_args()

    # Load image
    image = cv2.cvtColor(cv2.imread(args.input_image), cv2.COLOR_BGR2RGB)

    # Define the model
    network = models.vgg19(pretrained=True)
    layers = list(network.features.children())
    model = nn.Sequential(*layers[: (args.at_layer + 1)])
    if torch.cuda.is_available:
        model = model.cuda()

    t = time.time()
    # Extract deep dream image
    dreamed_image = deep_dream(
        image,
        model,
        iterations=args.iterations,
        lr=args.lr,
        octave_scale=args.octave_scale,
        num_octaves=args.num_octaves,
    )

    # Save and plot image
    os.makedirs("outputs", exist_ok=True)
    filename = args.input_image.split("/")[-1]

    print(time.time() - t, "s")

    uchar_img = cv2.convertScaleAbs(dreamed_image, alpha=255)
    cv2.imshow("img", cv2.cvtColor(uchar_img, cv2.COLOR_RGB2BGR))
    cv2.waitKey()
