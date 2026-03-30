import cv2
import numpy as np


def preprocess_building(img):
    img = img / 255.0
    return img.astype(np.float32)


def preprocess_road(img):
    img = img.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    img = (img - mean) / std
    return img


def preprocess_water(img):
    img = img.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    img = (img - mean) / std

    return img