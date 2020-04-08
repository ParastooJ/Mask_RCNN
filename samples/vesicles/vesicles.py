"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import glob
from PIL import Image

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class VesicleConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "vesicles"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + vesicles

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 50

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9



class VesicleInferenceConfig(VesicleConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 2
    
    IMAGES_PER_GPU = 2
    
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "none"
    
    IMAGE_MIN_DIM = 256
    
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7
############################################################
#  Dataset
############################################################

class VesicleDataset(utils.Dataset):

    def load_vesicle(self, dataset_dir, object_class="vesicle"):

        # add classes
        self.add_class("vesicle", 1, object_class)

        image_ids = glob.glob(os.path.join(dataset_dir, "*"))
        
        for id in image_ids:
            id = os.path.basename(id)
            img_path = os.path.join(dataset_dir, id, "images", "{}.png".format(id))
            mask_path = os.path.join(dataset_dir, id, "masks")
            image = Image.open(img_path)
            width, height = image.size

            self.add_image("vesicle", 
                            image_id=id, 
                            path=img_path, 
                            mask_path=mask_path, 
                            width=width, 
                            height=height)

    def assemble_masks(self, path):
        mask = None
        for i, mask_file in enumerate(next(os.walk(path))[2]):
            mask_ = Image.open(os.path.join(path, mask_file)).convert("RGB")
            mask_ = np.asarray(mask_)
            if i == 0:
                mask = mask_
                continue
            mask = mask | mask_
        return mask

    def gather_masks(self, path, info):
        files = glob.glob(os.path.join(path, "*.png"))
        mask = np.zeros([info['height'], info['width'], len(files)], dtype=np.uint8)
        for i, mask_file in enumerate(files):
            mask_ = Image.open(mask_file)
            mask_ = np.asarray(mask_)
            mask[:,:,i] = mask_
        return mask

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info['source'] != 'vesicle':
            return super(self._image_ids, self).load_mask(image_id)

        # read all the mask images and combine into one
        mask = self.gather_masks(image_info['mask_path'], image_info)

        # find the number of mask files
        masknum = len(glob.glob(os.path.join(image_info['mask_path'], "*.png")))

        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info['source'] == "vesicle":
            return info['path']
        else:
            super(self.__class__, self).image_reference(image_id)


