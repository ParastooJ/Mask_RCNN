import os
import sys
import json
import datetime
import numpy as np 
import skimage.io 
import glob
from imgaug import augmenters as iaa 
import cv2


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize


# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/nucleus/")

class NucleiConfig(Config):
    NAME = "nuclei"
    
    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 6

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + IC

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


class NucleiInferenceConfig(NucleiConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    
    IMAGES_PER_GPU = 1
    
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "none"
    
    IMAGE_MIN_DIM = 256
    
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7


class NucleiDataset(utils.Dataset):

    def load_nucleus(self, dataset_dir, subset):
        self.add_class("nuclei", 1, "nuclei")
        self.dataset_dir = dataset_dir

        assert subset in ["train", "val"]
        
        image_dir = os.path.join(dataset_dir, subset, "images")
        label_dir = os.path.join(dataset_dir, subset, "labels")
        image_list = glob.glob(os.path.join(image_dir, "*.png"))
        label_list = glob.glob(os.path.join(label_dir, "*.png"))

        self.image_dir = image_dir
        self.label_dir = label_dir 
        self.image_list = image_list
        self.label_list = label_list

        image_ids = [os.path.basename(i) for i in image_list]

        # Add images
        for image_id in image_ids:
            self.add_image("nuclei",
                           image_id=image_id,
                           path=os.path.join(image_dir, image_id))

    def load_mask(self, image_id):
        info = self.image_info[image_id]

        # Get mask directory from image path
        mask_dir = self.label_dir

        # Read mask files from .png image
        #maskimg = skimage.io.imread(os.path.join(mask_dir, info['id'])).astype(np.bool)
        maskimg = cv2.imread(os.path.join(mask_dir, info['id']), 0)

        cnts, hierarchy= cv2.findContours(maskimg.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros([maskimg.shape[0], maskimg.shape[1], len(cnts)],
                        dtype=np.uint8)
        for i, c in enumerate(cnts):
            ptx = c[:,0,0]
            pty = c[:,0,1]
            rr, cc = skimage.draw.polygon(pty, ptx)
            mask[rr,cc,i] = 1
            

        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "nuclei":
            print(info)
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
         