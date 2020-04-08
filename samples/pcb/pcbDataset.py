import os
import sys
import json
import glob
import datetime
import numpy as np
import skimage.draw
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


class pcbConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "pcb"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + IC

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class pcbDataset(utils.Dataset):
    def load_pcb(self, dataset_dir, subset, file_list=None, image_ext="jpg", object_class=None):
        """Load the pcb metal dataset
        dataset_dir: The root directory of the dataset (should include images and annotations dir with
                        annotations dir having subfolders for each class)
        subset: What to load (train, val, test)
        file_list: a text file containing a list of image names to consider
        image_ext: extension of the image default jpg
        object_class: list of object classes if not specified the annotation folder names will be considered
        """

        self.image_dir = os.path.join(dataset_dir, "images")
        self.annotations_dir = os.path.join(dataset_dir, "annotations")
        
        # add classes. 
        if object_class is None:
            obj = glob.glob(self.annotations_dir)
        
        print(obj)
        self.classes = []
        for i, obj_class in enumerate(obj):
            obj_class = os.path.basename(obj_class)
            if obj_class != "." or obj_class != "..":
                self.add_class("pcb", i, obj_class)
                self.classes.append(obj_class)

        if file_list in None:
            # read all the images from the image directory
            self.image_list = glob.glob(os.path.join(self.image_dir, "*.%s"%image_ext))
        else:
            # read all the images from the file_list
            with open(file_list, 'r') as f:
                self.image_list = [os.path.join(self.image_dir, m.strip('\n')) for m in f]

        self.file_list = [os.path.basename(mm)[:-4] for mm in self.image_list]
        self.annotation_list = [os.path.basename(im)[:-3]+"txt" for im in self.image_list]
        
        # add images to the dataset
        for i, name in enumerate(self.file_list):
            image = Image.open(os.path.join(self.image_dir, "%s.%s"%(name, image_ext)))
            width, height = image.size

            class_ids = []
            polygons = []
            for j, ann in enumerate(self.classes):
                fldr = os.path.join(self.annotations_dir, ann, "%s.txt"%(name))
                if not os.path.exists(fldr):
                    continue
                with open(fldr, "r") as f:
                    content = f.readlines()
                
                bboxes = [x.strip().split('\t')[1:] for x in content]
                class_id = np.ones(len(bboxes), dtype=np.int32) * (j+1)
                class_ids.append(class_id)
                
                for p in bboxes:
                    pp = {}
                    p = [int(float(s)) for s in p]
                    pp["name"] = "polygon"
                    x = [p[0], p[0]+p[2], p[0]+p[2], p[0]]
                    y = [p[1], p[1], p[1]+p[3], p[1]+p[3]]
                    pp["all_points_x"] = x
                    pp["all_points_y"] = y
                    polygons.append(pp)

        self.add_image("pcb",
                       image_id="%s.%s"%(name, image_ext),
                       path=self.image_dir,
                       width=width,
                       height=height,
                       polygons=polygons,
                       class_ids=class_ids)        
        

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        image_info = self.image_info[image_id]
        if image_info["source"] != "pcb":
            return super(self.__class__, self).load_mask(image_id)

        # convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        
        for i, p in enumerate(info["polygons"]):
            # get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # return mask, and array of class IDs of each instance.
        # since we have one class ID only, we return an array of 1s
        return mask.astype(np.bool), info["class_ids"]

    def image_reference(self, image_id):
        """ Return the path of the image """
        info = self.image_info[image_id]
        if info["source"] == "pcb":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

