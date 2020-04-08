import os
import sys
#import skimage
import matplotlib.pyplot as plt 

import warnings
warnings.filterwarnings("ignore")

ROOT_DIR = os.path.abspath("../../")

sys.path.append(ROOT_DIR)

from mrcnn import models as modellib, utils, visualize
from vesicles import VesicleDataset, VesiclesConfig

COCO_WEIGTHS_PATH = os.path.join(ROOT_DIR, "mask")

     
def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = VesicleDataset()
    dataset_train.load_vesicle(args.train_dataset)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = VesicleDataset()
    dataset_val.load_vesicle(args.val_dataset)
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')

    # Training - stage 2
    # fine tune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val, 
                learning_rate=LEARNING_RATE,
                epochs=120,
                layers='4+')
    
    # Training stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=LEARNING_RATE / 10,
                epochs=160,
                layers='all')

'''
def test(model, image_path):
    print("Running on {}".format(image_path))
    
    # Read image
    image = skimage.io.imread(image_path)

    # detect objects
    r = model.detect([image], verbose=1)[0]

    # Display results
    ax = get_ax(1)
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                ["Background", "IC"], r['scores'], ax=ax,
                                title="Predictions")

'''
############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect nuclei.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'test'")
    parser.add_argument('--train_dataset', required=False,
                        metavar="/path/to/vesicle/dataset/",
                        help='Directory of the vesicle train dataset')
    parser.add_argument('--val_dataset', required=False,
                        metavar="/path/to/vesicle/dataset/",
                        help='Directory of the vesicle val dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the detector on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.train_dataset, "Argument --dataset is required for training"
    elif args.command == "test":
        assert args.image,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = VesicleConfig()
    else:
        class InferenceConfig(VesicleConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "test":
        test(model, image_path=args.image)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'test'".format(args.command))