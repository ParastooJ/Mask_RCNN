nvidia-docker run --rm --name segmenter \
    -v /allen:/allen \
    -v /home/gayathrim/Dropbox/Mask_RCNN:/root/Mask_RCNN \
    -p 8887:8887 \
    -it maskrcnn /bin/bash