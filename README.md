# Darknet Python inference

## Getting started

- make environment by anaconda

    `conda create -n ENV_NAME python=3.10`

    `conda activate ENV_NAME`

- install requirements

    `pip install -r requirements.txt`

- Setup Darknet

    [Windows and Ubuntu](https://techzizou.com/yolo-installation-on-windows-and-linux/)

##Command Lines
- Inference image or image folder

    `python inference_images.py -i INPUT_IMAGE_OR_PATH -w WEIGHTS_FILE_NAME.weights -c CFG_FILE_NAME.cfg -d DATA_FILE_NAME.data`
    
- Inference video

    `python inference_video.py -i INPUT_VIDEO -o OUTPUT_VIDEO_NAME.mp4 -w WEIGHTS_FILE_NAME.weights -c CFG_FILE_NAME.cfg -d DATA_FILE_NAME.data`

- Addtional commands
    1. `-s` or `--save_labels` is save detections by YOLO form
    2. if you don't want to see inference windows `--dont_show`
    3. `--ext_output` show detections result 
    3. `-cd` or `--crop_detections` is crop detections and save cropped images
    4. `-cp` or `--crop_path` is directory for saved cropped images

