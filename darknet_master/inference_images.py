import argparse
import os
import glob
import random
import time
import cv2
import numpy as np
import darknet

def parser():
    parser = argparse.ArgumentParser(description="YOLO 이미지 객체 탐지")
    parser.add_argument("--input", "-i", type=str, default="",
                        help="객체 탐지를 하려고하는 이미지 또는 폴더 불러오기")
    parser.add_argument("--batch_size", "-bn", default=1, type=int,
                        help="number of images to be processed at the same time")
    parser.add_argument("--out_filename", "-o", type=str, default="",
                        help="탐지 결과 영상 다른 이름으로 저장하기")
    parser.add_argument("--weights", "-w", default="yolov4.weights",
                        help="yolo 미리 학습된 weights 파일 불러오기")
    parser.add_argument("--dont_show", action='store_true',
                        help="실행 창 띄우지 않기")
    parser.add_argument("--ext_output", action='store_true',
                        help="탐지된 바운딩 박스 좌표 표시하기")
    parser.add_argument("--config_file", "-c", default="cfg/yolov4.cfg",
                        help="cfg 파일 불러오기")
    parser.add_argument("--data_file", "-d", default="cfg/coco.data",
                        help="data 파일 불러오기")
    parser.add_argument("--thresh", "-th", type=float, default=.25,
                        help="특정 수치 밑의 인식된 박스를 지우는 역치 설정")
    parser.add_argument("--save_labels", "-s", action='store_true',
                    help="각 이미지의 탐지된 바운딩박스를 yolo 형식으로 저장")
    parser.add_argument("--crop_detections", "-cd", action='store_true',
                    help="탐지된 객체를 bbox 기반으로 crop하여 class별로 저장")
    parser.add_argument("--crop_path", "-cp", default='crop/detections',
                    help="crop한 이미지를 저장하는 위치 설정")
    
    return parser.parse_args()


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if args.input and not os.path.exists(args.input):
        raise(ValueError("Invalid image path {}".format(os.path.abspath(args.input))))


def check_batch_shape(images, batch_size):
    """
        Image sizes should be the same width and height
    """
    shapes = [image.shape for image in images]
    if len(set(shapes)) > 1:
        raise ValueError("Images don't have same shape")
    if len(shapes) > batch_size:
        raise ValueError("Batch size higher than number of images")
    return shapes[0]


def load_images(images_path):
    """
    If image path is given, return it directly
    For txt file, read it and return each line as image path
    In other case, it's a folder, return a list with names of each
    jpg, jpeg and png file
    """
    input_path_extension = images_path.split('.')[-1]
    if input_path_extension in ['jpg', 'jpeg', 'png']:
        return [images_path]
    elif input_path_extension == "txt":
        with open(images_path, "r") as f:
            return f.read().splitlines()
    else:
        return glob.glob(
            os.path.join(images_path, "*.jpg")) + \
            glob.glob(os.path.join(images_path, "*.png")) + \
            glob.glob(os.path.join(images_path, "*.jpeg"))


def prepare_batch(images, network, channels=3):
    width = darknet.network_width(network)
    height = darknet.network_height(network)

    darknet_images = []
    for image in images:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height),
                                   interpolation=cv2.INTER_LINEAR)
        custom_image = image_resized.transpose(2, 0, 1)
        darknet_images.append(custom_image)

    batch_array = np.concatenate(darknet_images, axis=0)
    batch_array = np.ascontiguousarray(batch_array.flat, dtype=np.float32)/255.0
    darknet_images = batch_array.ctypes.data_as(darknet.POINTER(darknet.c_float))
    return darknet.IMAGE(width, height, channels, darknet_images)


def get_iou(yolo_bbox1, yolo_bbox2):
    # input yolo_bbox = (cx, cy, w, h)
    cx, cy, w, h = yolo_bbox1
    bbox1 = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
    cx, cy, w, h = yolo_bbox2
    bbox2 = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    # compute the width and height of the intersection
    w = x2 - x1
    h = y2 - y1
    if w < 0 or h < 0:
        iou = 0
    else:
        inter = w * h
        iou = inter / (bbox1_area + bbox2_area - inter)
    return iou


def nms(detections, nms_thresh):
    m = 0
    while (m < len(detections) - 1):
        n = m + 1
        while (n < len(detections)):
            iou = get_iou(detections[m][2], detections[n][2])
            if iou > nms_thresh:
                del detections[n]
            else:
                n += 1
        m += 1
    return detections


def image_detection(image_or_path, network, class_names, class_colors, thresh, nms_thresh=.45):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    # if type(image_or_path) == "str":
    #     image = cv2.imread(image_or_path)
    # else:
    #     image = image_or_path
    image = cv2.imread(image_or_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    if nms_thresh:
        detections = nms(detections, nms_thresh)   
    darknet.free_image(darknet_image)    
    image_ = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image_, cv2.COLOR_BGR2RGB), detections

def batch_detection(network, images, class_names, class_colors,
                    thresh=0.25, hier_thresh=.5, nms=.45, batch_size=4):
    image_height, image_width, _ = check_batch_shape(images, batch_size)
    darknet_images = prepare_batch(images, network)
    batch_detections = darknet.network_predict_batch(network, darknet_images, batch_size, image_width,
                                                     image_height, thresh, hier_thresh, None, 0, 0)
    batch_predictions = []
    for idx in range(batch_size):
        num = batch_detections[idx].num
        detections = batch_detections[idx].dets
        if nms:
            darknet.do_nms_obj(detections, num, len(class_names), nms)
        predictions = darknet.remove_negatives(detections, class_names, num)
        images[idx] = darknet.draw_boxes(predictions, images[idx], class_colors)
        batch_predictions.append(predictions)
    darknet.free_batch_detections(batch_detections, batch_size)
    return images, batch_predictions


def image_classification(image, network, class_names):
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                                interpolation=cv2.INTER_LINEAR)
    darknet_image = darknet.make_image(width, height, 3)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.predict_image(network, darknet_image)
    predictions = [(name, detections[idx]) for idx, name in enumerate(class_names)]
    darknet.free_image(darknet_image)
    return sorted(predictions, key=lambda x: -x[1])


def convert2relative(image, bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    height, width, _ = image.shape
    return x/width, y/height, w/width, h/height


def save_annotations(name, image, detections, class_names):
    """
    Files saved with image_name.txt and relative coordinates
    """
    file_name = os.path.splitext(name)[0] + ".txt"
    with open(file_name, "w") as f:
        for label, confidence, bbox in detections:
            x, y, w, h = convert2relative(image, bbox)
            label = class_names.index(label)
            label_bbox_list = list("{} {:f} {:f} {:f} {:f}\n".format(label, x, y, w, h))
            for annotations in label_bbox_list:
                annotations = ' '.join(map(str, annotations))
                f.write(annotations)


def createFolder(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print('Error: 해당 경로에 폴더를 만들 수 없음 ' + dir)
    return dir


def original_image_(image, network):
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    original_image = cv2.imread(image)
    original_image_resized = cv2.resize(original_image, (width, height),interpolation=cv2.INTER_LINEAR)
    return original_image_resized


def crop_objects(name, image, detections, path, class_names):
    num_objects = len(detections)
    #create dictionary to hold count of objects for image name
    counts = dict()
    i = 0
    # print(name.split('.')[0].split('\\')[-1])
    for i in range(num_objects):
        # get count of class for part of image name
        class_name = detections[i][0]
        if class_name in class_names:
            counts[class_name] = counts.get(class_name, 0) + 1
            # get box coords
            x, y, w, h = detections[i][2]
            xmin = x - w/2
            ymin = y - h/2
            xmax = x + w/2
            ymax = y + h/2
            if xmin < 0:
                 xmin = 0
            if ymin < 0:
                ymin = 0
            if xmax > image.shape[1]:
                xmax = image.shape[1]
            if ymax > image.shape[0]:
                ymax = image.shape[0]

            # crop detection from image
            cropped_image = image[int(ymin):int(ymax), int(xmin):int(xmax)]
            # construct image name and join it to path for saving crop properly
            path2class = path + '/' + class_name
            createFolder(path2class)
            crop_image_name = class_name + '_' + name.split('.')[0].split('\\')[-1] + '_' + str(counts[class_name]) + '.jpg'
            crop_image_path = os.path.join(path2class, crop_image_name)
            # save image
            cv2.imwrite(crop_image_path, cropped_image)
        else:
            continue

def batch_detection_example():
    args = parser()
    check_arguments_errors(args)
    batch_size = 3
    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=batch_size
    )
    image_names = ['data/horses.jpg', 'data/horses.jpg', 'data/eagle.jpg']
    images = [cv2.imread(image) for image in image_names]
    images, detections,  = batch_detection(network, images, class_names,
                                           class_colors, batch_size=batch_size)
    for name, image in zip(image_names, images):
        cv2.imwrite(name.replace("data/", ""), image)
    print(detections)


def main():
    args = parser()
    check_arguments_errors(args)

    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=args.batch_size
    )

    images = load_images(args.input)

    index = 0
    while True:
        # loop asking for new image paths if no list is given
        if args.input:
            if index >= len(images):
                break
            image_name = images[index]
        else:
            image_name = input("Enter Image Path: ")
        prev_time = time.time()
        image, detections = image_detection(
            image_name, network, class_names, class_colors, args.thresh
            )
        if args.save_labels:
            save_annotations(image_name, image, detections, class_names)
        if args.crop_detections:
            original_image = original_image_(image_name, network)
            crop_objects(image_name, original_image, detections, args.crop_path, class_names)
        darknet.print_detections(detections, args.ext_output)
        fps = int(1/(time.time() - prev_time))
        print("FPS: {}".format(fps))
        if not args.dont_show:
            cv2.imshow('Inference', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        index += 1


if __name__ == "__main__":
    # unconmment next line for an example of batch processing
    # batch_detection_example()
    main()
