from ctypes import *
import random
import os
import cv2
import time
import darknet
import argparse
import sys
from threading import Thread, enumerate, Event
from queue import Queue

class_name_to_num = {'person':0,'bicycle':1,'car':2,'motorbike':3,'aeroplane':4,'bus':5,'train':6,'truck':7,'boat':8,'traffic light':9,'fire hydrant':10,
                    'stop sign':11,'parking meter':12,'bench':13,'bird':14,'cat':15,'dog':16,'horse':17,'sheep':18,'cow':19,'elephant':20,
                    'bear':21,'zebra':22,'giraffe':23,'backpack':24,'umbrella':25,'handbag':26,'tie':27,'suitcase':28,'frisbee':29,'skis':30,
                    'snowboard':31,'sports ball':32,'kite':33,'baseball bat':34,'baseball glove':35,'skateboard':36,'surfboard':37,'tennis racket':38,'bottle':39,'wine glass':40,
                    'cup':41,'fork':42,'knife':43,'spoon':44,'bowl':45,'banana':46,'apple':47,'sandwich':48,'orange':49,'broccoli':50,
                    'carrot':51,'hot dog':52,'pizza':53,'donut':54,'cake':55,'chair':56,'sofa':57,'pottedplant':58,'bed':59,'diningtable':60,
                    'toilet':61,'tvmonitor':62,'laptop':63,'mouse':64,'remote':65,'keyboard':66,'cell phone':67,'microwave':68,'oven':69,'toaster':70,
                    'sink':71,'refrigerator':72,'book':73,'clock':74,'vase':75,'scissors':76,'teddy bear':77,'hair drier':78,'toothbrush':79}

    
def parser():
    parser = argparse.ArgumentParser(description = "darknet YOLO 영상 객체 탐지")
    parser.add_argument("--input", "-i", type=str, default=0,
                        help="객체 탐지를 하려고하는 영상 불러오기, 아무것도 입력하지 않으면 웹캠 실시간 영상을 불러오기")
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
    parser.add_argument("--labels_path", "-l", type=str, default="data/autolabeling",
                        help="각 프레임마다 객체 탐지된 결과 출력 경로")
    return parser.parse_args()

def str2int(video_path):
    try:
        return int(video_path)
    except ValueError:
        return video_path
    
def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "역치는 반드시 0과 1사이의 실수 값이여야 합니다."
    if not os.path.exists(args.config_file):
        raise(ValueError("해당 경로에서 config를 찾지 못했습니다. {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("해당 경로에서 weights를 찾지 못했습니다. {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("해당 경로에서 data를 찾지 못했습니다. {}".format(os.path.abspath(args.data_file))))
    if str2int(args.input) == str and not os.path.exists(args.input):
        raise(ValueError("해당 경로에서 video를 찾지 못했습니다. {}".format(os.path.abspath(args.input))))
    
frame_queue = Queue()
darknet_image_queue = Queue(maxsize=1)
detections_queue = Queue(maxsize=1)
fps_queue = Queue(maxsize=1)

args = parser()
check_arguments_errors(args)
network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=1
    )
darknet_width = darknet.network_width(network)
darknet_height = darknet.network_height(network)
input_path = str2int(args.input)
cap = cv2.VideoCapture(input_path)
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
   
def set_saved_video(input_video, output_video, size):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video

def convert2relative(bbox):
    x, y, w, h  = bbox
    _height     = darknet_height
    _width      = darknet_width
    return x/_width, y/_height, w/_width, h/_height

def convert2original(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_x       = int(x * image_w)
    orig_y       = int(y * image_h)
    orig_width   = int(w * image_w)
    orig_height  = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted

def convert4cropping(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_left    = int((x - w / 2.) * image_w)
    orig_right   = int((x + w / 2.) * image_w)
    orig_top     = int((y - h / 2.) * image_h)
    orig_bottom  = int((y + h / 2.) * image_h)

    if (orig_left < 0): orig_left = 0
    if (orig_right > image_w - 1): orig_right = image_w - 1
    if (orig_top < 0): orig_top = 0
    if (orig_bottom > image_h - 1): orig_bottom = image_h - 1

    bbox_cropping = (orig_left, orig_top, orig_right, orig_bottom)

    return bbox_cropping

def video_capture(frame_queue, darknet_image_queue):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height),
                                   interpolation=cv2.INTER_LINEAR)
        frame_queue.put(frame)
        img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
        darknet_image_queue.put(img_for_detect)
    cap.release()

def inference(darknet_image_queue, detections_queue, fps_queue):
    while cap.isOpened():
        darknet_image = darknet_image_queue.get()
        prev_time = time.time()
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=args.thresh)
        detections_queue.put(detections)
        fps = int(1/(time.time() - prev_time))
        fps_queue.put(fps)
        print("FPS: {}".format(fps))
        darknet.print_detections(detections, args.ext_output)
        darknet.free_image(darknet_image)
    cap.release()

def createFolder(dir):
    path2labels = args.labels_path + "/" + dir.split('/')[-1].split('.')[0]
    try:
        if not os.path.exists(path2labels):
            os.makedirs(path2labels)
    except OSError:
        print('Error: 해당 경로에 폴더를 만들 수 없음 ' + path2labels)
    return path2labels
        
def createTxt(frame_num, detections_yoloform, dir):
    with open(dir + "/" + str(frame_num) + '.txt', 'w', encoding='UTF-8') as f:
        for annotations in detections_yoloform:
            annotations = ' '.join(map(str, annotations))
            f.write(annotations + '\n')

def createClasses(dir):
    with open(dir + '/classes.txt', 'w', encoding='UTF-8') as f:
        with open(args.data_file.split('.')[0] + '.names', 'r', encoding='UTF-8') as name_file:
            x = name_file.read()
            f.write(x)       

def drawing(frame_queue, detections_queue, fps_queue):
    random.seed(3)  # deterministic bbox colors
    video = set_saved_video(cap, args.out_filename, (video_width, video_height))
    path2labels = createFolder(args.out_filename) #저장할 폴더 생성
    frame_num = 1 #파일명을 위한 frame 번호 지정
    createClasses(path2labels)
    while cap.isOpened():
        frame = frame_queue.get()
        detections = detections_queue.get()
        fps = fps_queue.get()
        detections_adjusted = []
        detections_yoloform = []
        cv2.imwrite(path2labels +'/' + str(frame_num) + ".jpg", frame) #frame 이미지 저장
        if frame is not None:
            for label, confidence, bbox in detections:
                bbox_adjusted = convert2original(frame, bbox)
                bbox_relatived = convert2relative(bbox) #bbox 좌표 상대값으로 변경
                label_num = class_names.index(label) #label class를 번호로 변경
                detections_adjusted.append((str(label), confidence, bbox_adjusted))
                bbox_num_relatived = list(bbox_relatived)
                bbox_num_relatived.insert(0, label_num) #label class 번호를 앞으로 삽입
                detections_yoloform.append(bbox_num_relatived)
                createTxt(frame_num, detections_yoloform, path2labels) #label class 번호와 bbox 좌표값 txt로 저장
            image = darknet.draw_boxes(detections_adjusted, frame, class_colors)
            if not args.dont_show:
                cv2.imshow('Auto Labeling', image)
            if args.out_filename is not None:
                video.write(image)
            if cv2.waitKey(fps) == 27:
                break
            frame_num += 1
    cap.release()
    video.release()
    cv2.destroyAllWindows()               

def main(input, output, label_path):
    if input is not None:
        args.input = input
    if output is not None:
        args.out_filename = output
    if label_path is not None:
        args.labels_path = label_path
    
    t1 = Thread(target=video_capture, args=(frame_queue, darknet_image_queue))
    t2 = Thread(target=inference, args=(darknet_image_queue, detections_queue, fps_queue))
    t3 = Thread(target=drawing, args=(frame_queue, detections_queue, fps_queue))
    
    t1.start()
    t2.start()
    t3.start()
    # try:
    #     while True:
    #         time.sleep(1)
    # except (KeyboardInterrupt, SystemExit):
    t1.join()
    t2.join()
    t3.join()
    
    print("작업 완료")

if __name__ == '__main__':
    main(None, None, None)

