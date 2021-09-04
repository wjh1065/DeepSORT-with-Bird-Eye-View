import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from cv2point import *

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')

flags.DEFINE_string('video', './data/video/1.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('video2', './data/video/22.mp4', 'path to input video or set to 0 for webcam')

flags.DEFINE_string('output', './outputs/demo.avi', 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    max_cosine_distance2 = 0.4

    nn_budget = None
    nn_budget2 = None

    nms_max_overlap = 1.0
    nms_max_overlap2 = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    model_filename2 = 'model_data/mars-small128.pb'

    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    encoder2 = gdet.create_box_encoder(model_filename2, batch_size=1)

    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    metric2 = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance2, nn_budget2)

    # initialize tracker
    tracker = Tracker(metric)
    tracker2 = Tracker(metric2)

    # load configuration for object detector
    config = ConfigProto()
    config2 = ConfigProto()

    config.gpu_options.allow_growth = True
    config2.gpu_options.allow_growth = True

    session = InteractiveSession(config=config)
    session2 = InteractiveSession(config=config2)

    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    STRIDES2, ANCHORS2, NUM_CLASS2, XYSCALE2 = utils.load_config(FLAGS)
    input_size2 = FLAGS.size
    video_path2 = FLAGS.video2

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter2 = tf.lite.Interpreter(model_path=FLAGS.weights)

        interpreter.allocate_tensors()
        interpreter2.allocate_tensors()

        input_details = interpreter.get_input_details()
        input_details2 = interpreter2.get_input_details()

        output_details = interpreter.get_output_details()
        output_details2 = interpreter2.get_output_details()
        print(input_details)
        print(output_details)

        print(input_details2)
        print(output_details2)

    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        saved_model_loaded2 = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])

        infer = saved_model_loaded.signatures['serving_default']
        infer2 = saved_model_loaded2.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
        vid2 = cv2.VideoCapture(int(video_path2))
    except:
        vid = cv2.VideoCapture(video_path)
        vid2 = cv2.VideoCapture(video_path2)

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        width2 = int(vid2.get(cv2.CAP_PROP_FRAME_WIDTH))

        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        height2 = int(vid2.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fps = int(vid.get(cv2.CAP_PROP_FPS))
        fps2 = int(vid2.get(cv2.CAP_PROP_FPS))

        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        codec2 = cv2.VideoWriter_fourcc(*FLAGS.output_format)

        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
        out2 = cv2.VideoWriter(FLAGS.output, codec2, fps2, (width2, height2))

    frame_num = 0
    frame_num2 = 0

    # while video is running
    while True:
        return_value, frame = vid.read()
        return_value, frame2 = vid2.read()

        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

            image = Image.fromarray(frame)
            image2 = Image.fromarray(frame2)

        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        frame_num2 += 1
        print('1. Frame #: ', frame_num)
        print('2. Frame #: ', frame_num2)

        frame_size = frame.shape[:2]
        frame_size2 = frame2.shape[:2]

        image_data = cv2.resize(frame, (input_size, input_size))
        image_data2 = cv2.resize(frame2, (input_size2, input_size2))

        image_data = image_data / 255.
        image_data2 = image_data2 / 255.

        image_data = image_data[np.newaxis, ...].astype(np.float32)
        image_data2 = image_data2[np.newaxis, ...].astype(np.float32)

        start_time = time.time()
        start_time2 = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter2.set_tensor(input_details2[0]['index'], image_data2)

            interpreter.invoke()
            interpreter2.invoke()

            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            pred2 = [interpreter2.get_tensor(output_details2[i]['index']) for i in range(len(output_details2))]

            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))

                boxes2, pred_conf2 = filter_boxes(pred2[1], pred2[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size2, input_size2]))

            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))

                boxes2, pred_conf2 = filter_boxes(pred2[0], pred2[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size2, input_size2]))

        else:
            batch_data = tf.constant(image_data)
            batch_data2 = tf.constant(image_data2)

            pred_bbox = infer(batch_data)
            pred_bbox2 = infer(batch_data2)

            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

            for key, value in pred_bbox2.items():
                boxes2 = value[:, :, 0:4]
                pred_conf2 = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        boxes2, scores2, classes2, valid_detections2 = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes2, (tf.shape(boxes2)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf2, (tf.shape(pred_conf2)[0], -1, tf.shape(pred_conf2)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        num_objects2 = valid_detections2.numpy()[0]

        bboxes = boxes.numpy()[0]
        bboxes2 = boxes2.numpy()[0]

        bboxes = bboxes[0:int(num_objects)]
        bboxes2 = bboxes2[0:int(num_objects2)]

        scores = scores.numpy()[0]
        scores2 = scores2.numpy()[0]

        scores = scores[0:int(num_objects)]
        scores2 = scores2[0:int(num_objects2)]

        classes = classes.numpy()[0]
        classes2 = classes2.numpy()[0]

        classes = classes[0:int(num_objects)]
        classes2 = classes2[0:int(num_objects2)]


        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        original_h2, original_w2, _ = frame2.shape

        bboxes = utils.format_boxes(bboxes, original_h, original_w)
        bboxes2 = utils.format_boxes(bboxes2, original_h2, original_w2)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]
        pred_bbox2 = [bboxes2, scores2, classes2, num_objects2]


        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)
        class_names2 = utils.read_class_names(cfg.YOLO.CLASSES)


        # by default allow all classes in .names file
        # allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['person']
        allowed_classes2 = ['person']


        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        names2 = []

        deleted_indx = []
        deleted_indx2 = []

        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)

        for j in range(num_objects2):
            class_indx2 = int(classes2[j])
            class_name2 = class_names2[class_indx2]
            if class_name2 not in allowed_classes2:
                deleted_indx2.append(j)
            else:
                names2.append(class_name2)

        names = np.array(names)
        count = len(names)

        names2 = np.array(names2)
        count2 = len(names2)

        if FLAGS.count:
            cv2.putText(frame, "1. Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("1. Objects being tracked: {}".format(count))

            cv2.putText(frame2, "2. Objects being tracked: {}".format(count2), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("2. Objects being tracked: {}".format(count2))

        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        bboxes2 = np.delete(bboxes2, deleted_indx2, axis=0)

        scores = np.delete(scores, deleted_indx, axis=0)
        scores2 = np.delete(scores2, deleted_indx2, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        features2 = encoder(frame2, bboxes2)

        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]
        detections2 = [Detection(bbox2, score2, class_name2, feature2) for bbox2, score2, class_name2, feature2 in zip(bboxes2, scores2, names2, features2)]


        #initialize color map
        cmap = plt.get_cmap('tab20b')
        cmap2 = plt.get_cmap('tab20b')

        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
        colors2 = [cmap2(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        boxs2 = np.array([d.tlwh for d in detections2])

        scores = np.array([d.confidence for d in detections])
        scores2 = np.array([d.confidence for d in detections2])

        classes = np.array([d.class_name for d in detections])
        classes2 = np.array([d.class_name for d in detections2])

        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        indices2 = preprocessing.non_max_suppression(boxs2, classes2, nms_max_overlap2, scores2)

        detections = [detections[i] for i in indices]
        detections2 = [detections2[i] for i in indices2]

        # Call the tracker
        tracker.predict()
        tracker2.predict()

        tracker.update(detections)
        tracker2.update(detections2)

        # update tracks
        boxes_lcs_1=[]
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            test = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            test = np.array(test, dtype="int")
            boxes_lcs_1.append(test)
            class_name = track.get_class()
        # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

        boxes_lcs_2 = []
        for track2 in tracker2.tracks:
            if not track2.is_confirmed() or track2.time_since_update > 1:
                continue
            bbox2 = track2.to_tlbr()
            test2 = (int(bbox2[0]), int(bbox2[1]), int(bbox2[2]), int(bbox2[3]))
            test2 = np.array(test2, dtype="int")
            boxes_lcs_2.append(test2)
            class_name2 = track2.get_class()
            # draw bbox on screen
            color2 = colors2[int(track2.track_id) % len(colors2)]
            color2 = [i * 255 for i in color2]
            cv2.rectangle(frame2, (int(bbox2[0]), int(bbox2[1])), (int(bbox2[2]), int(bbox2[3])), color2, 2)
            cv2.rectangle(frame2, (int(bbox2[0]), int(bbox2[1] - 30)),
                          (int(bbox2[0]) + (len(class_name2) + len(str(track2.track_id))) * 17, int(bbox2[1])), color2, -1)
            cv2.putText(frame2, class_name2 + "-" + str(track2.track_id), (int(bbox2[0]), int(bbox2[1] - 10)), 0, 0.75,
                        (255, 255, 255), 2)



        # if enable info flag then print details about each track
            if FLAGS.info:
                print("1. Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
                print("2. Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track2.track_id),
                                                                                                    class_name2, (
                                                                                                    int(bbox2[0]),
                                                                                                    int(bbox2[1]),
                                                                                                    int(bbox2[2]),
                                                                                                    int(bbox2[3]))))
        """
        LCS
        """
        W = 720
        H = 1280

        W2 = 1280
        H2 = 720

        # points = [(764,657), (1119,526), (296,244), (144,254)]
        # src = np.float32(np.array(points[:4]))
        # dst = np.float32([[0, H], [W, H],[W, 0],[0, 0]])

        points = [(259, 541), (464, 528), (712, 1041), (6, 1038)]
        src = np.float32(np.array(points[:4]))
        dst = np.float32([[0, 0], [W,0],[W, H],[0, H]])

        points2 = [(238, 713), (830, 521), (833, 279), (210, 80)]
        src2 = np.float32(np.array(points2[:4]))
        dst2 = np.float32([[0, 0], [W2, 0], [W2, H2], [0, H2]])

        prespective_transform = cv2.getPerspectiveTransform(src, dst)
        prespective_transform2 = cv2.getPerspectiveTransform(src2, dst2)

        bottom_points = get_transformed_points(boxes_lcs_1, prespective_transform)
        bottom_points2 = get_transformed_points2(boxes_lcs_2, prespective_transform2)
        bottom_points_all= []

        bottom_points_1_mul = get_transformed_points_1_mul(boxes_lcs_1, prespective_transform)



        bottom_points_all = bottom_points_1_mul + bottom_points2
        print('1. bd_pnts : ', bottom_points)
        print('2. bd_pnts : ', bottom_points2)
        print('3. all_bd_pnts : ', bottom_points_all)
        bird_image = bird_eye_view(bottom_points)
        bird_image2 = bird_eye_view2(bottom_points2)
        bird_image_all = bird_eye_view_all(bottom_points_all)
        cv2.imshow('1. bird', bird_image)
        cv2.imshow('2. bird', bird_image2)
        cv2.imshow('3. all_bird', bird_image_all)
        red = (255, 0, 0)



        frame = cv2.circle(frame, (6, 1038), 5, red, 10)
        frame = cv2.circle(frame, (712, 1041), 5, red, 10)
        frame = cv2.circle(frame, (464, 528), 5, red, 10)
        frame = cv2.circle(frame, (259, 541), 5, red, 10)

        frame = cv2.line(frame, (6, 1038), (712,1041), red, 5)
        frame = cv2.line(frame, (712, 1041), (464,528), red, 5)
        frame = cv2.line(frame, (464, 528), (259,541), red, 5)
        frame = cv2.line(frame, (259, 541), (6,1038), red, 5)

#################################3

        frame2 = cv2.circle(frame2, (238, 713), 5, red, 10)
        frame2 = cv2.circle(frame2, (830, 521), 5, red, 10)
        frame2 = cv2.circle(frame2, (833, 279), 5, red, 10)
        frame2 = cv2.circle(frame2, (210, 80), 5, red, 10)

        frame2 = cv2.line(frame2, (238, 713), (830, 521), red, 5)
        frame2 = cv2.line(frame2, (830, 521), (833, 279), red, 5)
        frame2 = cv2.line(frame2, (833, 279), (210, 80), red, 5)
        frame2 = cv2.line(frame2, (210, 80), (238, 713), red, 5)


        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("1. FPS: %.2f" % fps)

        fps2 = 1.0 / (time.time() - start_time2)
        print("2. FPS: %.2f" % fps2)

        result = np.asarray(frame)
        result2 = np.asarray(frame2)

        frame_LCS = cv2.resize(frame,dsize=(360,640))
        frame_LCS_2 = cv2.resize(frame2, dsize=(640, 360))

        result = cv2.cvtColor(frame_LCS, cv2.COLOR_RGB2BGR)
        result2 = cv2.cvtColor(frame_LCS_2, cv2.COLOR_RGB2BGR)

        if not FLAGS.dont_show:
            cv2.imshow("1. Output Video", result)
            cv2.imshow("2. Output Video", result2)
        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
