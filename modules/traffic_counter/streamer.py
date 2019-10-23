import argparse
import imutils
import time
import cv2
import os
import glob
import json
import youtube_dl
import math as m
from enum import Enum
from mmdet.apis import init_detector, inference_detector
from ctypes import *
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import plot

from .tracker import sort as SORT
from .tracker.sort import np

from .tracker.deep_sort import tracker as DEEP_SORT
from .tracker.deep_sort.detection import Detection
from .tracker.deep_sort import nn_matching
#from .tracker.deep_sort import generate_detections as gdet

path_to_libdarknet = "/opt/darknet/libdarknet.so"  # if Linux
winGPUdll = "yolo_cpp_dll.dll"  # if Windows
net_input_size = 416, 416
mouse_event_end = False
first_click = True
line = []
Deep_sort = False


class Detector:
    """ This class is an interface for models on different frameworks """
    class ModelFramework(Enum):
        DARKNET = 1
        MMDET = 2
        TENSORFLOW = 3

    labels = None
    net = None
    framework = None

    def __init__(self, path_to_model):
        labels_path = glob.glob(os.path.sep.join([path_to_model, "*.names"]))
        if len(labels_path) == 0:
            self.framework = Detector.ModelFramework.MMDET
            config_path = glob.glob(os.path.sep.join([path_to_model, "*.py"]))[0]
            checkpoint_path = glob.glob(os.path.sep.join([path_to_model, "*.pth"]))[0]

            # build the model from a config file and a checkpoint file
            self.net = init_detector(config_path, checkpoint_path, device='cuda:0')
            self.labels = self.net.CLASSES
        else:
            self.framework = Detector.ModelFramework.DARKNET
            labels_path = labels_path[0].encode("ascii")
            config_path = glob.glob(os.path.sep.join([path_to_model, "*.cfg"]))[0].encode("ascii")
            weights_path = glob.glob(os.path.sep.join([path_to_model, "*.weights"]))[0].encode("ascii")

            # build the model from a config file and a weights file
            import_darknet()
            self.net = lib.load_network_custom(config_path, weights_path, 0, 1)  # batch size = 1
            self.labels = open(labels_path).read().strip().split("\n")


    def darknet_predict(self, frame, encoder, thresh=.3, nms=.45, hier_thresh=0):
        """ Perform the forward pass using darknet, do thresholding by the confidence and make NMS """
        global Deep_sort

        height, width = frame.shape[:2]
        num_classes = len(self.labels)
        num = c_int(0)
        pnum = pointer(num)
        letter_box = 0

        # resize and swap RB channel
        blob = cv2.dnn.blobFromImage(frame, 1, net_input_size, swapRB=True, crop=False)

        # normalize and convert to IMAGE
        image, blob = array_to_image(blob[0])

        # perform forward pass
        lib.network_predict_image(self.net, image)

        # get DETECTIONs and do thresholding by confidence
        dets = lib.get_network_boxes(self.net, width, height, thresh, hier_thresh, None, 0, pnum, letter_box)
        num = pnum[0]

        # convert DETECTIONs to 3 lists: boxes, confidences, class_ids
        boxes = []
        confidences = []
        class_ids = []
        for j in range(num):
            b = dets[j].bbox
            w = int(b.w)
            h = int(b.h)
            # left top corner
            x = int(b.x - b.w / 2)
            y = int(b.y - b.h / 2)

            max_prob = -1
            class_id = -1
            for k in range(num_classes):
                if dets[j].prob[k] > max_prob:
                    max_prob = dets[j].prob[k]
                    class_id = k

            boxes.append([x, y, w, h])
            confidences.append(dets[j].objectness)
            class_ids.append(class_id)

        # do nms
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, thresh, nms)
        idxs = idxs.flatten() if len(idxs) > 0 else []

        detections = []
        class_ids_to_keep = []
        if Deep_sort:
            bboxes = []
            scores = []
            for i in idxs:
                x, y = (boxes[i][0], boxes[i][1])
                w, h = (boxes[i][2], boxes[i][3])
                # [left top corner, right bottom corner, confidence]
                bboxes.append([x, y, w, h])
                scores.append(confidences[i])
                # keep class_id of objects, that passed threshold
                class_ids_to_keep.append(class_ids[i])
            features = encoder(frame, bboxes)
            detections = [Detection(bbox, score, feature, class_id) for bbox, score, feature, class_id in
                              zip(bboxes, scores, features, class_ids_to_keep)]
        else:
            for i in idxs:
                x, y = (boxes[i][0], boxes[i][1])
                w, h = (boxes[i][2], boxes[i][3])
                # [left top corner, right bottom corner, confidence]
                detections.append([x, y, x + w, y + h, confidences[i]])
                # keep class_id of objects, that passed threshold
                class_ids_to_keep.append(class_ids[i])
            detections = np.asarray(detections)

        # free array from memory by pointer
        lib.free_detections(dets, num)

        return detections, class_ids_to_keep


    def mmdet_predict(self, image, confidence):
        """ Perform the forward pass using mmdet and do thresholding by the confidence """
        # do forward pass
        bbox_result = inference_detector(self.net, image)

        # get class ids and detections as ndarrays
        class_ids = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        class_ids = np.concatenate(class_ids)
        bboxes = np.vstack(bbox_result)

        # do thresholding by confidence
        scores = bboxes[:, -1]
        inds = scores > confidence
        bboxes = bboxes[inds, :]
        class_ids = class_ids[inds]

        return bboxes, class_ids


    def predict(self, image, encoder, confidence=0.3, nms=0.45):
        """ Perform the forward pass and the thresholding """
        if self.framework is Detector.ModelFramework.DARKNET:
            bboxes, class_ids = self.darknet_predict(image, encoder, confidence, nms)
        elif self.framework is Detector.ModelFramework.MMDET:
            bboxes, class_ids = self.mmdet_predict(image, confidence)
        else: raise NotImplementedError()

        return bboxes, class_ids


    def dispose(self):
        raise NotImplementedError()


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


def import_darknet():
    """ Import darknet library and configure its functions """
    global lib
    has_gpu = True
    lib = CDLL(winGPUdll, RTLD_GLOBAL) if os.name == "nt" else CDLL(path_to_libdarknet, RTLD_GLOBAL)

    lib.network_width.argtypes = [c_void_p]
    lib.network_width.restype = c_int
    lib.network_height.argtypes = [c_void_p]
    lib.network_height.restype = c_int

    if has_gpu:
        set_gpu = lib.cuda_set_device
        set_gpu.argtypes = [c_int]

    lib.get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
    lib.get_network_boxes.restype = POINTER(DETECTION)

    lib.free_detections.argtypes = [POINTER(DETECTION), c_int]

    lib.load_network_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
    lib.load_network_custom.restype = c_void_p

    lib.do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

    lib.network_predict_image.argtypes = [c_void_p, IMAGE]
    lib.network_predict_image.restype = POINTER(c_float)


def array_to_image(arr):
    """ Normalize the ndarray and convert it to the IMAGE """
    # need to return old values to avoid python freeing memory
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w, h, c, data)
    return im, arr


def detect_image(net, labels, frame, encoder, thresh=.5, hier_thresh=0, nms=.5):
    """ Resize and normalize image, perform the forward pass of the net and make nms """
    global Deep_sort

    height, width = frame.shape[:2]
    num_classes = len(labels)
    num = c_int(0)
    pnum = pointer(num)
    letter_box = 0

    # resize and swap RB channel
    blob = cv2.dnn.blobFromImage(frame, 1, net_input_size, swapRB=True, crop=False)

    # normalize and convert to IMAGE
    image, blob = array_to_image(blob[0])

    # perform forward pass
    lib.network_predict_image(net, image)

    # get DETECTIONs and do thresholding by confidence
    dets = lib.get_network_boxes(net, width, height, thresh, hier_thresh, None, 0, pnum, letter_box)
    num = pnum[0]

    # convert DETECTIONs to 3 lists: boxes, confidences, class_ids
    boxes = []
    confidences = []
    class_ids = []
    for j in range(num):
        b = dets[j].bbox
        w = int(b.w)
        h = int(b.h)
        # left top corner
        x = int(b.x - b.w / 2)
        y = int(b.y - b.h / 2)

        max_prob = -1
        class_id = -1
        for k in range(num_classes):
            if dets[j].prob[k] > max_prob:
                max_prob = dets[j].prob[k]
                class_id = k

        boxes.append([x, y, w, h])
        confidences.append(dets[j].objectness)
        class_ids.append(class_id)

    # do nms
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, thresh, nms)
    idxs = idxs.flatten() if len(idxs) > 0 else []

    detections = []
    class_ids_to_keep = []
    bboxes = []
    scores = []

    if Deep_sort:
        for i in idxs:
            x, y = (boxes[i][0], boxes[i][1])
            w, h = (boxes[i][2], boxes[i][3])
            # [left top corner, right bottom corner, confidence]
            bboxes.append([x, y, w, h])
            scores.append(confidences[i])
            # keep class_id of objects, that passed threshold
            class_ids_to_keep.append(class_ids[i])
    else:
        for i in idxs:
            x, y = (boxes[i][0], boxes[i][1])
            w, h = (boxes[i][2], boxes[i][3])
            # [left top corner, right bottom corner, confidence]
            detections.append([x, y, x + w, y + h, confidences[i]])
            # keep class_id of objects, that passed threshold
            class_ids_to_keep.append(class_ids[i])

    if Deep_sort:
        features = encoder(frame, bboxes)
        detections = [Detection(bboxes, scores, features) for bboxes, scores, features in zip(bboxes, scores, features)]
    else:
        detections = np.asarray(detections)

    # free array from memory by pointer
    lib.free_detections(dets, num)

    return detections, class_ids_to_keep

  
def intersect(a, b, c, d):
    """ Return true if line segments AB and CD intersect """
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)

def parse_args():
    """ Parse and return dictionary of the arguments """
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,
                    help="the path to input video or stream")
    ap.add_argument("-o", "--output_dir", required=True,
                    help="the path to output video directory")
    ap.add_argument("-m", "--model", required=True,
                    help="the path to model directory")
    ap.add_argument("-t", "--tracker", default="sort",
                    help="the name of used tracker")
    ap.add_argument("-c", "--confidence", type=float, default=0.3,
                    help="minimum probability to filter weak detections")
    ap.add_argument("-n", "--nms", type=float, default=0.45,
                    help="threshold when applying non-maxima suppression")
    ap.add_argument("-s", "--stats", action='store_true',
                    help="the flag for collecting statistics")
    ap.add_argument("--frame_to_break", type=int, default=1000,
                    help="maximum number of the frames to process")
    args = vars(ap.parse_args())
    return args


def clear_dir(directory):
    """ Remove all files from the directory """
    files = glob.glob(os.path.sep.join([directory, "*"]))
    for f in files:
        os.remove(f)


def cross_direction_on_right_side(line, point):
    v1 = [line[1][0] - line[0][0], line[1][1] - line[0][1]]
    v2 = [line[1][0] - point[0], line[1][1] - point[1]]
    xp = v1[0] * v2[1] - v1[1] * v2[0]
    if xp > 0:
        return False
    elif xp < 0:
        return True
    else:
        return False


def track_and_count(dets, tracker, line, counters, class_ids, class_to_detect, class_buffer_size,frame_index):
    """ Track objects, check and count intersection of gate line """
    global Deep_sort
    # track objects
    to_crop = []
    if Deep_sort:
        tracker.predict()
        tracker.update(dets, class_buffer_size)
        tracks = tracker.tracks
    else:
        if dets.size != 0:
            dets = np.insert(dets, [5], list([[i] for i in class_ids]), axis=1)
        buf=False
        if(frame_index==0):
            buf=True
        tracks = tracker.update(dets, class_buffer_size,buf)
    previous = track_and_count.boxes.copy() if hasattr(track_and_count, "boxes") else []
    previous_ids = track_and_count.ids.copy() if hasattr(track_and_count, "ids") else []
    if not hasattr(track_and_count, "log"):
        track_and_count.log = []
        for e in line:
            track_and_count.log.append([0])

    track_and_count.boxes = []
    track_and_count.ids = []

    if Deep_sort:
        tracks = []
        for e in tracker.tracks:
            if not e.is_confirmed() or e.time_since_update > 1:
                continue
            box = e.to_tlbr()
            tracks.append([box[0], box[1], box[2], box[3], e.track_id, int(np.median(e.class_id))])

    for i in range(len(tracks)):
        # track = tracks[i]
        # top left corner is (x0, y0), bottom right corner is (x1, y1)
        x0, y0 = int(tracks[i][0]), int(tracks[i][1])
        x1, y1 = int(tracks[i][2]), int(tracks[i][3])
        index_id = int(tracks[i][4])
        class_id = int(tracks[i][5])
        center = int((x0 + x1) / 2), int((y0 + y1) / 2)
        shift = (center, None)
        # if the object was detected in the prev frame calculate the shift and check intersection
        for line_indx in range(len(line)):
            if index_id in previous_ids:
                previous_box = previous[previous_ids.index(index_id)]
                prev_center = previous_box[4][0]
                if intersect(center, prev_center, line[line_indx][0], line[line_indx][1]):
                    if len(class_to_detect[line_indx]) == 0 or class_ids[i] in class_to_detect[line_indx]:
                        if index_id not in track_and_count.log[line_indx]:
                            if cross_direction_on_right_side(line[line_indx], center):
                                track_and_count.log[line_indx].insert(0, index_id)
                                if len(track_and_count.log[line_indx]) > 100:
                                    del (track_and_count.log[line_indx][-1])
                                to_crop.append([x0, y0, x1, y1, index_id, class_id, line_indx])
                                if class_id in counters[line_indx]:
                                    counters[line_indx][class_id] += 1
                                else:
                                    counters[line_indx][class_id] = 1
                shift = (center, prev_center)

        # append the box to the box dictionary
        box = [x0, y0, x1, y1, shift, index_id, class_id]

        track_and_count.boxes.append(box)
        track_and_count.ids.append(index_id)
    return track_and_count.boxes, to_crop


def collect_stats(boxes, class_ids, ways, frameIndex, stats_collect, frame_by_frame):
    """Collect stats"""
    if stats_collect is True:
        i = int(0)
        one_frame = []
        for box in boxes:
            if (box[5] in ways.keys()) is False:
                ways[box[5]] = []
                ways[box[5]].append(class_ids[i])
            ways[box[5]].append(
                [int((box[2] + box[0]) / 2), int(((box[3] + box[1]) / 2)-20), int(box[0]), int(box[1]-20),
                 int(box[2] - box[0]), int(box[3] - box[1]), frameIndex])
            i = i + 1
            one_frame.append([box[0], (box[1]-20), box[2], (box[3]-20)])
        frame_by_frame.append(one_frame)


def save_stats(ways, stats_collect, arg, frame_by_frame):
    """Arhive and save collected stats to files"""
    if stats_collect is True:
        path_to_frame_by_frame = os.path.sep.join([arg, 'frame_by_frame.txt'])
        with open(path_to_frame_by_frame, 'w') as f:
            json.dump(frame_by_frame, f)
        arh_ways = []
        for i in ways:
            arh_ways.append(ways[i])
        path_to_ways = os.path.sep.join([arg, 'ways.txt'])
        with open(path_to_ways, 'w') as f:
            json.dump(arh_ways, f)


def draw(frame, line, counters, boxes, colors, labels):
    """ Draw the gate line, counter, object borders, ids and shift """
    # for each object draw its border, id and shift
    for i in range(len(boxes)):
        left_top = boxes[i][0], boxes[i][1]
        right_bottom = boxes[i][2], boxes[i][3]
        center, prev_center = boxes[i][4]

        color = [int(c) for c in colors[boxes[i][5] % len(colors)]]
        cv2.rectangle(frame, left_top, right_bottom, color, 2)
        text = "{},{}".format(boxes[i][5], labels[boxes[i][6]])
        cv2.putText(frame, text, (left_top[0], left_top[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if ((prev_center is not None)and(prev_center!=0)):
            cv2.line(frame, center, prev_center, color, 3)

    # draw gate line
    if hasattr(draw, "line_colors"):
        l_color = draw.line_colors
    else:
        draw.line_colors = [(255, 0, 0),
                            (0, 255, 0),
                            (0, 0, 255),
                            (255, 255, 0),
                            (255, 0, 255),
                            (0, 255, 255),
                            (0, 0, 0),
                            (255, 255, 255),
                            (255, 128, 128),
                            (128, 128, 255),
                            (128, 255, 128),
                            (128, 128, 128)]
        l_color = draw.line_colors
    w=int(frame.shape[1]/400)
    for line_indx in range(len(line)):
        cv2.line(frame, line[line_indx][0], line[line_indx][1], l_color[line_indx], w)
    # draw counter
    '''sum_ = []
    for i in range(len(counters)):
        sum_.append(0)
        for k in counters[i].keys():
            sum_[i] += counters[i][k]

    cv2.putText(frame, str(sum(sum_)), (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2)
    h = 0
    for i in range(len(line)):
        cv2.putText(frame, "line#{}: {}".format(i, sum_[i]), (50, 80 + 23 * h), cv2.FONT_HERSHEY_DUPLEX, 1.0, l_color[i], 1)
        h += 1
        for k in counters[i].keys():
            cv2.putText(frame, "    {}: {}".format(labels[k], counters[i][k]), (50, 80 + 23 * h), cv2.FONT_HERSHEY_DUPLEX, 1.0,
                        (255, 255, 0), 1)
            h += 1'''


def save(frame, output_dir, save_png=False, frame_index=-1):
    """ Save the frame to the output video (and to .png) """
    # check if the video writer is exist
    if not hasattr(save, "writer") or not save.writer.isOpened():
        # initialize our video writer
        dst = os.path.sep.join([output_dir, "output_drones_yolov3-slim832.avi"])
        fourcc = cv2.VideoWriter_fourcc(*"DIVX")
        save.writer = cv2.VideoWriter(dst, fourcc, 30, (frame.shape[1], frame.shape[0]), True)

    # write the output frame to video
    save.writer.write(frame)

    # write the output frame as .png
    if save_png:
        path_to_frame = os.path.sep.join([output_dir, "frame-{}.png".format(frame_index)])
        cv2.imwrite(path_to_frame, frame)
def centroid_histogram(clt):
    """ Make histogram for dominant colors"""
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist


def plot_colors(hist, centroids):
    """ Returns image of bar representing histogram"""

    # initialize the bar chart representing the relative frequency of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    # loop over the percentage of each cluster and the color of each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50), color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar


def crop_objects(frame, to_crop, frame_index, labels, padding=(10, 10), n_clusters=5, saveImages=False, showImages=False):
    """ Process objects from main frame that crossed the line"""

    for e in to_crop:
        img = frame[e[1]-padding[1]:e[3]+padding[1], e[0]-padding[0]:e[2]+padding[0]]

        image = img.reshape((img.shape[0] * img.shape[1], 3))
        clt = KMeans(n_clusters=n_clusters)
        clt.fit(image)
        hist = centroid_histogram(clt)
        bar = plot_colors(hist, clt.cluster_centers_)
        if showImages:
            cv2.imshow("hist", bar)
            cv2.imshow("f:{} line{} id:{} class:{} ".format(frame_index, e[6], e[4], labels[e[5]],), img)
            cv2.waitKey(0)
            cv2.destroyWindow("hist")
            cv2.destroyWindow("f:{} line{} id:{} class:{} ".format(frame_index, e[6], e[4], labels[e[5]]))
        if saveImages:
            cv2.imwrite("output/f:{}\nline{}\nid:{}\nclass:{}\n.png".format(frame_index, e[6], e[4], labels[e[5]]), img)
            cv2.imwrite("output/hist\nf:{},line{},id:{},class:{}.png".format(frame_index, e[6], e[4], labels[e[5]]), bar)

def save_in_json():
    """ Save settings file in json format """
    global line
    temp_dict = {'lines': line, "filter": list([[]] * len(line))}
    json_dict = json.dumps(temp_dict)
    with open('input/settings.json', 'w') as json_file:
        json.dump(json_dict, json_file)


def load_json(lines_path):
    global line
    """ Load settings file from json file """

    with open(lines_path) as file:
        data = json.loads(json.load(file))

    line = data['lines']
    filters = data['filter']
    return filters


def lines_list2tuple(counters):
    """ Reformat read lines_ array """
    global line
    temp = []

    for l in line:
        for i in range(len(l)):
            if i == 0:
                temp.append([tuple(l[i])])
            else:
                temp[-1].append(tuple(l[i]))

    line = temp
    # init counters
    for i in range(len(line)):
        counters.append({})


def read_stats(labels):
    """ Read stats data for plots """
    label, values = [], []
    total_v = {}

    with open("stats.txt") as file:
        total_frames = file.readline()
        counters = eval(file.readline())

    for e in counters:
        label.append([])
        values.append([])
        for k in e.keys():
            label[-1].append("{}".format(labels[k]))
            values[-1].append(e[k])
            if labels[k] not in total_v.keys():
                total_v[labels[k]] = 0
            total_v[labels[k]] += e[k]
    return total_frames, label, values, total_v


def make_plots(values, label, total_v, width=1200, height=1500, rows=2, columns=1, main_title="Total info"):
    fig = {
        "data": [],
        "layout": {
            "autosize": False,
            "width": width,
            "height": height,
            "title": main_title,
            "grid": {"rows": rows + 1, "columns": columns},
            "annotations": []
        }
    }
    for i in range(rows):
        for j in range(columns):
            temp = {"values": values[i],
                    "labels": label[i],
                    "domain": {"row": i},
                    "title": "Line#{}".format(i),
                    "titlefont": {
                        "size": 20
                    },
                    "name": "Line#{}".format(i),
                    "hoverinfo": "value+label",
                    "type": "pie"
                    }

            fig["data"].append(temp)
            temp = {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "",
                "x": 0.5,
                "y": 0.5
            }
            fig["layout"]["annotations"].append(temp)
    fig["data"].append({
        "values": list(total_v.values()),
        "labels": list(total_v.keys()),
        "domain": {"row": rows},
        "title": "Total",
        "titlefont": {
            "size": 20
        },
        "name": "Total",
        "textposition": "inside",
        "hoverinfo": "value+label",
        "type": "pie"
    })
    fig["layout"]["annotations"].append({
        "font": {
            "size": 20
        },
        "showarrow": False,
        "text": "",
        "x": 0.5,
        "y": 0.5
    })
    return fig


def check_youtube(src, quality="opt"):
    if src.startswith("https://www.youtube.com/") or src.startswith("https://youtu.be/"):
        format_index = -1
        max_uint16 = 0xFFFF

        ydl = youtube_dl.YoutubeDL({})
        info_dict = ydl.extract_info(src, download=False)
        formats = info_dict.get('formats', None)
        diagonals = dict([
            (i, m.sqrt(f.get('height') ** 2 + f.get('width') ** 2))
            for i, f in enumerate(formats)
            if f.get('height') is not None and f.get('width') is not None
        ])

        if quality == "min":
            min_diag = max_uint16
            for idx, diag in diagonals.items():
                if diag < min_diag:
                    min_diag = diag
                    format_index = idx
        elif quality == "max":
            max_diag = 0
            for idx, diag in diagonals.items():
                if diag > max_diag:
                    max_diag = diag
                    format_index = idx
        elif quality == "opt":
            min_diff = max_uint16
            net_diag = m.sqrt(net_input_size[0] ** 2 + net_input_size[1] ** 2)
            for idx, diag in diagonals.items():
                diff = diag - net_diag
                diff = diff if diff >= 0 else max_uint16
                if diff < min_diff:
                    min_diff = diff
                    format_index = idx

        src = formats[format_index].get('url', None)
    return src

def assign_tracker(tracker_model):
    global Deep_sort
    model = tracker_model.lower()
    if model == "sort":
        Deep_sort = False
    elif model == "deep_sort":
        Deep_sort = True

def save_current_det(counters,labels,path,exit):
    sum_ = []
    for i in range(len(counters)):
        sum_.append(0)
        for k in counters[i].keys():
            sum_[i] += counters[i][k]
    current_det={}
    for i in range(len(line)):
        current_det[i] = [0,0]
        current_det[i][0]=sum_[i]
        current_det[i][1]={}
        for k in counters[i].keys():
            current_det[i][1][k]=[labels[k], counters[i][k]]
    current_det["exit"]=exit
    path_to_current_det = os.path.sep.join([path, 'current_det.txt'])
    with open(path_to_current_det, 'w') as f:
        json.dump(current_det, f)

class Streamer(object):
    def __init__(self, src,lines_path,nn_type,frame_max,upl):
        self.upl=upl
        self.save_png = False
        self.save_video = False
        self.save_cropped_images = False
        self.show_cropped_images = False
        self.show_frames = True
        self.show_plots = False
        self.quality = "max"
        self.n_clusters = 5
        self.padding = (10, 10)
        self.max_tracker_age = 10
        self.min_tracker_hit = 3
        self.max_cosine_distance = 0.3
        self.nn_budget = None
        self.nms_max_overlap = 1.0
        self.class_buffer_size = 1
        self.ways = {}
        self.frame_index = 0
        self.counters = []
        self.class_to_detect = []
        self.frame_by_frame = []
        self.total_detector = 0.
        self.total_tracker = 0.
        self.total_output = 0.
        self.total_videowriter = 0.
        self.encoder=None
        self.args = {}
        #self.clear_dir(args["output_dir"])
        self.stats_collect = True
        self.src = src
        # load the model
        model=os.path.join("modules/traffic_counter/models",nn_type)
        self.tracker_name="sort"
        assign_tracker(self.tracker_name)
        self.detector = Detector(model)
        # initialize a list of colors
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(200, 3), dtype="uint8")
        # init tracker
        self.Deep_sort=False
        if self.Deep_sort!=True:
            self.tracker = SORT.Sort(self.max_tracker_age, self.min_tracker_hit)
        else:
            self.feature_model_filename = 'modules/traffic_counter/tracker/deep_sort/mars-small128.pb'
            self.encoder = gdet.create_box_encoder(self.feature_model_filename, batch_size=25)
            self.metric = nn_matching.NearestNeighborDistanceMetric("cosine",self.max_cosine_distance,self.nn_budget)
            self.tracker = DEEP_SORT.Tracker(metric)
        self.video = cv2.VideoCapture(src)
        self.class_to_detect = load_json(lines_path)
        lines_list2tuple(self.counters)
        self.video = cv2.VideoCapture(src)
        self.confidence=0.15
        self.nms=0.45
        self.frame_max=frame_max
        self.end=False

    def __del__(self):
        self.video.release()

    def get_frame(self):
        global line
        success, frame = self.video.read()
        if success:
            detections, class_ids = self.detector.predict(frame, self.encoder, confidence=self.confidence, nms=self.nms)
            bboxes, to_crop = track_and_count(detections, self.tracker, line, self.counters, class_ids, self.class_to_detect,self.class_buffer_size,self.frame_index)
            # process objects that crossed lines
            if self.save_cropped_images or self.show_cropped_images:
                crop_objects(frame, to_crop, self.frame_index, self.detector.labels, self.padding, self.n_clusters, self.save_cropped_images,
                             self.show_cropped_images)
            # write stats if necessary
            if self.stats_collect:
                collect_stats(bboxes, class_ids, self.ways, self.frame_index, self.stats_collect, self.frame_by_frame)
            # draw gate line, counters, object borders and ids on frame
            draw(frame, line, self.counters, bboxes, self.colors, self.detector.labels)
            self.frame_index=self.frame_index+1
            if(self.frame_index==(self.frame_max-1)):
                save_stats(self.ways, self.stats_collect, self.upl, self.frame_by_frame)
                self.end=True
            save_current_det(self.counters,self.detector.labels,self.upl,self.end)
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()
            return frame
        return None