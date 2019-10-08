import cv2
import time
import numpy as np
import os
from tqdm import tqdm
import pickle
import urllib.request


def get_iou(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Parameters
    ----------
    boxA = np.array( [ xmin,ymin,xmax,ymax ] )
    boxB = np.array( [ xmin,ymin,xmax,ymax ] )
    Returns
    -------
    float
        in [0, 1]
    """

    bb1 = dict()
    bb1['x1'] = boxA[0]
    bb1['y1'] = boxA[1]
    bb1['x2'] = boxA[2]
    bb1['y2'] = boxA[3]

    bb2 = dict()
    bb2['x1'] = boxB[0]
    bb2['y1'] = boxB[1]
    bb2['x2'] = boxB[2]
    bb2['y2'] = boxB[3]

    # Determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Compute the area of both bounding boxes area
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    assert iou >= 0.0
    assert iou <= 1.0

    return iou


def preprocessing_wider_dataset(root_dir, splits=['train', 'val'], min_w=15, min_h=15):
    """
    wider dataset preprocessing
    data structure rule : {root_dir}/wider_face_split
                          {root_dir}/WIDER_train
                          {root_dir}/WIDER_val

    :param root_dir: root directory
    :param splits: list, dataset type for test
    :param min_w: face minimum width pixel
    :param min_h: face minimum height pixel
    :return: dict, dataset {'image_path': np.array([x1,y1,x2,y2])}
    """

    # Extract bounding box ground truth from dataset annotations, also obtain each image path
    # and maintain all information in one dictionary
    dataset = dict()

    for split in splits:
        with open('{}/wider_face_split/wider_face_{}_bbx_gt.txt'.format(root_dir, split), 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.split('\n')[0]
            if line.endswith('.jpg'):
                image_path = os.path.join(root_dir, 'WIDER_%s' % (split), 'images', line)
                dataset[image_path] = []
            line_components = line.split(' ')
            if len(line_components) > 1:

                # Discard annotation with invalid image information, see dataset/wider_face_split/readme.txt for details
                if int(line_components[7]) != 1:
                    x1 = int(line_components[0])
                    y1 = int(line_components[1])
                    w = int(line_components[2])
                    h = int(line_components[3])

                    # In order to make benchmarking more valid, we discard faces with width or height less than 15 pixel,
                    # we decide that face less than 15 pixel will not informative enough to be detected
                    if w > min_w and h > min_h:
                        dataset[image_path].append(
                            np.array([x1, y1, x1 + w, y1 + h]))

    return dataset


def evaluate(detect_func, dataset=None, use_batch=None):
    """
    evaluate detection model on wider dataset's validation data

    :param detect_func: face detection function , rule : function(cv_image) -> x1,y1,x2,y2
    :param root_dir: root directory
    :param splits: list, dataset type for test
    :param min_w: face minimum width pixel
    :param min_h: face minimum height pixel
    :param use_batch: if None, use all of dataset(wider) otherwise use ratio of use_batch(float)
    :return: dict,test result
    """
    if dataset is None:
        file = urllib.request.urlopen('https://kr-py-prd-data.s3.ap-northeast-2.amazonaws.com/common/val_wider_dataset.pkl')
        dataset = pickle.load(file)
    if use_batch:
        n_use = int(len(dataset) * use_batch)
        dataset = {k: v for k, v in zip(list(dataset.keys())[:n_use], list(dataset.values())[:n_use])}
    n_data = len(dataset.keys())
    data_total_iou = 0
    data_total_precision = 0
    data_total_inference_time = 0
    positive_false = 0
    negative_true = 0
    total_face = 0

    # Evaluate face detector and iterate it over dataset
    for i, key in tqdm(enumerate(dataset), total=n_data):
        image_data = cv2.imread(key)
        face_bbs_gt = np.array(dataset[key])
        total_gt_face = len(face_bbs_gt)
        total_face += total_gt_face

        start_time = time.time()
        face_pred = detect_func(image_data)
        inf_time = time.time() - start_time
        data_total_inference_time += inf_time

        total_iou = 0
        tp = 0
        pred_dict = dict()
        if len(face_pred) < len(face_bbs_gt):
            positive_false += (len(face_bbs_gt) - len(face_pred))
        elif len(face_pred) > len(face_bbs_gt):
            negative_true += (len(face_pred) - len(face_bbs_gt))

        for gt in face_bbs_gt:
            max_iou_per_gt = 0
            for i, pred in enumerate(face_pred):
                if i not in pred_dict.keys():
                    pred_dict[i] = 0
                iou = get_iou(gt, pred)
                if iou > max_iou_per_gt:
                    max_iou_per_gt = iou
                if iou > pred_dict[i]:
                    pred_dict[i] = iou
            total_iou = total_iou + max_iou_per_gt

        if total_gt_face != 0:
            if len(pred_dict.keys()) > 0:
                for i in pred_dict:
                    if pred_dict[i] >= 0.5:
                        tp += 1
                precision = float(tp) / float(total_gt_face)

            else:
                precision = 0

            image_average_iou = total_iou / total_gt_face
            image_average_precision = precision

            data_total_iou += image_average_iou
            data_total_precision += image_average_precision

    result = dict()
    result['average_iou'] = float(data_total_iou) / float(n_data)
    result['mean_average_precision'] = float(data_total_precision) / float(n_data)
    result['average_inferencing_time'] = float(data_total_inference_time) / float(n_data)
    result['total_face'] = total_face
    result['positive_false'] = positive_false
    result['negative_true'] = negative_true

    for k, v in result.items():
        print(detect_func.__name__, ':', k, v)

    return result
