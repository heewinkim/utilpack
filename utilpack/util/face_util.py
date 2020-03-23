# -*- coding: utf-8 -*-
"""
===============================================
face_util module
===============================================

========== ====================================
========== ====================================
 Module     face_util module
 Date       2019-07-29
 Author     heewinkim
========== ====================================

*Abstract*
    * PyFaceUtil 클래스 제공 - 메서드 (get_align_faces,vis_face_landmarks
    * shape_predictor 모델 다운로드 경로

===============================================
"""

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__)) # common.util
parent_dir = os.path.dirname(current_dir) # common
sys.path.insert(0,parent_dir)
from dlib import shape_predictor,rectangle
from collections import OrderedDict
from utilpack.core.maths import PyMaths
from urllib.request import urlopen
from tqdm import tqdm
import numpy as np
import math
import cv2
import os


class FaceAligner:

    def __init__(self,desiredLeftEye=(0.3,0.3),desiredFaceWidth=150, desiredFaceHeight=150,predictor_type=5):
        """

        :param desiredLeftEye: eye margin
        :param desiredFaceWidth: face width
        :param desiredFaceHeight: face height
        :param predictor_type: number of landmark, 5 or 68
        """

        raise NotImplementedError("Write '_download_landmark' inner method.")

        self._predictor_type=int(predictor_type)

        PREDICTOR_PATH = current_dir + '/landmark_{}.dat'.format(predictor_type)

        if not os.path.exists(PREDICTOR_PATH):
            self._download_landmark(PREDICTOR_PATH)

        self._predictor = shape_predictor(PREDICTOR_PATH)
        if predictor_type==68:
            self._landmarks_idxes = OrderedDict([
                ("right_lip", range(48, 49)),
                ("left_lip", range(54, 55)),
                ("right_eye", range(36, 42)),
                ("left_eye", range(42, 48)),
                ("nose", range(30, 31)),
                ("jaw", range(8, 9))
            ])
        else:
            self._landmarks_idxes = OrderedDict([
                ("right_eye", range(2, 4)),
                ("left_eye", range(0, 2)),
                ("nose", range(4, 5))
                ])

        self._desiredLeftEye = desiredLeftEye
        self._desiredFaceWidth = desiredFaceWidth
        self._desiredFaceHeight = desiredFaceHeight

        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self._desiredFaceHeight is None:
            self._desiredFaceHeight = self._desiredFaceWidth

    def _download_landmark(self,dst_file):

        url = 'landmark_{}.dat'.format(self._predictor_type)
        print('download landmark.dat..')
        with urlopen(url) as src, open(dst_file, 'wb') as dst:
            data = src.read(1024)
            pbar = tqdm(total=int(np.ceil(src.length/1024)))
            while len(data) > 0:
                pbar.update(1)
                dst.write(data)
                data = src.read(1024)
            pbar.close()

    def _get_face_orientation(self,image_shape, landmarks):
        """
        얼굴 시선에 대한 정보를 얻습니다.

        :param image_shape: tuple,(height,width,channel)
        :param landmarks: left_eye,right_eye,nose,left_lip,right_lip,jaw
        :return: float(roll),float(pitch),float(yaw),list(coordinate_pts)
        """

        image_size = image_shape[:2]

        image_points = np.array([
            landmarks['nose'],  # Nose tip
            landmarks['jaw'],  # Chin
            landmarks['left_eye'],  # Left eye left centroid
            landmarks['right_eye'],  # Right eye right centroid
            landmarks['left_lip'],  # Left Mouth corner
            landmarks['right_lip']  # Right mouth corner
        ], dtype="double")

        model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-165.0, 170.0, -135.0),  # Left eye left centroid
            (165.0, 170.0, -135.0),  # Right eye right centroid
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner
        ])

        # Camera internals
        center = (image_size[1] / 2, image_size[0] / 2)
        focal_length = center[0] / np.tan(60 / 2 * np.pi / 180)
        camera_matrix = np.array([[focal_length, 0, center[0]],[0, focal_length, center[1]],[0, 0, 1]], dtype="double")

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,dist_coeffs, flags=cv2.cv2.SOLVEPNP_ITERATIVE)

        axis = np.float32([[200, 0, 0], [0, 200, 0], [0, 0, 200]])

        imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        coordinate_pts = [landmarks['nose']]+[list(map(float,v[0])) for v in imgpts]

        rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

        proj_matrix = np.hstack((rvec_matrix, translation_vector))
        eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

        pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]

        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))

        return float(roll), float(pitch), float(yaw), coordinate_pts

    def _extract_landmarkPts(self,shape,idxes,mean_pts=True):
        """
        dlib shape predictor결과에서 idxex리스트에 해당하는 각 특징들을 추출합니다.

        :param shape: dlib's shape_predictor result
        :param idxes: idx list
        :param mean_pts: idx가 여러개인경우 그 특징 좌표들을 중심점인 하나의 좌표로 압축할지
        :return: tuple(points)
        """

        points = [(shape.part(idx).x,shape.part(idx).y) for idx in idxes]
        if mean_pts:
            points = tuple([int(value) for value in np.mean(points,axis=0)])
        return points

    def _align(self,faceImage,leftEyePts,rightEyePts):
        """
        보정된 얼굴을 얻습니다.

        :param faceImage: 크롭된 얼굴이미지
        :param leftEyePts: 좌안
        :param rightEyePts: 우안
        :return: aligned_faceImage
        """

        dY = rightEyePts[1] - leftEyePts[1]
        dX = rightEyePts[0] - leftEyePts[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180
        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self._desiredLeftEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self._desiredLeftEye[0])
        desiredDist *= self._desiredFaceWidth
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyePts[0] + rightEyePts[0]) // 2,
                      (leftEyePts[1] + rightEyePts[1]) // 2)

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        tX = self._desiredFaceWidth * 0.5
        tY = self._desiredFaceHeight * self._desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
        (w, h) = (self._desiredFaceWidth, self._desiredFaceHeight)
        aligned_faceImage = cv2.warpAffine(faceImage, M, (w, h), flags=cv2.INTER_CUBIC)

        # return the aligned face
        return aligned_faceImage

    def align_face(self,faceImage,rgbImage=True):
        """
        align_face

        :param faceImage: cv faceImage
        :return: aligned faceImages,landmarks_list
        """

        # convert the landmark (x, y)-coordinates to a NumPy array
        rgb_image = cv2.cvtColor(faceImage, cv2.COLOR_BGR2RGB)
        r = rectangle(0,0,faceImage.shape[1],faceImage.shape[0])

        shape = self._predictor(rgb_image, r)

        left_eye = self._extract_landmarkPts(shape, self._landmarks_idxes['left_eye'])
        right_eye= self._extract_landmarkPts(shape, self._landmarks_idxes['right_eye'])
        nose = self._extract_landmarkPts(shape, self._landmarks_idxes['nose'])
        angle = PyMaths.get_degree(right_eye, left_eye)
        landmarks = {
            'left_eye': left_eye,
            'right_eye': right_eye,
            'nose': nose,
            'angle': angle
        }

        if self._predictor_type==68:
            left_lip = self._extract_landmarkPts(shape, self._landmarks_idxes['left_lip'])
            right_lip = self._extract_landmarkPts(shape, self._landmarks_idxes['right_lip'])
            jaw = self._extract_landmarkPts(shape, self._landmarks_idxes['jaw'])
            landmarks.update({
                'left_lip': left_lip,
                'right_lip': right_lip,
                'jaw': jaw})
            roll, pitch, yaw, coordinate_pts = self._get_face_orientation(faceImage.shape, landmarks)
            landmarks.update({
                'roll': roll,
                'pitch': pitch,
                'yaw': yaw})

        if rgbImage:
            aligned_faceImage = self._align(rgb_image,left_eye,right_eye)
        else:
            aligned_faceImage = self._align(faceImage, left_eye, right_eye)

        for k in self._landmarks_idxes.keys():
            landmarks[k] = (max(0,min(faceImage.shape[1],landmarks[k][0])),max(0,min(faceImage.shape[0],landmarks[k][1])))

        return aligned_faceImage, landmarks


class PyFaceUtil(object):

    @staticmethod
    def crop_face(image,x1,y1,x2,y2,copy=True):
        """
        cropped 된 얼굴을 반환합니다.

        Overloaded function list.

        :param image: image
        :param x1,y1,x2,y2: face's coordinates, each points are left,top,right,bottom.
        :param copy: return copy data, default True


        :return: cropped face
        """
        if copy:
            return image[y1:y2,x1:x2,:].copy()
        else:
            return image[y1:y2, x1:x2, :]

    @staticmethod
    def draw_face(image,x1,y1,x2,y2,color=(255,0,255), thickness=3):
        """
        img_cv 위에 face 사각형을 그립니다.

        :param image: image
        :param x1,y1,x2,y2: face's coordinates, each points are left,top,right,bottom.
        :return: None
        """
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)