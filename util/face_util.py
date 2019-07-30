# -*- coding: utf-8 -*-
"""
===============================================
face_util module
===============================================

========== ====================================
========== ====================================
 Module     face_util module
 Date       2019-07-29
 Author     hian
 Comment    `관련문서링크 <call to heewinkim >`_
========== ====================================

*Abstract*
    * FaceAligner 클래스 제공 - 메서드 (get_align_faces,vis_face_landmarks

===============================================
"""


import cv2
import numpy as np
from dlib import shape_predictor,rectangle
from collections import OrderedDict
import matplotlib.pyplot as plt


#For dlib’s 68-point facial landmark detector:
FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("inner_mouth", (60, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])

#For dlib’s 5-point facial landmark detector:
FACIAL_LANDMARKS_5_IDXS = OrderedDict([
    ("right_eye", (2, 3)),
    ("left_eye", (0, 1)),
    ("nose", (4,5))
])


class FaceAligner:

    def __init__(self, num_landmark_pts=68, desiredLeftEye=(0.3,0.3),desiredFaceWidth=160, desiredFaceHeight=160):

        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        if num_landmark_pts==5:
            PREDICTOR_PATH = "shape_predictor_5_face_landmarks.dat"
            self._predictor = shape_predictor(PREDICTOR_PATH)
            self._facial_landmarks_idxs = FACIAL_LANDMARKS_5_IDXS
        elif num_landmark_pts==68:
            PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
            self._predictor = shape_predictor(PREDICTOR_PATH)
            self._facial_landmarks_idxs = FACIAL_LANDMARKS_68_IDXS

        self._desiredLeftEye = desiredLeftEye
        self._desiredFaceWidth = desiredFaceWidth
        self._desiredFaceHeight = desiredFaceHeight

        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self._desiredFaceHeight is None:
            self._desiredFaceHeight = self._desiredFaceWidth

    def _shape_to_np(self,shape, dtype="int"):

        # initialize the list of (x, y)-coordinates
        coords = np.zeros((shape.num_parts, 2), dtype=dtype)

        # loop over all facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, shape.num_parts):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        # return the list of (x, y)-coordinates
        return coords

    def get_alignFaces(self, image, faceRects,color_mode='bgr'):
        """
        align_face

        :param image: image
        :param faceRects: (x1,y1,x2,y2)
        :param color_mode: one of 'rgb','bgr'
        :return: aligned faceImages,landmarks_list
        """
        faceImages=[]
        landmarks_list=[]

        # convert the landmark (x, y)-coordinates to a NumPy array
        if color_mode=='bgr':
            rgb_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image.copy()

        for faceRect in faceRects:
            r = rectangle(*faceRect)
            shape = self._predictor(rgb_image, r)
            shape = self._shape_to_np(shape)

            (lStart, lEnd) = self._facial_landmarks_idxs["left_eye"]
            (rStart, rEnd) = self._facial_landmarks_idxs["right_eye"]
            (nStart, nEnd) = self._facial_landmarks_idxs["nose"]

            leftEyePts = shape[lStart:lEnd]
            rightEyePts = shape[rStart:rEnd]
            nosePts = shape[nStart:nEnd]
            # landmarks_list.append({
            #     'left_eye':
            # })

            # compute the center of mass for each eye
            leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
            rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

            # compute the angle between the eye centroids
            dY = rightEyeCenter[1] - leftEyeCenter[1]
            dX = rightEyeCenter[0] - leftEyeCenter[0]
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
            eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                          (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

            # grab the rotation matrix for rotating and scaling the face
            M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

            # update the translation component of the matrix
            tX = self._desiredFaceWidth * 0.5
            tY = self._desiredFaceHeight * self._desiredLeftEye[1]
            M[0, 2] += (tX - eyesCenter[0])
            M[1, 2] += (tY - eyesCenter[1])

            # apply the affine transformation
            (w, h) = (self._desiredFaceWidth, self._desiredFaceHeight)
            aligned_face = cv2.warpAffine(image, M, (w, h),flags=cv2.INTER_CUBIC)

            # return the aligned face
            faceImages.append(aligned_face)

        return faceImages

    def vis_face_landmarks(self,image,faceRects,colors=None,thickness=2,directShow=False):
        """
        show landmarks

        :param image: image
        :param faceRects: (x1, y1, x2, y2)
        :param colors: will color on landmarks
        :param alpha: alpha blending
        :return: image
        """
        # create two copies of the input image -- one for the
        # overlay and one for the final output image
        overlay = image.copy()
        output = image.copy()

        # if the colors list is None, initialize it with a unique
        # color for each facial landmark region
        if colors is None:
            colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
                (168, 100, 168), (158, 163, 32),(255,100,255),
                (163, 38, 32), (180, 42, 220)]

        for faceRect in faceRects:
            r = rectangle(*faceRect)
            shape = self._predictor(image, r)
            shape = self._shape_to_np(shape)

            # loop over the facial landmark regions individually
            if len(self._facial_landmarks_idxs)==3:
                for i,(j,k) in enumerate(self._facial_landmarks_idxs.values()):
                    pt = tuple(shape[j])
                    cv2.circle(output,pt,thickness,colors[i],-1)

            else:
                for i, name in enumerate(self._facial_landmarks_idxs.keys()):
                    # grab the (x, y)-coordinates associated with the
                    # face landmark
                    (j, k) = self._facial_landmarks_idxs[name]
                    pts = shape[j:k]

                    # check if are supposed to draw the jawline
                    if name == "jaw":
                        # since the jawline is a non-enclosed facial region,
                        # just draw lines between the (x, y)-coordinates
                        for l in range(1, len(pts)):
                            ptA = tuple(pts[l - 1])
                            ptB = tuple(pts[l])
                            cv2.line(overlay, ptA, ptB, colors[i], 2)

                    # otherwise, compute the convex hull of the facial
                    # landmark coordinates points and display it
                    else:
                        hull = cv2.convexHull(pts)
                        cv2.drawContours(overlay, [hull], -1, colors[i], -1)

                # apply the transparent overlay
                cv2.addWeighted(overlay, 0.75, output, 1 - 0.75, 0, output)

        if directShow:
            plt.imshow(output[...,::-1])
            plt.show()
        else:
            # return the output image
            return output


