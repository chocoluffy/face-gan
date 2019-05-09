import numpy as np
import cv2
from PIL import Image
import math
from collections import OrderedDict
from imutils import face_utils
import imutils
import dlib
import sys

def enlarge_diagonal(x1, y1, x2, y2):
	r = 0.30
	x1_ = x1 - (x2-x1)*r
	x2_ = x2 + (x2-x1)*r

	y1_ = y1 - (y2-y1)*r
	y2_ = y2 + (y2-y1)*r

	return x1_, y1_, x2_, y2_


def resize_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height = image.shape[0]
    width = image.shape[1]
    shape_min = min(height, width)
    width_new = int(width/shape_min*257)
    image = imutils.resize(image, width=width_new)

    if (height>=width):
        image = image[0:256, 0:256]
    else:
        padding = int((width-height)/2)
        image = image[0:256, padding:padding+256]

    image2 = Image.fromarray(image)
    image2.save(path)

def eye_detector(shape_predictor, image_dir):
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)
    
    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(image_dir)
    original_shape = image.shape
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    left_eye = []
    right_eye = []
    
    # loop over the face detections
    for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the landmark (x, y)-coordinates to a NumPy array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
    
            # loop over the face parts individually
            for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                # clone the original image so we can draw on it, then
                # display the name of the face part on the image
                clone = image.copy()
    
                # loop over the subset of facial landmarks, drawing the
                # specific face part
                for (x, y) in shape[i:j]:
                    cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
                    if name == 'left_eye':
                        left_eye.append((x, y))
                    if name == 'right_eye':
                        right_eye.append((x, y))
                        
    return left_eye, right_eye, image.shape, original_shape

def get_bounding_box_helper(x1, y1, x2, y2, h1, h2):
    delta_x1 = None
    delta_y1 = None
    delta_x2 = None
    delta_y2 = None

    if y1 != y2:
        k = (x2-x1)/(y1-y2)
        delta_x1 = h1/math.sqrt(k*k+1)
        delta_y1 = k*h1/math.sqrt(k*k+1)
        
        delta_x2 = h2/math.sqrt(k*k+1)
        delta_y2 = k*h2/math.sqrt(k*k+1)
    else:
        delta_x1 = 0
        delta_y1 = h1

        delta_x2 = 0
        delta_y2 = h2
    
    point1 = [x1+delta_x1, y1+delta_y1]
    point2 = [x1-delta_x2, y1-delta_y2]
    point3 = [x2+delta_x1, y2+delta_y1]
    point4 = [x2-delta_x2, y2-delta_y2]
    
    return np.array([point1, point2, point4, point3], np.int32)
    
def get_bounding_box(x1, y1, x2, y2):
    dist = math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
    h1_ratio = 0.7
    h2_ratio = 0.7
    
    return get_bounding_box_helper(x1, y1, x2, y2, dist*h1_ratio, dist*h2_ratio)

def make_mask(x1, y1, x2, y2, image_shape):
    points = get_bounding_box(x1, y1, x2, y2)
    mask = np.zeros(image_shape, dtype = "uint8")
    cv2.fillConvexPoly(mask, points,(255, 255, 255))
    return mask

if __name__ == "__main__":
    image_path = sys.argv[2]
    mask_path = sys.argv[4]
    resize_image(image_path)
    left, right, shape, original_shape = eye_detector("./model/shape_predictor_68_face_landmarks.dat", image_path)

    if (len(left) != 0 or len(right) != 0):
        left_mask = make_mask(*enlarge_diagonal(left[0][0], left[0][1], left[3][0], left[3][1]), shape)
        right_mask = make_mask(*enlarge_diagonal(right[0][0], right[0][1], right[3][0], right[3][1]), shape)
        mask = left_mask + right_mask
        mask_image = Image.fromarray(mask).resize([original_shape[1], original_shape[0]])
        mask_image.save(mask_path)
