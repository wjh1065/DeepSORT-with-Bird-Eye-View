import cv2
import numpy as np



def get_transformed_points(boxes, prespective_transform):
    bottom_points = []
    # print('boxes : ',boxes)
    for box in boxes:
        # print('1. box : ',box)
        pnts = np.array([[[int((box[0] + box[2]) * 0.5), int((box[1] + box[3]) * 0.5)]]], dtype="float32")
        # print('2. pnts : ', pnts)
        bd_pnt = cv2.perspectiveTransform(pnts, prespective_transform)[0][0]
        # print('prespective_transform : ', prespective_transform)
        pnt = [int(bd_pnt[0]), int(bd_pnt[1])]
        # print('3. pnt : ',pnt)
        bottom_points.append(pnt)

    return bottom_points


def get_transformed_points2(boxes, prespective_transform):
    bottom_points = []
    # print('boxes : ',boxes)
    for box in boxes:
        # print('1. box : ',box)
        pnts = np.array([[[int((box[0] + box[2]) * 0.5), int((box[1] + box[3]) * 0.5)]]], dtype="float32")
        # print('2. pnts : ', pnts)
        bd_pnt = cv2.perspectiveTransform(pnts, prespective_transform)[0][0]
        # print('prespective_transform : ', prespective_transform)
        pnt = [int(bd_pnt[0]), int(bd_pnt[1])]
        # print('3. pnt : ',pnt)
        bottom_points.append(pnt)

    return bottom_points

def get_transformed_points_1_mul(boxes, prespective_transform):
    bottom_points = []
    # print('boxes : ',boxes)
    for box in boxes:
        # print('1. box : ',box)
        pnts = np.array([[[int((box[0] + box[2]) * 0.5), int((box[1] + box[3]) * 0.5)]]], dtype="float32")
        # print('2. pnts : ', pnts)
        bd_pnt = cv2.perspectiveTransform(pnts, prespective_transform)[0][0]
        # print('prespective_transform : ', prespective_transform)
        pnt = [int(bd_pnt[0])*2, int(bd_pnt[1])]
        # print('3. pnt : ',pnt)
        bottom_points.append(pnt)

    return bottom_points


def bird_eye_view(bottom_points):
    w = 720
    h = 1280

    red = (0, 0, 255)
    green = (0, 255, 0)
    yellow = (0, 255, 255)
    white = (200, 200, 200)

    # scale_w = 0.3125
    # scale_h = 1.0
    scale_w, scale_h = get_scale(w, h)
    blank_image = np.zeros((640,360, 3), np.uint8)
    blank_image[:] = white

    for i in bottom_points:
        blank_image = cv2.circle(blank_image, (int(i[0] * scale_w), int(i[1] * scale_h)), 5, green, 10)
    return blank_image


def bird_eye_view2(bottom_points):
    w = 1280
    h = 720

    red = (0, 0, 255)
    green = (0, 255, 0)
    yellow = (0, 255, 255)
    white = (200, 200, 200)

    # scale_w = 0.3125
    # scale_h = 1.0
    scale_w, scale_h = get_scale2(w, h)
    blank_image = np.zeros((360,640, 3), np.uint8)
    blank_image[:] = white

    for i in bottom_points:
        blank_image = cv2.circle(blank_image, (int(i[0] * scale_w), int(i[1] * scale_h)), 5, green, 10)
    return blank_image


def bird_eye_view_all(bottom_points_all):
    w = 1280
    h = 1280

    red = (0, 0, 255)
    green = (0, 255, 0)
    yellow = (0, 255, 255)
    white = (200, 200, 200)

    # scale_w = 0.3125
    # scale_h = 1.0
    scale_w, scale_h = get_scale_all(w, h)
    blank_image = np.zeros((640,640, 3), np.uint8)
    blank_image[:] = white

    blank_image = cv2.line(blank_image, (0, 0), (640, 0), red, 5)
    blank_image = cv2.line(blank_image, (640, 0), (640, 640), red, 5)
    blank_image = cv2.line(blank_image, (640, 640), (320, 640), red, 5)
    blank_image = cv2.line(blank_image, (320, 640), (320, 320), red, 5)
    blank_image = cv2.line(blank_image, (320, 320), (0, 320), red, 5)


    blank_image = cv2.line(blank_image, (320, 0), (640, 0), green, 3)
    blank_image = cv2.line(blank_image, (640, 0), (640, 320), green, 3)
    blank_image = cv2.line(blank_image, (640, 320), (320, 320), green, 3)
    blank_image = cv2.line(blank_image, (320, 320), (320, 0), green, 3)

    for i in bottom_points_all:
        blank_image = cv2.circle(blank_image, (int(i[0] * scale_w), int(i[1] * scale_h)), 5, green, 10)
    return blank_image



def get_scale(W, H):
    dis_w = 360
    dis_h = 640

    return float(dis_w / W), float(dis_h / H)

def get_scale2(W, H):
    dis_w = 640
    dis_h = 360

    return float(dis_w / W), float(dis_h / H)

def get_scale_all(W, H):
    dis_w = 640
    dis_h = 640

    return float(dis_w / W), float(dis_h / H)




