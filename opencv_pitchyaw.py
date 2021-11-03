#!/usr/bin/env python3
"""
OpenCV-based camera calibration by extracting pitch and yaw angles from
lane lines and unprojecting the vanishing point in image coordinates to
camera coordinates. Use like:

$ python3 ./opencv_pitchyaw.py 0.hevc > 0.txt
"""
import cv2
import sys
import numpy as np

FOCAL_LENGTH = 910.0
FRAME_SIZE   = (1164, 874)
K = np.array([ # camera intrinsics
  [FOCAL_LENGTH,  0.0,  float(FRAME_SIZE[0])/2],
  [0.0,  FOCAL_LENGTH,  float(FRAME_SIZE[1])/2],
  [0.0,  0.0,                              1.0]])
K_inv  = np.linalg.inv(K)
CENTER = np.array(FRAME_SIZE)//2
TRAPEZOID = [np.array([ # cuts out sides and top part of the frame
    (0, 8*FRAME_SIZE[1]//10),
    (FRAME_SIZE[0], 8*FRAME_SIZE[1]//10),
    (7*FRAME_SIZE[0]//10, 15*FRAME_SIZE[1]//32),
    (3*FRAME_SIZE[0]//10, 15*FRAME_SIZE[1]//32)], dtype='int32')]

def get_line(rho, theta):
    a  = np.cos(theta)
    b  = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 10000*(-b))
    y1 = int(y0 + 10000*(a))
    x2 = int(x0 - 10000*(-b))
    y2 = int(y0 - 10000*(a))
    return np.array([x1, y1]), np.array([x2, y2])

def get_intersect(p1, p2, q1, q2):
    s = np.vstack([p1, p2, q1, q2])
    h = np.hstack((s, np.ones((4, 1))))
    l1 = np.cross(h[0], h[1])
    l2 = np.cross(h[2], h[3])
    x, y, z = np.cross(l1, l2) # intersection
    return (x/z, y/z) if z != 0 else (float('inf'), float('inf'))

def draw_line(img, rho, theta):
    (x1,y1), (x2,y2) = get_line(rho, theta)
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0), 3)

def roi(img, vertices=TRAPEZOID, fill=255):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        ignore_mask_color = (fill,) * img.shape[2]
    else:
        ignore_mask_color = fill
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    return cv2.bitwise_and(img, mask)

def find_lines(edges, is_left=True, confidence=100, img=None):
    result = []
    lines = cv2.HoughLines(edges, 1, np.pi/180, confidence)
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            p1, p2 = get_line(rho, theta)
            dist_from_center = np.abs(np.cross(p2-p1, p1-CENTER) / np.linalg.norm(p2-p1))
            if dist_from_center < 30:
                if is_left and 40 < np.degrees(theta) < 60:
                    result += [line[0]]
                elif not is_left and 115 < np.degrees(theta) < 135:
                    result += [line[0]]
    return result

def get_flow(prev, curr):
    flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY),
                                        cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY),
                                        None, pyr_scale=0.25, levels=1, winsize=10,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=None)
    horz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)
    return horz.astype('uint8')

def filter_bg(flow):
    thresh = cv2.cvtColor(flow, cv2.COLOR_GRAY2RGB)
    cnts, vals = np.histogram(np.array(thresh), 255)
    cidx     = np.argmax(cnts)
    bgcol    = vals[cidx]
    above = thresh  <= bgcol+2
    below = bgcol-2 <= thresh
    thresh[above * below] = 255
    return cv2.cvtColor(thresh, cv2.COLOR_RGB2GRAY)

def crop_LR(edges):
    left_edges = edges.copy()
    left_edges[:2*FRAME_SIZE[1]//5, :]   = 0    
    left_edges[:, 3*FRAME_SIZE[0]//5:]   = 0
    left_edges[7*FRAME_SIZE[1]//10:, :]  = 0
    right_edges = edges.copy()
    right_edges[:2*FRAME_SIZE[1]//5, :]  = 0
    right_edges[:, :2*FRAME_SIZE[0]//5]  = 0
    right_edges[7*FRAME_SIZE[1]//10:, :] = 0
    return left_edges, right_edges

def calc_pitch_yaw(left_lines, right_lines, pitches, yaws, img=None):
    if 0 < len(left_lines) and 0 < len(right_lines):
        left_rho, left_theta   = np.median(left_lines, 0)
        right_rho, right_theta = np.median(right_lines, 0)
        if not img is None:
            draw_line(img, left_rho, left_theta)
            draw_line(img, right_rho, right_theta)
        a1, a2 = get_line(left_rho, left_theta)
        b1, b2 = get_line(right_rho, right_theta)
        vp = get_intersect(a1, a2, b1, b2)
        _, pitch, yaw = rpy_from_vp(vp)
        pitches += [abs(pitch)]
        yaws    += [abs(yaw)]
    return np.mean(pitches), np.mean(yaws)

def unproject(img_pts, K_inv=K_inv):
    img_pts = np.array(img_pts)
    input_shape = img_pts.shape
    img_pts = np.atleast_2d(img_pts)
    img_pts = np.hstack((img_pts, np.ones((img_pts.shape[0], 1))))
    cam_pts = img_pts.dot(K_inv.T)
    cam_pts[(img_pts < 0).any(axis=1)] = np.nan
    return cam_pts[:, :2].reshape(input_shape)

def rpy_from_vp(vp):
    vp_cam = unproject(vp)
    yaw_calib = np.arctan(vp_cam[0])
    pitch_calib = np.arctan(-vp_cam[1])*np.cos(yaw_calib)
    return 0, pitch_calib, yaw_calib # roll assumed 0

if __name__ == "__main__":
    pitches = [0.]
    yaws    = [0.]
    cap = cv2.VideoCapture(sys.argv[1])
    _, prev = cap.read()
    print(0., 0.) # first frame
    while True:
        ret, frame = cap.read()
        if not ret or cv2.waitKey(1)==27:
            break
        flow   = get_flow(prev, frame)
        thresh = filter_bg(flow)
        prev   = frame.copy()
        L, R   = crop_LR(cv2.Canny(roi(thresh), 100, 255))
        left_lines  = find_lines(L, is_left=True,  confidence=50, img=frame)
        right_lines = find_lines(R, is_left=False, confidence=50, img=frame)
        pitch, yaw  = calc_pitch_yaw(left_lines, right_lines, pitches, yaws, frame)
        cv2.imshow('frame', frame)
        print(pitch, yaw)
    print(np.mean(pitches), np.mean(yaws)) # last frame
    cv2.destroyAllWindows()
    cap.release()
