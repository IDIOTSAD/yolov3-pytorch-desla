



#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2, random, math, copy
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from xycar_msgs.msg import xycar_motor
from std_msgs.msg import String
from yolov3_trt.msg import BoundingBoxes, BoundingBox
import sys
import os
import signal

tm = cv2.TickMeter()
frame = np.empty(shape = [0])
bridge = CvBridge()
pub = None

time_ex , time_now = 0,0
Width = 640
Height = 480
Offset = 380
Gap = 55
speed  = 15
angle = 0
count = 0
prev_speed = 0
prev_lpos = 0
prev_rpos = 0
Purpose_speed = 15

Kp = 0.35
Ki = 0
Kd = 0.15

p_error, d_error, i_error = 0,0,0

obj_id = -1
box_width = 0
box_height = 0
bbox_threshold = 3000

Stopped = False # 정지, 횡단보도 표지판이 있을 때 멈췄다가 5초뒤 출발해야 함.
                # 5초간 정지했으면 이미지에 해당 표지판이 있더라도 무시하고 주행해야함.

def box_callback(data):
    global obj_id, Stopped, curr_box_width, curr_box_height, box_width, box_height, bbox_threshold, box_xmin, box_xmax, box_ymin,box_ymax
    obj_id = -1 # 검출 실패
    for bbox in data.bounding_boxes:
        curr_box_width = bbox.xmax-bbox.xmin
        curr_box_height = bbox.ymax-bbox.ymin
        
        if (curr_box_width*curr_box_height > bbox_threshold) and (curr_box_width*curr_box_height > box_width*box_height):
            obj_id = bbox.id
            box_width = curr_box_width
            box_height = curr_box_height
            obj_id = bbox.id
            box_xmin = bbox.xmin
            box_xmax = bbox.xmax
            box_ymin = bbox.ymin
            box_ymax = bbox.ymax
            
    if Stopped and (obj_id in [2,4]):
        Stopped = False

def isRED(img):
    global box_xmin,box_xmax,box_ymin, box_ymax, y_min, y_max, x_min, x_max
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.inRange(gray,150,255)
    roi = gray[y_min:y_max,x_min:x_max]
    
    count = 0
    for i in roi[roi.shape[0]/2:,roi.shape[1]/2] :
        if count > 20 :
            return True
        if i==255 :
            count += 1
    return False

def PID_control(error) :
    global Kp, Kd, Ki
    global p_error, d_error, i_error

    d_error = error - p_error
    p_error = error
    i_error += error
    return Kp * p_error + Kd * d_error + Ki * i_error

def img_callback(data) :
    global frame
    frame = bridge.imgmsg_to_cv2(data, "bgr8")


# 인식 된 직선을 왼,오른 차선으로 구분
def divide_left_right(lines):
    min_slope_threshold = 0
    max_slope_threshold = 10

    slopes = []
    new_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if x2 - x1 == 0:
            slope = 0
        else:
            slope = float(y2 - y1) / float(x2 - x1)

        #차선이 아닌 직선 필터링 (ex.수평선)
        if abs(slope) > min_slope_threshold and abs(slope) < max_slope_threshold:
            slopes.append(slope)
            new_lines.append(line[0])

    left_lines = []
    right_lines = []
    
    for i in range(len(slopes)):
        slope = slopes[i]
        x1, y1, x2, y2 = new_lines[i]

        if (slope < 0) and (x2 < Width / 2):
            left_lines.append([new_lines[i].tolist()])
        elif (slope > 0) and (x1 > Width / 2):
            right_lines.append([new_lines[i].tolist()])

    return left_lines, right_lines
def isRED(img):
    global box_xmin,box_xmax,box_ymin, box_ymax, y_min, y_max, x_min, x_max
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.inRange(gray,150,255)
    roi = gray[y_min:y_max,x_min:x_max]
    
    count = 0
    for i in roi[roi.shape[0]/2:,roi.shape[1]/2] :
        if count > 20 :
            return True
        if i==255 :
            count += 1
    return False
# 차선의 대표 직선을 찾고, 기울기, 절편을 반환
def get_line_params(lines):
    # sum of x, y, m
    x_sum = 0.0
    y_sum = 0.0
    m_sum = 0.0

    num = len(lines)
    if num == 0:
        return 0, 0

    for line in lines:
        x1, y1, x2, y2 = line[0]

        x_sum += x1 + x2
        y_sum += y1 + y2
        m_sum += float(y2 - y1) / float(x2 - x1)

    x_avg = x_sum / (num * 2)
    y_avg = y_sum / (num * 2)
    m = m_sum / num
    b = y_avg - m * x_avg

    return m, b

# get lpos, rpos
def get_line_pos(img, lines, left=False, right=False):

    m, b = get_line_params(lines)

#차선을 못찾았을 경우 왼쪽은 -1, 오른쪽은 641로함.
    if m == 0 and b == 0:
        if left:
            pos = -1
        if right:
            pos = Width + 1
    else:
        y = Gap / 2
        pos = (y - b) / m

    return int(pos)

# 영상 전처리 및 차선 찾기
def process_image(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    norm = cv2.normalize(gray,None,0,255,cv2.NORM_MINMAX)
    blur_gray = cv2.GaussianBlur(norm, (5, 5), 0)
        
    edge_img = cv2.Canny(np.uint8(blur_gray), 140, 70)

    roi = edge_img[Offset: Offset + Gap, 0: Width]
        
    all_lines = cv2.HoughLinesP(roi, 1, math.pi / 180, 30, 30, 10)

    if all_lines is None:
        return 0, 640
    left_lines, right_lines = divide_left_right(all_lines)

    lpos = get_line_pos(frame, left_lines, left=True)
    rpos = get_line_pos(frame, right_lines, right=True)
    return lpos, rpos

#publish xycar_motor msg
def drive(Angle, Speed):
    
    msg = xycar_motor()
        
    if Angle >= 50:
        Angle = 50
    if Angle <= -50:
        Angle = -50
            
    msg.angle = Angle
        
    #제어 안정성을 높이기 위해 천천히 가속
    if Speed != -10:
        if Speed < Purpose_speed:
            Speed += 1
        else:
            Speed = Purpose_speed
    
    #곡선 주행시 차선을 이탈하지 않도록 각도에 비례해 감속
    if abs(Angle) > 20 and Speed != -10:
        msg.speed = (Speed -(0.17 * abs(Angle)))
    else:
        msg.speed = (Speed -(0.1 * abs(Angle)))

    pub.publish(msg)

# 차선 잃을시 멈췄다 후진

####################
### 5초간 정지 함수 ###
####################

# TickMeter()를 이용하여 시간을 측정
# 정지를 시작한 시간을 time_ex에 저장
# time_now 에 현재 시간 저장
# 두 시간의 차이가 5초보다 커질때까지 계속 반복

def stop():
    global time_ex, time_now, Stopped
    
    drive(angle,0)
    if time_ex == 0:
        time_ex = time_now
                
    while time_now - time_ex < 5:
        time_now = tm.getTimeSec()
    
    time_ex = 0
    Stopped = True


       
def start():
    global tm
    global Stopped
    global Width, Height, cap, prev_l, prev_r
    global pub, frame, count, angle, speed
    global time_ex, time_now
    
    rospy.init_node("auto_drive")
    pub = rospy.Publisher('xycar_motor',xycar_motor,queue_size=1)

    image_sub = rospy.Subscriber("/usb_cam/image_raw",Image,img_callback)
    bbox_sub = rospy.Subscriber("/yolov3_trt_ros/detections", BoundingBoxes, box_callback, queue_size=1)
    rospy.sleep(2)
    
    tm.reset()
    tm.start()
    time_ex, time_now = 0,0
    
    drive(0,0)
        
    while True:
        while not frame.size == (Width*Height*3) :
            continue
        
        # 시간 측정
        tm.stop()
        time_now = tm.getTimeSec()
        tm.start()
        
        #신호등 & 정지 & 횡단보도
        if (obj_id == 3 and isRED(frame)) or ((not Stopped) and (obj_id in [2,4])):
            stop()
            continue
        
        lpos, rpos = process_image(frame)
        
        # 차선 1개라도 검출 못 할시 표지판 인식
        if rpos == Width +1 or lpos == -1:
            
            # 표지판 x or ignore일시
            if obj_id == -1:
                lpos = prev_l
                rpos = prev_r
            
            # 좌회전
            elif obj_id == 0:
                lpos = prev_l
                rpos = lpos + 440
            
            # 우회전
            elif obj_id == 1:
                lpos = rpos - 440
                rpos = prev_r
            
        
        #도로 내부에 다른 물체때문에 오검출 되었을 경우 처리
        else:
            if rpos - lpos < 430:
                if abs(rpos-prev_rpos) < 100:
                    lpos = rpos - 450
                if abs(lpos-prev_lpos) < 100:
                    rpos = lpos + 450
        
        center = (lpos + rpos) / 2
        error = (center - (Width / 2))

        angle = PID_control(error)

        drive(angle , speed)
        prev_l = lpos
        prev_r = rpos
    
    rospy.spin()
    
if __name__ == '__main__':
    start()
