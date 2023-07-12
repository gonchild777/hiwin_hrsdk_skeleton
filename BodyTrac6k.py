import cv2
import mediapipe as mp
from pyparsing import one_of
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

import socket
import time
from sympy import Abs, capture, false, true

import numpy as np
from sympy import degree
import math

UDP_IP = '192.168.0.205'
UDP_PORT = 5000
socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

def justremap(input, input_domain=(0,1), output_domain=(0,1)):
    return (input - input_domain[0])*(output_domain[1]-output_domain[0])/(input_domain[1]-input_domain[0]) + output_domain[0]

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))


a_one_last = 0
a_one_f = 0
a_two_last = 0
a_two_f = 0
a_three_last = 0
a_three_f = 0
a_five_last = 0
a_five_f = 0






cap = cv2.VideoCapture("ABAB.mp4")
with mp_pose.Pose(
   static_image_mode=True,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.9) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    if(results.pose_landmarks != None):  
        keypoints = ''
        keypoints_list = []
        for data_point in results.pose_landmarks.landmark:

            x_ = justremap(data_point.x,(0,1),(1,0))
            y_ = justremap(data_point.y,(0,1),(1,0))
            point_list = []
            point_list.append(round(float(x_),3))
            point_list.append(round(float(y_),3))
            point_list.append(round(float(data_point.z),3))
            point_list.append(round(float(data_point.visibility),3))
            keypoints_list.append(point_list)
        send_switch = 0    
#####################################################
        
        #第一軸
        shoulder_vec = (keypoints_list[12][0]-keypoints_list[11][0],
        keypoints_list[12][1]-keypoints_list[11][1],
        keypoints_list[12][2]-keypoints_list[11][2])
        vec_ = (shoulder_vec[0]+10,0,0)
        #如果一邊比較深,乘-1 兩位數小數點
        if keypoints_list[12][2]>keypoints_list[11][2]:
            shoulder_angle_ = round(math.degrees(angle(vec_,shoulder_vec))*-1,2)
        else:
            shoulder_angle_ = round(math.degrees(angle(vec_,shoulder_vec)),2)
        #從-90~90 到' 140~-140  
        if(shoulder_angle_>-90 and shoulder_angle_<90):
            a_one = round(justremap(shoulder_angle_,(90,-90),(140,-140)),2)
            if abs(abs(a_one)-abs(a_one_last)) <= 5:
               pass
            else:
               a_one_f = a_one
               send_switch += 1
            a_one_last = a_one

        #第二軸
        
        waist_vec = (keypoints_list[23][0]-keypoints_list[11][0],
        keypoints_list[23][1]-keypoints_list[11][1],
        keypoints_list[23][2]-keypoints_list[11][2])
        foot_vec = (keypoints_list[23][0]-keypoints_list[25][0],
        keypoints_list[23][1]-keypoints_list[25][1],
        keypoints_list[23][2]-keypoints_list[25][2])  
          
        if keypoints_list[23][1]>keypoints_list[11][1]:
               foot_angle_ = round(math.degrees(angle(foot_vec,waist_vec)),2)
        else:
               foot_angle_ = round(math.degrees(angle(foot_vec,waist_vec)),2)
        print(foot_angle_)
        if foot_angle_>-180 and foot_angle_<180:
            a_two = round(justremap(foot_angle_,(180,100),(-130,-20)),2)
            if abs(abs(a_two)-abs(a_two_last)) <= 6:
                pass
            else:
                a_two_f = a_two
                send_switch += 1
            a_two_last = a_two   
        if a_two_f== 0:
           a_two_f = 90
        #第三軸
        
        hand_vec = (keypoints_list[11][0]-keypoints_list[13][0],
        keypoints_list[11][1]-keypoints_list[13][1],
        keypoints_list[11][2]-keypoints_list[13][2])
        body_vec = (keypoints_list[11][0]-keypoints_list[23][0],
        keypoints_list[11][1]-keypoints_list[23][1],
        keypoints_list[11][2]-keypoints_list[23][2]) 
        
        if keypoints_list[11][1]>keypoints_list[13][1]:
               hand_angle_ = round(math.degrees(angle(hand_vec,body_vec)),2)
        else:
                hand_angle_ = round(math.degrees(angle(hand_vec,body_vec)),2)
        print(hand_angle_)
        if  hand_angle_>-180 and  hand_angle_<180:
            a_three = round(justremap(hand_angle_,(180,40),(-20,90)),2)
            if abs(abs(a_three)-abs(a_three_last)) <= 2:
                pass
            else:
                a_three_f = a_three
                send_switch += 1
            a_three_last = a_three

        if a_three_f == 0:
           a_three_f = 10

        #第五軸
        
        elbow_vec = (keypoints_list[13][0]-keypoints_list[11][0],
        keypoints_list[13][1]-keypoints_list[11][1],
        keypoints_list[13][2]-keypoints_list[11][2])
        elbow_vec2 = (keypoints_list[13][0]-keypoints_list[15][0],
        keypoints_list[13][1]-keypoints_list[15][1],
        keypoints_list[13][2]-keypoints_list[15][2])  
          
        if keypoints_list[13][1]>keypoints_list[11][1]:
               elbow_angle_ = round(math.degrees(angle(elbow_vec,elbow_vec2)),2)
        else:
               elbow_angle_ = round(math.degrees(angle(elbow_vec,elbow_vec2)),2)
        print(elbow_angle_)
        if elbow_angle_>-180 and elbow_angle_<180:
            a_five = round(justremap(elbow_angle_,(180,40),(-40,40)),2)
            if abs(abs(a_five)-abs(a_five_last)) <= 2:
                pass
            else:
                a_five_f = a_five
                send_switch += 1
            a_five_last = a_five   
        if a_five_f == 0:
           a_five_f = 0











        send_ = str(a_one_f)+'?'+str(a_two_f)+'?'+str(a_three_f)+'?'+str(a_five_f)
        print(send_)
        if send_switch>0:
           socket.sendto((str(send_)).encode(),(UDP_IP,UDP_PORT))
        #time.sleep(0.1)
#########################################################################
        


        
        #point_ = ""
        #for i in range(len(keypoints_list)):
        #  point_+=str(keypoints_list[i][0]) + "," + str(keypoints_list[i][2]) + "," + str(keypoints_list[i][1]) + "," +str(keypoints_list[i][3]) + "!"
        
        #計算兩向量的夾角，利用numpy






        #socket.sendto((str(point_)).encode(),(UDP_IP,UDP_PORT))
        #time.sleep(0.1)


    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
cap.release()