#!/usr/bin/env python
#----------------------------------------------------------------------------
#
# Created By: Victor Zhi Heung Ngo
# Email: psxvn1@nottingham.ac.uk
# Created Date: Feburary 2022
#
# ---------------------------------------------------------------------------
""" Details about the module and for what purpose it was built for"""
# ---------------------------------------------------------------------------
#
# Tested on Ubuntu 18.04, kernel 5.11, ROS Melodic
#
# Where specificed using the '#---' annotation, the author of the original 
# work is referenced.
#
# ---------------------------------------------------------------------------


import cv2
import rospy
import imutils

from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class QRCode():
    def __init__(self):
        self.rate = rospy.Rate(60)
        self.debug = rospy.get_param('~debug')
        self.A_001 = cv2.imread('/home/victor/spot_ws/src/spot_ros_autonomy/images/A_001.png')
        self.B_001 = cv2.imread('/home/victor/spot_ws/src/spot_ros_autonomy/images/B_001.png')
        self.C_001 = cv2.imread('/home/victor/spot_ws/src/spot_ros_autonomy/images/C_001.png')
        self.D_001 = cv2.imread('/home/victor/spot_ws/src/spot_ros_autonomy/images/D_001.png')

        self.front_camera_img_sub = rospy.Subscriber('/camera1_IR/color/image_rect_color', Image, self.camera_img_callback)
        self.front_image_pub = rospy.Publisher('/camera1_IR/color/image_rect_color/image_overlayed', Image, queue_size=1)


    def camera_img_callback(self, msg):
        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(msg, 'bgr8')

        alpha = 0

        self.A_001_resized_img = imutils.resize(self.A_001, width=int((image.shape[1] * 0.22)))
        self.B_001_resized_img = imutils.resize(self.B_001, width=int((image.shape[1] * 0.22)))
        self.C_001_resized_img = imutils.resize(self.C_001, width=int((image.shape[1] * 0.22)))
        self.D_001_resized_img = imutils.resize(self.D_001, width=int((image.shape[1] * 0.22)))

        qr_code_width = self.A_001_resized_img.shape[1]
        qr_code_height = self.A_001_resized_img.shape[0]

        image_width = image.shape[1]
        image_height = image.shape[0]

        inner_width_to_edge = (image_width - qr_code_width)
        inner_height_to_edge = (image_height - qr_code_height)

        added_image_A = cv2.addWeighted(image[0:qr_code_height,0:qr_code_width,:],alpha,self.A_001_resized_img[0:qr_code_height,0:qr_code_width,:],1-alpha,0)
        added_image_B = cv2.addWeighted(image[0:qr_code_height,inner_width_to_edge:image_width,:],alpha,self.B_001_resized_img[0:qr_code_height,0:qr_code_width,:],1-alpha,0)
        added_image_C = cv2.addWeighted(image[inner_height_to_edge:image_height,0:qr_code_width,:],alpha,self.C_001_resized_img[0:qr_code_height,0:qr_code_width,:],1-alpha,0)
        added_image_D = cv2.addWeighted(image[inner_height_to_edge:image_height,inner_width_to_edge:image_width,:],alpha,self.D_001_resized_img[0:qr_code_height,0:qr_code_width,:],1-alpha,0)
        
        image[0:qr_code_height,0:qr_code_width] = added_image_A
        image[0:qr_code_height,inner_width_to_edge:image_width] = added_image_B
        image[inner_height_to_edge:image_height,0:qr_code_width] = added_image_C
        image[inner_height_to_edge:image_height,inner_width_to_edge:image_width] = added_image_D

        img_msg = bridge.cv2_to_imgmsg(image, 'bgr8')
        self.front_image_pub.publish(img_msg)
     

if __name__ == "__main__":
    rospy.init_node('qr_code_overlay_node')
    while not rospy.is_shutdown():
        try:
            QRCode()
            rospy.spin()
        except rospy.ROSInterruptException:
            pass
       
