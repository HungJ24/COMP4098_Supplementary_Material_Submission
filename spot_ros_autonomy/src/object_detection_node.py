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


import tf
import numpy as np
import rospy
import ros_numpy.point_cloud2 as rnpy
import scipy.spatial as scpy

from cv_bridge import CvBridge
from pyzbar import pyzbar
from darknet_ros_msgs.msg import BoundingBoxes
from geometry_msgs.msg import PointStamped
from image_geometry import PinholeCameraModel
from std_msgs.msg import Int8
from sensor_msgs.msg import Image, CameraInfo, PointCloud2

class ObjectDetect:
    def __init__(self):
        self.rate = rospy.Rate(20)
        self.debug = rospy.get_param('~debug')
        self.frame_id = rospy.get_param('~frame_id', 'front_camera_link_optical')
        self.bridge = CvBridge()

        self.qr_predefined_list = ['A_001','B_001','C_001','D_001']


        self.attention_duration = 0
        self.distracted_duration = 0
        self.total_time = 0
        self.attention_rate = 2
        self.intention_goal_list = []
        self.xyz_array = []
        self.per_boxes = []
        self.screen_calibrated = False

        self.front_camera_info_sub = rospy.Subscriber('/camera1_IR/depth/camera_info', CameraInfo, self.camera_info_callback)
        self.pointcloud_sub = rospy.Subscriber('/camera1_IR/depth/points', PointCloud2, self.pointcloud_callback)
        self.tobii_img_sub = rospy.Subscriber('/tobii_camera/image_converted', Image, self.tobii_img_callback)
        self.front_img_sub = rospy.Subscriber('/camera1_IR/color/image_rect_color', Image, self.camera_img_callback)
        self.front_box_sub = rospy.Subscriber('/front/darknet_ros/bounding_boxes', BoundingBoxes, self.front_box_callback)
        self.gaze_sub = rospy.Subscriber('/tobii_gaze', PointStamped, self.gaze_point_callback)

        self.intention_goal_pub = rospy.Publisher('/object_coordinates/3d_coordinates', PointStamped, queue_size= 5)
        self.attention_pub = rospy.Publisher('/attention_score', Int8, queue_size=5)

        self.listener = tf.TransformListener()


    def camera_info_callback(self, msg):
        self.front_camera_header_frame_id = self.frame_id
        self.front_camera_info = PinholeCameraModel()
        self.front_camera_info.fromCameraInfo(msg)

        if self.debug == True:
            rospy.loginfo("[DEBUG] Yolo Node: Camera info obtained")

    
    def camera_img_callback(self, msg):
        front_img =self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.frame_1_width = front_img.shape[1]
        self.frame_1_height = front_img.shape[0]

        
    def pointcloud_callback(self, msg):
        #Get the pointcloud into an array
        self.xyz_array = rnpy.get_xyz_points(rnpy.pointcloud2_to_array(msg), remove_nans=True, dtype=np.float32)

        self.eval_looking_at_screen()

    
    def tobii_img_callback(self, msg):
        tobii_img = self.bridge.imgmsg_to_cv2(msg, 'mono8')
        detected_qr_codes = pyzbar.decode(tobii_img)
        
        self.qr_codes(detected_qr_codes)
    
    
    def gaze_point_callback(self, msg):
        self.gaze_point_x = msg.point.x 
        self.gaze_point_y = msg.point.y
        self.gp = (self.gaze_point_x, self.gaze_point_y)

        if self.debug == True:
            rospy.loginfo("[DEBUG] Yolo Node: Gaze point recieved")
            rospy.loginfo(self.gp)

    
    def front_box_callback(self, msg):
        boxes = msg.bounding_boxes
        if boxes:
            for box in boxes:
                if box.Class == "person": 
                    xmin = box.xmin
                    xmax = box.xmax
                    ymin = box.ymin
                    ymax = box.ymax
                    
                    # Convert bounding box coordinates from the camera frame to 
                    # tobii frame, using the ratio between the camera frame and
                    # the camera frame as seen from the tobii frame.
                    if self.frame_to_frame_ratio_height > 0 and self.frame_to_frame_ratio_width > 0:
                        box_xymin = self.pixel_coordinate_conversion(xmin, ymin, self.frame_to_frame_ratio_width, self.frame_to_frame_ratio_height, "real_world")
                        box_xymax = self.pixel_coordinate_conversion(xmax, ymax, self.frame_to_frame_ratio_width, self.frame_to_frame_ratio_height, "real_world")

                        # perm_1 = ['A_001','B_001','C_001']
                        # perm_2 = ['B_001','C_001','D_001']
                        # perm_3 = ['A_001','C_001','D_001']   
                        # perm_4 = ['A_001','B_001','D_001']
                        if self.screen_calibration_type == "perm_1":
                            tobii_box_xmin, tobii_box_ymin = box_xymin[0] + self.top_left_screen[0], box_xymin[1] + self.top_left_screen[1]
                            tobii_box_xmax, tobii_box_ymax = box_xymax[0] + self.top_left_screen[0], box_xymax[1] + self.top_left_screen[1]
                        if self.screen_calibration_type == "perm_2":
                            tobii_box_xmin, tobii_box_ymin = box_xymin[0] + self.bottom_left_screen[0], box_xymin[1] + self.top_right_screen[1]
                            tobii_box_xmax, tobii_box_ymax = box_xymax[0] + self.bottom_left_screen[0], box_xymax[1] + self.top_right_screen[1]
                        if self.screen_calibration_type == "perm_3":
                            tobii_box_xmin, tobii_box_ymin = box_xymin[0] + self.top_left_screen[0], box_xymin[1] + self.top_left_screen[1]
                            tobii_box_xmax, tobii_box_ymax = box_xymax[0] + self.top_left_screen[0], box_xymax[1] + self.top_left_screen[1]
                        if self.screen_calibration_type == "perm_4":
                            tobii_box_xmin, tobii_box_ymin = box_xymin[0] + self.top_left_screen[0], box_xymin[1] + self.top_left_screen[1]
                            tobii_box_xmax, tobii_box_ymax = box_xymax[0] + self.top_left_screen[0], box_xymax[1] + self.top_left_screen[1]
                        if self.screen_calibration_type == "perm_all":
                            tobii_box_xmin, tobii_box_ymin = box_xymin[0] + self.top_left_screen[0], box_xymin[1] + self.top_left_screen[1]
                            tobii_box_xmax, tobii_box_ymax = box_xymax[0] + self.top_left_screen[0], box_xymax[1] + self.top_left_screen[1]

                        rospy.loginfo("box_xymin " + str(box_xymin))
                        rospy.loginfo("box_xymax " + str(box_xymax))
                        rospy.loginfo("tobii_xymin " + str(tobii_box_xmin) +  " " + str(tobii_box_ymin))
                        rospy.loginfo("tobii_xymax " + str(tobii_box_xmax) +  " " + str(tobii_box_ymax))
                        rospy.loginfo("gp " + str(self.gp))
                        fixation_duration = 0
                        if self.gp[0]>=tobii_box_xmin and self.gp[0]<=tobii_box_xmax and self.gp[1]>=tobii_box_ymin and self.gp[1]<=tobii_box_ymax:
                            fixate_start_time = rospy.get_time()
                            while self.gp[0]>=tobii_box_xmin and self.gp[0]<=tobii_box_xmax and self.gp[1]>=tobii_box_ymin and self.gp[1]<=tobii_box_ymax:
                                rospy.loginfo("Yolo Node: registering interest...")
                                fixation_update = rospy.get_time()
                                fixation_duration = fixation_duration + (fixation_update - fixate_start_time)
                                if self.gp[0]<=tobii_box_xmin or self.gp[0]>=tobii_box_xmax or self.gp[1]<=tobii_box_ymin or self.gp[1]>=tobii_box_ymax:
                                    break
                                
                                # If gaze point lies within the bounding box for longer 0.300 seconds 
                                # (Fixation value obtained from preliminary user study).
                                # Register fixation as interest in object.
                                if fixation_duration >= 0.300:
                                    rospy.loginfo("Yolo Node: Fixation duration, " + str(fixation_duration))
                                    # Check if this 3D point has already been registered before
                                    # using a 5% upper and lower bound for x, y.
                                    screen_coordinates_xy = self.get_bounding_box_pixel_coordinates(box)
                                    #rectified_point = self.front_camera_info.rectifyPoint(screen_coordinates_xy)
                                    real_world_point = self.pixel_to_3d_point(screen_coordinates_xy[0],screen_coordinates_xy[1])

                                    # For each point registered in the intention goal list,
                                    # check if the current 3D coordinate is within a 5%
                                    # deviation, if True then point has already been registered.
                                    if len(self.intention_goal_list)>=1:
                                        check = self.check_deviation(real_world_point)
                                        if check == True:
                                            rospy.loginfo("Yolo Node: Point already registered")
                                            break
                                        else:
                                            self.intention_goal_list.append(real_world_point)
                                            self.intention_goal_pub.publish(real_world_point)                                               
                                            rospy.loginfo("Yolo Node: Point registered successfully!")
                                            break
                                    elif len(self.intention_goal_list)==0:
                                        self.intention_goal_list.append(real_world_point)
                                        self.intention_goal_pub.publish(real_world_point)
                                        rospy.loginfo("Yolo Node: Point registered successfully!")  
                                        break
                        else:
                            break
                        if self.debug == True:
                            print_out = " ".join(["[DEBUG] Yolo Node:", str(box_xymin[0]), str(box_xymin[1]), str(box_xymax[0]), str(box_xymax[1])])
                            rospy.loginfo(print_out)
                    else:
                        rospy.loginfo("Yolo Node: Frame ratio is not set")
                        break
        else:
            rospy.loginfo("Yolo Node: No bounding boxes")
                

    
    def qr_codes(self, data):

        detected_qr_codes = data

        if not detected_qr_codes:
            self.screen_calibrated = False
            rospy.loginfo("Yolo Node: No QRCODES detected")

        elif detected_qr_codes:
            #rospy.loginfo("Yolo Node: Decoding QRCODES...")
            found = []
            for qr_code in detected_qr_codes:
                (x, y, width, height) = qr_code.rect
                qr_code_info = qr_code.data.decode("utf-8")
                qr_code_bbox = (x, y, (x + width), (y + height))

                if qr_code_info in self.qr_predefined_list:
                    # qr_code_info = QR Code information 'A_001','B_001',...
                    #
                    # Takes the outer coordinates of each QR Code point to form a large bounding box,
                    # of the entire screen.
                    #
                    # Using QR Codes to define screen boundaries rather than Yolo object detection to 
                    # enable detection of a specific screen and not all screens, 
                    # i.e. TV's, Computer Monitors, other screens visable by the participant.
                    if qr_code_info == "A_001": # A_001 top left
                        self.top_left_screen = (qr_code_bbox[0], qr_code_bbox[1])
                        found.append((qr_code_info, qr_code_bbox))
                    if qr_code_info == "B_001": # B_001 top right
                        self.top_right_screen = (qr_code_bbox[2], qr_code_bbox[1])
                        found.append((qr_code_info, qr_code_bbox))
                    if qr_code_info == "C_001": # C_001 bottom left
                        self.bottom_left_screen = (qr_code_bbox[0], qr_code_bbox[3])
                        found.append((qr_code_info, qr_code_bbox))
                    if qr_code_info == "D_001": # D_001 bottom right
                        self.bottom_right_screen = (qr_code_bbox[2], qr_code_bbox[3])
                        found.append((qr_code_info, qr_code_bbox))

                if self.debug == True:
                    rospy.loginfo("[DEBUG] Yolo Node: " + str(qr_code_info) + str(qr_code_bbox))
        if found:
            # frame_1 = Real world screen.
            # frame_2 = Tobii Glasses frame of real world screen.

            # Dynamic width (xmax - xmin)
            # Dynamic Height (ymax - ymin)

            found.sort()
            #rospy.loginfo("found " + str(found))
            perm_1 = ['A_001','B_001','C_001']
            perm_2 = ['B_001','C_001','D_001']
            perm_3 = ['A_001','C_001','D_001']   
            perm_4 = ['A_001','B_001','D_001']

            extract_qr_info = []
            if len(found) >= 3:
                for i in range(len(found)):
                    extract_qr_info.append(found[i][0]) 
                #rospy.loginfo(extract_qr_info)
                if extract_qr_info == perm_1:
                    self.bottom_right_screen = (found[1][1][2], found[2][1][3])
                    self.frame_2_width = self.top_right_screen[0] - self.top_left_screen[0]
                    self.frame_2_height = self.bottom_left_screen[1] - self.top_left_screen[1]

                    # Obtain ratio between screen dimensions for calculation in converting pixel coordinates
                    # between frames.
                    self.frame_to_frame_ratio_width = float(self.frame_2_width/self.frame_1_width)
                    self.frame_to_frame_ratio_height = float(self.frame_2_height/self.frame_1_height)
                    self.screen_calibration_type = "perm_1"
                    rospy.loginfo("Yolo Node: D_001 missing value estimated")

                if extract_qr_info == perm_2:
                    self.top_left_screen = (found[1][1][0], found[0][1][1])
                    self.frame_2_width = self.bottom_right_screen[0] - self.bottom_left_screen[0]
                    self.frame_2_height = self.bottom_right_screen[1] - self.top_right_screen[1]

                    # Obtain ratio between screen dimensions for calculation in converting pixel coordinates
                    # between frames.
                    self.frame_to_frame_ratio_width = float(self.frame_2_width/self.frame_1_width)
                    self.frame_to_frame_ratio_height = float(self.frame_2_height/self.frame_1_height)
                    self.screen_calibration_type = "perm_2"
                    rospy.loginfo("Yolo Node: A_001 missing value estimated")

                if extract_qr_info == perm_3:
                    self.top_right_screen = (found[2][1][2], found[0][1][1])
                    self.frame_2_width = self.bottom_right_screen[0] - self.bottom_left_screen[0]
                    self.frame_2_height = self.bottom_left_screen[1] - self.top_left_screen[1]

                    # Obtain ratio between screen dimensions for calculation in converting pixel coordinates
                    # between frames.
                    self.frame_to_frame_ratio_width = float(self.frame_2_width/self.frame_1_width)
                    self.frame_to_frame_ratio_height = float(self.frame_2_height/self.frame_1_height)
                    self.screen_calibration_type = "perm_3"
                    rospy.loginfo("Yolo Node: B_001 missing value estimated")

                if extract_qr_info == perm_4:
                    self.bottom_left_screen = (found[0][1][0], found[2][1][3])
                    self.frame_2_width = self.top_right_screen[0] - self.top_left_screen[0]
                    self.frame_2_height = self.bottom_right_screen[1] - self.top_right_screen[1]

                    # Obtain ratio between screen dimensions for calculation in converting pixel coordinates
                    # between frames.
                    self.frame_to_frame_ratio_width = float(self.frame_2_width/self.frame_1_width)
                    self.frame_to_frame_ratio_height = float(self.frame_2_height/self.frame_1_height)
                    self.screen_calibration_type = "perm_4"
                    rospy.loginfo("Yolo Node: C_001 missing value estimated")

                if extract_qr_info == self.qr_predefined_list :

                    # Dynamic width (xmax - xmin)
                    # Dynamic Height (ymax - ymin)
                    self.frame_2_width = self.top_right_screen[0] - self.top_left_screen[0]
                    self.frame_2_height = self.bottom_right_screen[1] - self.top_right_screen[1]

                    # Obtain ratio between screen dimensions for calculation in converting pixel coordinates
                    # between frames.
                    self.frame_to_frame_ratio_width = float(self.frame_2_width/self.frame_1_width)
                    self.frame_to_frame_ratio_height = float(self.frame_2_height/self.frame_1_height)
                    self.screen_calibration_type = "perm_all"
                    rospy.loginfo("Yolo Node: All QRCODES found")

                self.screen_calibrated = True
            elif len(found) < 3:
                rospy.loginfo("Yolo Node: Not all QRCODES detected")
                self.screen_calibrated = False

            if self.debug == True:
                rospy.loginfo("Yolo Node: QRCODES detected " + str(found))
                
    
    def eval_looking_at_screen(self):

        # Attention score is calculated as a percentage of time the gaze remains in within
        # the screen bounaries. Every time the gaze enters and leaves the screen, the amount
        # of time (in seconds) is totalled. Once the total time exceeds "#" seconds, the score is
        # calculated and published.

        # While gaze point is within the detected screen boundaries 
        # defined by the QR codes, proceed with person detection.
        self.attention_duration = 0
        self.distracted_duration = 0
        if self.gp[0]>=self.top_left_screen[0] and self.gp[0]<=self.top_right_screen[0] and self.gp[1]>=self.top_left_screen[1] and self.gp[1]<=self.bottom_right_screen[1]:
            attention_start = rospy.get_time()
            while self.gp[0]>=self.top_left_screen[0] and self.gp[0]<=self.top_right_screen[0] and self.gp[1]>=self.top_left_screen[1] and self.gp[1]<=self.bottom_right_screen[1]:
                rospy.loginfo("Yolo Node: Looking at screen")
                attention_update = rospy.get_time()
                self.attention_duration = self.attention_duration + (attention_update - attention_start)
                self.total_time = self.attention_duration + self.distracted_duration
                if self.total_time >= float(self.attention_rate):
                    self.attention_score()
                if self.gp[0]<=self.top_left_screen[0] or self.gp[0]>=self.top_right_screen[0] or self.gp[1]<=self.top_left_screen[1] or self.gp[1]>=self.bottom_right_screen[1]:
                    break
                
        if self.gp[0]<=self.top_left_screen[0] or self.gp[0]>=self.top_right_screen[0] or self.gp[1]<=self.top_left_screen[1] or self.gp[1]>=self.bottom_right_screen[1]:             
            distracted_start_time = rospy.get_time()
            while self.gp[0]<=self.top_left_screen[0] or self.gp[0]>=self.top_right_screen[0] or self.gp[1]<=self.top_left_screen[1] or self.gp[1]>=self.bottom_right_screen[1]:                
                rospy.loginfo("Yolo Node: Looking away from screen")
                distracted_update = rospy.get_time()
                self.distracted_duration = self.distracted_duration + (distracted_update - distracted_start_time)
                self.total_time = self.attention_duration + self.distracted_duration
                if self.gp[0]>=self.top_left_screen[0] and self.gp[0]<=self.top_right_screen[0] and self.gp[1]>=self.top_left_screen[1] and self.gp[1]<=self.bottom_right_screen[1]:
                    break
                if self.total_time >= float(self.attention_rate):
                    self.attention_score()

    
    def attention_score(self):
        attention_score = (self.attention_duration/self.attention_rate) * 100

        if attention_score >= float(100):
            attention_score = int(100)
            self.attention_pub.publish(attention_score)
            if self.debug == True:
                rospy.loginfo("Yolo Node: Published " + str(attention_score))
        else:
            self.attention_pub.publish(int(attention_score))
            if self.debug == True:
                rospy.loginfo("Yolo Node: Published " + str(attention_score))

    
    def image_to_tiles(self, img, scale):
        image = img
        width = image.shape[1]
        height = image.shape[0]

        l1 = image[0:(height/scale), 0:(width/scale)]
        l2 = image[(height/scale):height, 0:(width/scale)]
        r1 = image[0:(height/scale), (width/scale):width]
        r2 = image[(height/scale):height, (width/scale):width]
        #cv2.imshow('image', l1)
        #cv2.waitKey(5)
        return (l1,l2,r1,r2)
                    
    
    def pixel_coordinate_conversion(self, pixel_x, pixel_y, width_r, height_r, ref_frame):
        frame_to_frame_ratio_width = width_r
        frame_to_frame_ratio_height = height_r
        if ref_frame == "tobii":
            screen_pixel_x = float(pixel_x / frame_to_frame_ratio_width)
            screen_pixel_y = float(pixel_y / frame_to_frame_ratio_height)
            coordinates = (screen_pixel_x, screen_pixel_y)

        elif ref_frame == "real_world":
            tobii_pixel_x = float(pixel_x * frame_to_frame_ratio_width)
            tobii_pixel_y = float(pixel_y * frame_to_frame_ratio_height)
            coordinates = (tobii_pixel_x, tobii_pixel_y)
        # Return the converted coordinates
        return coordinates

    
    def pixel_to_3d_point(self, box_x, box_y):
        #Use camera model to get somewhat accurate 3d postiion for X & Y
        vector = self.front_camera_info.projectPixelTo3dRay((box_x,box_y))
        #Noramlise the 3dray
        ray_z = [el / vector[2] for el in vector]
        #Assign vector results to easier accessable varaibles 
        x = vector[0]
        y = vector[1]
        #Use cKDTree to search for and accurate Z with results closest to the one inputtted
        points = self.xyz_array[:,0:2] # drops the z column
        tree = scpy.cKDTree(points)
        tree_index = tree.query((x, y))[1] # this returns a tuple, we want the index
        result = self.xyz_array[tree_index, 2]
        #Assign new Z to normalised X Y
        ray_z[2] = result
        positions = ray_z

        point = PointStamped()
        point.header.frame_id = self.front_camera_info.tfFrame()
        point.header.stamp = rospy.Time.now()
        point.point.x = positions[0]
        point.point.y = positions[1]
        point.point.z = 1

        # Transform the point between the camera optical frame and the map frame.
        self.listener.waitForTransform(self.front_camera_info.tfFrame(), "map", rospy.Time.now(), rospy.Duration(5.0))
        
        tf_point = self.listener.transformPoint("map", point)
        rospy.loginfo(str(tf_point))
        return tf_point

    
    def get_bounding_box_pixel_coordinates(self, box):
        # Get the center pixel coordinate of the bounding box.
        x = (box.xmin + box.xmax)/2
        y = (box.ymin + box.ymax)/2
        pixel_coordinate = (int(x), int(y))
        return pixel_coordinate

    
    def check_deviation(self, real_point):
        check_list = []
        for point in self.intention_goal_list:
            rospy.loginfo("Yolo Node: Checking against existing list...")

            # Compare new point with each existing point within a 1.2x1.2m square around the
            # existing point.  z values are ignored.
             
            x_upper_bound = point.point.x + 0.6
            x_lower_bound = point.point.x - 0.6
            y_upper_bound = point.point.y + 0.6
            y_lower_bound = point.point.y - 0.6

            if real_point.point.x>=x_lower_bound and real_point.point.x<=x_upper_bound:
                x_point_within = True
            else:
                x_point_within = False

            if real_point.point.y>=y_lower_bound and real_point.point.y<=y_upper_bound:
                y_point_within = True
            else:
                y_point_within = False

            if x_point_within == True and y_point_within == True:
                check_list.append(True)
            else:
                check_list.append(False)

        if True in check_list:
            return True
        else:
            return False

if __name__ == "__main__":
    rospy.init_node('object_detection_node')
    while not rospy.is_shutdown():
        try:
            ObjectDetect()
            rospy.spin()
        except rospy.ROSInterruptException:
            pass