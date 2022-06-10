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
# tobiiglassesctrl ver==2.0.3
#
# Tested on Ubuntu 18.04, kernel 5.11, ROS Melodic.
#
# This is a modification of Amy Phung's TobiiGlassesController, 
# [Available at https://github.com/AmyPhung/tobii_glasses_ros] 
# based on Davide De Tommaso and Agnieszka Wykowska. 2019. TobiiGlassesPySuite.
# [available at https://github.com/ddetommaso/TobiiGlassesPyController]
#
# Where specificed using the '#---' annotation, the author of the original 
# work is referenced.
#
# ---------------------------------------------------------------------------


import cv2
import rospy
import time

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from imutils.video import WebcamVideoStream
from imutils.video import FPS

if hasattr(__builtins__, 'raw_input'):
      input=raw_input

from tobiiglassesctrl import TobiiGlassesController

class TobiiGlasses:
    def __init__(self):
        self.rate = rospy.Rate(60)
        self.debug = rospy.get_param('~debug')
        rospy.loginfo("Tobii Node: Creating Project and Participant")

        self.frame_id = rospy.get_param('~frame_id', 'tobii_camera')
        ipv4_address_wlan = rospy.get_param('~ipv4_address_wlan', '192.168.71.50')
        wlan = rospy.get_param('~wlan')

        if wlan == True:
            self.tobiiglasses = TobiiGlassesController(ipv4_address_wlan, video_scene=True)
        elif wlan == False:
            rospy.loginfo("Tobii Node: Looking for Tobii Glasses on Network...")
            self.tobiiglasses = TobiiGlassesController(None, video_scene=True)

        self.resized_img_pub = rospy.Publisher('/tobii_camera/image_converted', Image, queue_size=1)
        self.gaze_pub = rospy.Publisher('/tobii_gaze', PointStamped, queue_size=1)
        self._prev_ts = 0

        self.calibrate_check = False

        while self.calibrate_check is False:
            self.calibrate()
        
        if wlan==True:
            path = "rtsp://%s:8554/live/scene" % ipv4_address_wlan
            rospy.loginfo(path)
        else:
            path = "rtsp://%s:8554/live/scene" %self.tobiiglasses.get_address()
            rospy.loginfo(path)
        

        self.cap = WebcamVideoStream(path).start()
        time.sleep(1.0)
        self.fps = FPS().start()

        self.stream()

    def calibrate(self):
        battery_level = self.tobiiglasses.get_battery_level()
        battery_time_remain = (self.tobiiglasses.get_battery_remaining_time() / 60)
        print_out = " ".join(["Tobii Node: [Level:", str(battery_level), "%", "Mins:", str(float(battery_time_remain)), "]"])
        rospy.loginfo(print_out)
        rospy.loginfo("Tobii Node: Calibrate Participant")
        project_id = self.tobiiglasses.create_project("Fuzzy Variable Autonomy of Mobile Robots")
        participant_id = self.tobiiglasses.create_participant(project_id, "001")
        calibration_id = self.tobiiglasses.create_calibration(project_id, participant_id)
        input("Tobii Node: Put the calibration marker in front of the user, then press enter to calibrate")
        self.tobiiglasses.start_calibration(calibration_id)

        res = self.tobiiglasses.wait_until_calibration_is_done(calibration_id)

        if res is False:
            msg = "Tobii Node: Calibration Failed!"
            rospy.loginfo(msg)
        else:
            msg = "Tobii Node: Calibration Success!"
            rospy.loginfo(msg)
            self.calibrate_check = True


    def stream(self):
        # Read until video is completed.
        self.tobiiglasses.start_streaming()

        while not rospy.is_shutdown():
            battery_level = self.tobiiglasses.get_battery_level()
            battery_time_remain = (self.tobiiglasses.get_battery_remaining_time() / 60)
            print_out = " ".join(["Tobii Node: [Level:", str(battery_level), "%", "Mins:", str(float(battery_time_remain)), "]"])
            rospy.loginfo(print_out)
            
            frame = self.cap.read()
            gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_img, 150,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU )
            cv2.bitwise_not(thresh, thresh) 
            height, width = thresh.shape[:2]

            # ---------------------------------------------------------------------------
            # Author: Amy Phung
            # Extract gaze data.
            data = self.tobiiglasses.get_data()
            data_pts = data['pts']
            data_gp  = data['gp']

            time_offset = rospy.get_time() - data_pts['ts'] / 1000000.0
            t_ros_pts = data_pts['ts'] / 1000000.0 + time_offset # Image ROS time.
            t_gp_pts = data_gp['ts'] / 1000000.0 + time_offset # Gaze ROS time.
            # ---------------------------------------------------------------------------

            # Convert frame to ROS message.
            bridge = CvBridge()

            img_msg = bridge.cv2_to_imgmsg(thresh, 'mono8')
            img_msg.header.frame_id = self.frame_id
            img_msg.header.stamp = rospy.Time.from_sec(t_ros_pts)
            # Publish frame.
            self.resized_img_pub.publish(img_msg)

            # ---------------------------------------------------------------------------
            # Author: Amy Phung
            # Check for new gaze detection.
            if data_gp['ts'] > 0 and self._prev_ts != data_gp['ts']:
                # Convert gaze data to ROS message.
                gaze_msg = PointStamped()
                gaze_msg.header.frame_id = self.frame_id
                gaze_msg.header.stamp = rospy.Time.from_sec(t_gp_pts)
                gaze_msg.point.x = int(data_gp['gp'][0]*width)
                gaze_msg.point.y = int(data_gp['gp'][1]*height)

                # Publish gaze data.
                self.gaze_pub.publish(gaze_msg)

                # Update previous timestamp.
                self._prev_ts = data_gp['ts']
            # ---------------------------------------------------------------------------

            self.fps.update()

        # When everything done, release the video capture object.
        self.fps.stop()
        rospy.loginfo("Tobii Node: [INFO] elasped time in minutes: {:.2f}".format((self.fps.elapsed()/60)))
        rospy.loginfo("Tobii Node: [INFO] approx. FPS: {:.2f}".format(self.fps.fps()))
        self.cap.stop()

        self.tobiiglasses.stop_streaming()
        self.tobiiglasses.close()

    def bounding_box_callback(self):
        pass


if __name__ == "__main__":
    rospy.init_node('tobii_controller_node')
    while not rospy.is_shutdown():
        try:
            TobiiGlasses()
            rospy.spin()
        except rospy.ROSInterruptException:
            pass