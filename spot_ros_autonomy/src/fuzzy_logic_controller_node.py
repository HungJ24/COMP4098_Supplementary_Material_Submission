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
# CHAMP is an open source development framework for building new quadrupedal 
# robots and developing new control algorithms.
# [Available at https://github.com/chvmp/champ]
#
# Tested on Ubuntu 18.04, kernel 5.11, ROS Melodic
#
# Where specificed using the '#---' annotation, the author of the original 
# work is referenced.
#
# ---------------------------------------------------------------------------


import rospy
import actionlib
import tf
import numpy as np
import math

from actionlib_msgs.msg import GoalStatus, GoalID
from cv_bridge import CvBridge
from darknet_ros_msgs.msg import BoundingBoxes
from geometry_msgs.msg import Pose, Pose2D, PointStamped, Twist, PoseStamped
from sensor_msgs.msg import Image, Joy, LaserScan
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.srv import GetPlanRequest, GetPlan
from nav_msgs.msg import Odometry
from std_msgs.msg import Int8

from rpy2 import robjects

import roslib; roslib.load_manifest('champ_teleop')
from champ_msgs.msg import Pose as PoseLite


class FLC:
    def __init__(self):
        self.rate = rospy.Rate(20)
        self.debug = rospy.get_param('~debug')
        self.ai = rospy.get_param('~ai')

        self.speed = 0.6
        self.turn = 0.6
        self.dist_threshold = 0.4
        self.rear_dist_threshold = 0.5

        self.front_closest_obstacle = None

        self.rear_closest_obstacle = None

        self.left_closest_obstacle = None

        self.right_closest_obstacle = None

        self.attention_score = 0
        self.intention_goal_err = 0
        self.intention_goal_list = []
        self.intention_goal_paths = []
        self.visited = []
        self.current_odom = None
        self.goal = None
        self.largest_box = None
        self.per_boxes = []
        self.front_img = None

        self.robot_pose = Pose2D()
        
        try:
            r = robjects.r
            robjects.r("""source('~/spot_ws/src/spot_ros_autonomy/src/FLC.R')""")
        except:
            rospy.loginfo("FLC Node: Error opening file")
        
        self.obs_fis_eval_r = robjects.globalenv['return_obs_eval_fis']
        self.loa_fis_eval_r = robjects.globalenv['return_loa_eval_fis']

        if self.ai == True:
            self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
            self.front_img_sub = rospy.Subscriber('/camera1_IR/color/image_raw', Image, self.front_img_callback)
            self.cancel_pub = rospy.Publisher('/move_base/cancel', GoalID, queue_size=1)
            self.attention_score_sub = rospy.Subscriber('/attention_score', Int8, self.attention_callback)
            self.intention_goal_sub = rospy.Subscriber('/object_coordinates/3d_coordinates', PointStamped, self.intention_goal_callback)

        self.velocity_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.pose_lite_pub = rospy.Publisher('/body_pose/raw', PoseLite, queue_size=10)
        self.pose_pub = rospy.Publisher('/body_pose', Pose, queue_size=10)

        self.front_laser_sub = rospy.Subscriber('/front/scan/laserscan', LaserScan, self.front_depth_callback)
        self.rear_laser_sub = rospy.Subscriber('/rear/scan/laserscan', LaserScan, self.rear_depth_callback)
        self.left_laser_sub = rospy.Subscriber('/left/scan/laserscan', LaserScan, self.left_depth_callback)
        self.right_laser_sub = rospy.Subscriber('/right/scan/laserscan', LaserScan, self.right_depth_callback)
        self.front_box_sub = rospy.Subscriber('/front/darknet_ros/bounding_boxes', BoundingBoxes, self.front_box_callback)

        rospy.sleep(5)

        self.joy_subscriber = rospy.Subscriber('/joy', Joy, self.joy_callback)


    
    def odom_callback(self, msg):
        # Get (x, y, theta) specification from odometry topic
        quaternion = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                        msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(quaternion)

        self.robot_pose.theta = yaw
        self.robot_pose.x = msg.pose.pose.position.x
        self.robot_pose.y = msg.pose.pose.position.y


    def front_depth_callback (self, msg):
        front_depth = np.array(msg.ranges[341:682])
        front_left_depth = np.array(msg.ranges[683:1023])
        front_right_depth = np.array(msg.ranges[0:340])

        for i in range(len(front_depth)):
            nan = math.isnan(front_depth[i])
            if nan == True:
                front_depth[i] = float(1)
        for i in range(len(front_left_depth)):
            nan = math.isnan(front_left_depth[i])
            if nan == True:
                front_left_depth[i] = float(1)
        for i in range(len(front_right_depth)):
            nan = math.isnan(front_right_depth[i])
            if nan == True:
                front_right_depth[i] = float(1)

        self.front_closest_obstacle = round(min(min(front_depth),min(front_left_depth),min(front_right_depth)),2)
        if self.debug == True:
            rospy.loginfo("FLC Node: Front - " + str(self.front_closest_obstacle))
        
    
    def rear_depth_callback (self, msg):
        rear_depth = np.array(msg.ranges[341:682])
        rear_left_depth = np.array(msg.ranges[683:1023])
        rear_right_depth = np.array(msg.ranges[0:340])

        for i in range(len(rear_depth)):
            nan = math.isnan(rear_depth[i])
            if nan == True:
                rear_depth[i] = float(1)
        for i in range(len(rear_left_depth)):
            nan = math.isnan(rear_left_depth[i])
            if nan == True:
                rear_left_depth[i] = float(1)
        for i in range(len(rear_right_depth)):
            nan = math.isnan(rear_right_depth[i])
            if nan == True:
                rear_right_depth[i] = float(1)

        self.rear_closest_obstacle = round(min(min(rear_depth),min(rear_left_depth),min(rear_right_depth)),2)
        if self.debug == True:
            rospy.loginfo("FLC Node: Rear - " + str(self.rear_closest_obstacle))

    
    def left_depth_callback (self, msg):
        left_depth = np.array(msg.ranges[341:682])
        left_front_depth = np.array(msg.ranges[0:340])
        left_rear_depth = np.array(msg.ranges[683:1023])

        for i in range(len(left_depth)):
            nan = math.isnan(left_depth[i])
            if nan == True:
                left_depth[i] = float(1)
        for i in range(len(left_front_depth)):
            nan = math.isnan(left_front_depth[i])
            if nan == True:
                left_front_depth[i] = float(1)
        for i in range(len(left_rear_depth)):
            nan = math.isnan(left_rear_depth[i])
            if nan == True:
                left_rear_depth[i] = float(1)

        self.left_closest_obstacle = round(min(min(left_depth),min(left_front_depth),min(left_rear_depth)),2)
        if self.debug == True:
            rospy.loginfo("FLC Node: Left - " + str(self.left_closest_obstacle))

    
    def right_depth_callback (self, msg):
        right_depth = np.array(msg.ranges[341:682])
        right_front_depth = np.array(msg.ranges[683:1023])
        right_rear_depth = np.array(msg.ranges[0:340])

        for i in range(len(right_depth)):
            nan = math.isnan(right_depth[i])
            if nan == True:
                right_depth[i] = float(1)
        for i in range(len(right_front_depth)):
            nan = math.isnan(right_front_depth[i])
            if nan == True:
                right_front_depth[i] = float(1)
        for i in range(len(right_rear_depth)):
            nan = math.isnan(right_rear_depth[i])
            if nan == True:
                right_rear_depth[i] = float(1)

        self.right_closest_obstacle = round(min(min(right_depth),min(right_front_depth),min(right_rear_depth)),2)
        if self.debug == True:
            rospy.loginfo("FLC Node: Right - " + str(self.right_closest_obstacle))


    def check_closest_obs(self):
        closest_obs = min(self.front_closest_obstacle, self.rear_closest_obstacle, self.left_closest_obstacle, self.right_closest_obstacle)

        if isinstance(closest_obs, (int, float)):
            if math.isnan(closest_obs):
                self.closest_obs = 1
            else:
                self.closest_obs = closest_obs
        else:
            self.closest_obs = 1

        if self.debug == True:
            rospy.loginfo("FLC Node: Closest - " + str(self.closest_obs))


    def front_img_callback(self, msg):
        bridge = CvBridge()
        self.front_img = bridge.imgmsg_to_cv2(msg, 'bgr8')
        if self.debug == True:
            rospy.loginfo("[DEBUG] Yolo Node: Camera Image obtained")

    
    def front_box_callback(self, msg):
        if not msg.bounding_boxes:
            self.largest_box = None
        else:
            boxes = msg.bounding_boxes
            if boxes:
                for box in boxes:
                    if box.Class == "person": 
                        self.per_boxes.append(box)
            else:
                self.per_boxes = []

    
    def attention_callback(self, msg):
        attention_score = msg.data
        self.attention_score = int(attention_score)
        if self.debug == True:
            print_out = " ".join(["[DEBUG] FLC Node: Attention Score,", str(self.attention_score)])
            rospy.loginfo(print_out)

    
    def intention_goal_callback(self, msg):
        goal_point = msg

        if goal_point not in self.intention_goal_list:
            self.intention_goal_list.append(goal_point) 
            if self.debug == True:
                print_out = " ".join(["[DEBUG] FLC Node: Goal Point added -", str(goal_point.point.x), str(goal_point.point.y), str(goal_point.point.z)])
                rospy.loginfo(print_out)
        else:
            rospy.loginfo("FLC Node: Goal Point already exists")

    
    def joy_callback(self, msg):

        self.check_closest_obs()
        
        while self.closest_obs <= self.dist_threshold or self.rear_closest_obstacle <= self.rear_dist_threshold:
            rospy.loginfo("FLC Node: Obstacle! Speed set to 0.1 m/s")
        
            obs_fis_out = self.obs_fis_eval_r(self.front_closest_obstacle, self.rear_closest_obstacle, self.left_closest_obstacle, self.right_closest_obstacle)
            obs_fis_out_np = np.array(obs_fis_out)
            obs_fis_crisp = obs_fis_out_np[0,0]

            if math.isnan(obs_fis_crisp):
                if self.debug == True:
                    rospy.loginfo("FLC Node: Nan value from obstacle FLC")
                break
            else:
                self.obstacle_avoidance_fis(obs_fis_crisp)
                self.check_closest_obs()
            
            if self.debug == True:
                print_out = " ".join(["[DEBUG] FLC Node: front:", str(self.front_closest_obstacle), ", rear:", str(self.rear_closest_obstacle), ", left:", str(self.left_closest_obstacle), ", right:", str(self.right_closest_obstacle)])
                rospy.loginfo(obs_fis_crisp)
                rospy.loginfo(print_out)
                self.obstacle_avoidance_fis(obs_fis_crisp)

        if self.ai == True:
            self.check_LOA()
        
            if self.LOA <= 0.4:
                if self.intention_goal_list:
                    goal_path_pair = self.closest_goal_point()
                    if goal_path_pair[0] == False and goal_path_pair[1] == False:
                        rospy.loginfo("FLC Node: Closest goal point unattainable")
                    else:
                        self.goal = goal_path_pair[0]
                        self.intention_goal_err = int(self.calculate_motion_err(goal_path_pair[1]))
                        rospy.loginfo("FLC Node: Intention-based Goal Motion Error - " + str(self.intention_goal_err))
                        rospy.loginfo("FLC Node: Moving to goal")
                if self.goal == None:
                    rospy.loginfo("FLC Node: No goal to target")
                    self.joy_cmd(msg)
                else:
                    self.move_to_goal(self.goal)

            if self.LOA > 0.4 and self.LOA < 0.6:
                if self.per_boxes:
                    for box in self.per_boxes:
                        if box.xmin>=0 and box.xmax<=self.front_img.shape[1] and box.ymin>=0 and box.ymax<=self.front_img.shape[0]:
                            current_largest = box.xmax * box.ymax
                            if self.largest_box == None:
                                self.largest_box = (current_largest, box)
                            elif current_largest>self.largest_box[0]:
                                self.largest_box = (current_largest, box)

                    box_center_pixel = self.get_bounding_box_pixel_coordinates(self.largest_box[1])
                    box_x = box_center_pixel[0]
                    prop_error = box_x - ((self.front_img.shape[1]) / 2)

                    twist = Twist()
                    if prop_error < 0:
                        twist.angular.z = 0.1
                        self.velocity_pub.publish(twist)
                    if prop_error > 0:
                        twist.angular.z = -0.1
                        self.velocity_pub.publish(twist)
                    if prop_error == 0:
                        twist.linear.x = 0
                        twist.linear.y = 0
                        twist.angular.z = 0
                        self.velocity_pub.publish(twist)
                        rospy.loginfo("FLC Node: Attention Score - " + str(self.attention_score))
                    rospy.loginfo("FLC Node: Centering on bbox")
                else:
                    rospy.loginfo("FLC Node: No Bounding boxes")

            if self.LOA >= 0.6:
                self.cancel_pub.publish(GoalID())
                self.joy_cmd(msg)

        elif self.ai == False:
            self.joy_cmd(msg)
            
    
    def joy_cmd(self, msg):

        # If no obstacle return to normal speed 0.6 m/s.
        rospy.loginfo("FLC Node: Area Clear Speed set to 0.6 m/s")
        speed = 0.6 
        turn = 0.6

        twist = Twist()

        twist.linear.x = msg.axes[1] * speed
        twist.linear.y = msg.buttons[4] * msg.axes[0] * speed
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = (not msg.buttons[4]) * msg.axes[0] * turn
        self.velocity_pub.publish(twist)

        body_pose_lite = PoseLite()
        body_pose_lite.x = 0
        body_pose_lite.y = 0
        body_pose_lite.roll = 0
        body_pose_lite.pitch = 0
        body_pose_lite.yaw = 0

        if msg.axes[5]<0:
            body_pose_lite.z = 0

        self.pose_lite_pub.publish(body_pose_lite)

        body_pose = Pose()
        body_pose.position.z = body_pose_lite.z

        quaternion = tf.transformations.quaternion_from_euler(body_pose_lite.roll, body_pose_lite.pitch, body_pose_lite.yaw)
        body_pose.orientation.x = quaternion[0]
        body_pose.orientation.y = quaternion[1]
        body_pose.orientation.z = quaternion[2]
        body_pose.orientation.w = quaternion[3]

        self.pose_pub.publish(body_pose)


    def check_LOA(self):
        
        LOA_out = self.loa_fis_eval_r(self.attention_score, self.intention_goal_err)
        LOA_np = np.array(LOA_out)
        check = LOA_np[0,0]
        if isinstance(check, (int, float)):
            self.LOA = check
        else:
            self.LOA = 1
        if self.debug == True:
            rospy.loginfo(self.LOA)

    
    def get_bounding_box_pixel_coordinates(self, box):
        # Get the center pixel coordinate of the bounding box.
        x = (box.xmin + box.xmax)/2
        y = (box.ymin + box.ymax)/2
        pixel_coordinate = (int(x), int(y))
        return pixel_coordinate

    
    def calculate_motion_err(self, path):
        rospy.loginfo("FLC Node: Calculating motion error...")
        point = path[0]
        point_angular_z = point.pose.orientation.z

        # Optimal velocity is the maximum linear speed of the robot.
        # Take current robot linear velocity in respect to direction
        # and compare against the optimal path's orientation.
        optimal_velocity = 0.6
        optimal_orientation = point_angular_z
        current_velocity_x = self.robot_pose.x
        current_angular_z = self.robot_pose.theta

        if current_velocity_x < 0:
            # Robot is moving in reverse, this action is most likely intented.
            error = 0
        else:
            angular_deviation = current_angular_z / optimal_orientation
            velocity_deviation = current_velocity_x / optimal_velocity
            # Current orientation of yaw should be within 3% of the optimal path.
            if angular_deviation<=0.03 or angular_deviation>=0.97:
                error = 0
            # As the robot rotates away from the optimal orientation (0.188-3.14 radians)
            # then the error increases from 0-100%
            elif angular_deviation<0.5 and velocity_deviation<=0.7:
                angular_error = ((angular_deviation - 0.03) / 0.46)
                velocity_error = 1 - velocity_deviation
                error = (angular_error + velocity_error) / 2
            # As the robot rotates away from the optimal orientation (6.09-3.14 radians)
            # then the error increases from 0-100%
            elif angular_deviation>=0.5 and velocity_deviation<=0.7:
                angular_error = ((1 - (angular_deviation + 0.03)) / 0.48)
                velocity_error = 1 - velocity_deviation
                error = (angular_error + velocity_error) / 2
        rospy.loginfo("FLC Node: Motion error = " + str(error))
        return error


    def closest_goal_point(self):
        rospy.loginfo("FLC Node: Getting best path...")
        get_plan = rospy.ServiceProxy('/move_base/make_plan', GetPlan, False)
        plan_req = GetPlanRequest()
        best_goal = None
        best_path = None
        current_path = None

        for goal in self.intention_goal_list:
            start = PoseStamped()
            start.header.frame_id = 'map'
            start.header.stamp = rospy.Time.now()
            start.pose.position.x = self.robot_pose.x
            start.pose.position.y = self.robot_pose.y

            Goal = PoseStamped()
            Goal.header.frame_id = 'map'
            Goal.header.stamp = rospy.Time.now()
            Goal.pose.position.x = goal.point.x
            Goal.pose.position.y = goal.point.y

            plan_req.start = start
            plan_req.goal = Goal
            plan_req.tolerance = 0.5
            try:
                respond = get_plan(plan_req)
                success = True
            except rospy.ServiceException as exc:
                rospy.loginfo("FLC Node: Service excpetion for GetPlan: " + str(exc))

            if success == True:
                current_path = len(respond.plan.poses)
                if best_path == None:
                    best_path, best_goal = current_path, goal
                elif current_path < best_path:
                    best_path, best_goal = current_path, goal    
            else:
                rospy.loginfo("FLC Node: Could not get plan")

        if success == True:
            rospy.loginfo("FLC Node: Best Path = " + str(best_path))
            return (best_goal, respond.plan.poses)

    
    def move_to_goal(self, point):
        if point not in self.visited:
            final_goal = point
            client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

            while not client.wait_for_server(rospy.Duration.from_sec(5.0)):
                rospy.loginfo("FLC Node: Waiting for the move_base action server")

            goal = MoveBaseGoal()

            goal.target_pose.header.frame_id = "map"
            goal.target_pose.header.stamp = rospy.Time.now()

            # Set positions of the goal location
            goal.target_pose.pose.position.x = final_goal.point.x
            goal.target_pose.pose.position.y = final_goal.point.y
            goal.target_pose.pose.orientation.x = 0.0
            goal.target_pose.pose.orientation.y = 0.0
            goal.target_pose.pose.orientation.z = 0.0
            goal.target_pose.pose.orientation.w = 1.0

            rospy.loginfo("FLC Node: Sending goal location on map")

            client.send_goal(goal)
            client.wait_for_result(rospy.Duration(30))
        
            if (client.get_state() ==  GoalStatus.SUCCEEDED):
                rospy.loginfo("FLC Node: You have reached the destination") 
                self.intention_goal_list.remove(self.goal)
                self.visited.append(self.goal)       
            else:
                rospy.loginfo("FLC Node: The robot failed to reach the destination")
        else:
            rospy.loginfo("FLC Node: Goal already visited")


    def obstacle_avoidance_fis(self, obs_fis_crisp):
        # Returns a number in the range(0,10).
        # Number represents the direction of movement from an obstacle.
        speed = 0.2
        turn = 0.2
        twist = Twist()

        if obs_fis_crisp<=0.5: # Move forward
            rospy.loginfo("FLC Node: Moving forward")
            twist.linear.x = speed
            twist.linear.y = 0.0
            twist.linear.z = 0.0
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = 0.0
            self.velocity_pub.publish(twist)

        # Move back
        if obs_fis_crisp>0.5 and obs_fis_crisp<=1.5:
            rospy.loginfo("FLC Node: Moving back")
            twist.linear.x = -speed
            twist.linear.y = 0.0
            twist.linear.z = 0.0
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = 0.0
            self.velocity_pub.publish(twist)

        # Move left
        if obs_fis_crisp>1.5 and obs_fis_crisp<=2.5:
            rospy.loginfo("FLC Node: Moving left")
            twist.linear.x = 0.0
            twist.linear.y = speed
            twist.linear.z = 0.0
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = 0.0
            self.velocity_pub.publish(twist)

        # Move right
        if obs_fis_crisp>2.5 and obs_fis_crisp<=3.5:
            rospy.loginfo("FLC Node: Moving right")
            twist.linear.x = 0.0
            twist.linear.y = -speed
            twist.linear.z = 0.0
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = 0.0
            self.velocity_pub.publish(twist)

        # Rotate left
        if obs_fis_crisp>3.5 and obs_fis_crisp<=4.5:
            rospy.loginfo("FLC Node: Rotate left")
            twist.linear.x = 0.0
            twist.linear.y = 0.0
            twist.linear.z = 0.0
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = turn
            self.velocity_pub.publish(twist)

        # Rotate right
        if obs_fis_crisp>4.5 and obs_fis_crisp<=5.5:
            rospy.loginfo("FLC Node: Rotate right")
            twist.linear.x = 0.0
            twist.linear.y = 0.0
            twist.linear.z = 0.0
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = -turn
            self.velocity_pub.publish(twist)

        # Forward left diagonal
        if obs_fis_crisp>5.5 and obs_fis_crisp<=6.5:
            rospy.loginfo("FLC Node: Moving forward left diagonal")
            twist.linear.x = speed
            twist.linear.y = speed
            twist.linear.z = 0.0
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = 0.0
            self.velocity_pub.publish(twist)

        # Forward right diagonal
        if obs_fis_crisp>6.5 and obs_fis_crisp<=7.5:
            rospy.loginfo("FLC Node: Moving forward right diagonal")
            twist.linear.x = speed
            twist.linear.y = -speed
            twist.linear.z = 0.0
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = 0.0
            self.velocity_pub.publish(twist)

        # Reverse left diagonal
        if obs_fis_crisp>7.5 and obs_fis_crisp<=8.5:
            rospy.loginfo("FLC Node: Moving reverse left diagonal")
            twist.linear.x = -speed
            twist.linear.y = speed
            twist.linear.z = 0.0
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = 0.0
            self.velocity_pub.publish(twist)

        # Reverse right diagonal
        if obs_fis_crisp>8.5 and obs_fis_crisp<=9.5:
            rospy.loginfo("FLC Node: Moving reverse right diagonal")
            twist.linear.x = -speed
            twist.linear.y = -speed
            twist.linear.z = 0.0
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = 0.0
            self.velocity_pub.publish(twist)

        #Robot is surrounded by obstacles: Stop Moving.
        if obs_fis_crisp>9.5 and obs_fis_crisp<=10:            
            rospy.loginfo("FLC Node: Stopped moving")
            twist.linear.x = 0.0
            twist.linear.y = 0.0
            twist.linear.z = 0.0
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = 0.0
            self.velocity_pub.publish(twist)
        

if __name__ == "__main__":
    rospy.init_node('fuzzy_logic_controller_node')
    while not rospy.is_shutdown():
        try:
            FLC()
            rospy.spin()
        except rospy.ROSInterruptException:
            pass