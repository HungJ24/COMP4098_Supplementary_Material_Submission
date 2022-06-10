#!/usr/bin/env python

import rospy
import tf


from geometry_msgs.msg import Pose, Twist
from sensor_msgs.msg import Joy

import roslib; roslib.load_manifest('champ_teleop')
from champ_msgs.msg import Pose as PoseLite


class Train:
    def __init__(self):
        self.rate = rospy.Rate(10)
        self.debug = rospy.get_param('~debug')

        self.speed = 0.6
        self.turn = 0.6
        

        self.velocity_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.pose_lite_pub = rospy.Publisher('/body_pose/raw', PoseLite, queue_size=10)
        self.pose_pub = rospy.Publisher('/body_pose', Pose, queue_size=10)

        self.joy_subscriber = rospy.Subscriber('/joy', Joy, self.joy_callback)

    
    def joy_callback(self, msg):

        twist = Twist()

        twist.linear.x = msg.axes[1] * self.speed
        twist.linear.y = msg.buttons[4] * msg.axes[0] * self.speed
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = (not msg.buttons[4]) * msg.axes[0] * self.turn
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
    

if __name__ == "__main__":
    rospy.init_node('training_node')
    while not rospy.is_shutdown():
        try:
            Train()
            rospy.spin()
        except rospy.ROSInterruptException:
            pass