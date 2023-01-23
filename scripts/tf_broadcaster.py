#!/usr/bin/env python
import roslib
import rospy
import tf
import numpy as np
from tf.transformations import quaternion_from_euler
from nav_msgs.msg import Odometry


class broadcaster:
    def __init__(self):
        rospy.init_node('tf_broadcaster')
        self.odom_subscriber = rospy.Subscriber('/odom', Odometry, self.handle_odom)
        self.odom_subscriber = rospy.Subscriber('/ground_truth', Odometry, self.handle_ground_truth)
        self.br = tf.TransformBroadcaster()

        rospy.spin()


    def handle_ground_truth(self, msg):
        timestamp = msg.header.stamp
        orient = msg.pose.pose.orientation
        gt_quat = np.array([orient.x, orient.y, orient.z, orient.w])

        # Publish the transform over tf
        self.br.sendTransform(
            (msg.pose.pose.position.x, msg.pose.pose.position.y, 0.),
            gt_quat,
            timestamp,
            "ground_truth",
            "map"
        )


    def handle_odom(self, msg):
        timestamp = msg.header.stamp
        dummy_quat = np.array([0, 0, 0, 1])
        orient = msg.pose.pose.orientation
        odom_quat = np.array([orient.x, orient.y, orient.z, orient.w])

        # Publish the transform over tf
        self.br.sendTransform(
            (msg.pose.pose.position.x, msg.pose.pose.position.y, 0.),
            odom_quat,
            timestamp,
            "base_link",
            "odom"
        )

        self.br.sendTransform(
            (0, 0, 0),
            dummy_quat,
            timestamp,
            "laser_frame",
            "base_link"
        )


if __name__ == '__main__':
    try:
        broadcaster()
    except rospy.ROSInterruptException:
        print("\n")
        pass