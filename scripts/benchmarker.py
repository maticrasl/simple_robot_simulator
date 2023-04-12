#!/usr/bin/env python
import rospy

import math
import numpy as np

from geometry_msgs.msg import Point, Quaternion
from nav_msgs.msg import Odometry
import os
import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from tf2_msgs.msg import TFMessage


scan_angle = 270
scan_step = 3
scan_angle_rad = np.deg2rad(scan_angle)
scan_step_rad = np.deg2rad(scan_step)
scan_average = 1
laser_frequency = 1
scan_range_min = 0.001
scan_range_max = 1.5


class Benchmarker:
    def __init__(self):
        rospy.init_node('benchmarker')

        self.benchmarking_path = rospy.get_param('~kitti_path', '/home/matic/Documents/Magistrska/Benchmarking')
        self.mapfile = rospy.get_param('~mapfile')
        self.generate_gt = int(rospy.get_param('~generate_gt', "0"))

        kitti_path = f"{self.benchmarking_path}/kitti_files/{self.mapfile}/"
        os.makedirs(kitti_path, exist_ok=True)

        if self.generate_gt == 1:
            self.gt_file = open(kitti_path + "ground_truth.txt", 'w')
            rospy.Subscriber('/ground_truth', Odometry, self.gt_write_kitti, self.gt_file, queue_size=50)
        else:
            self.odom_file = open(kitti_path + "odom.txt", 'w')
            rospy.Subscriber('/tf_old', TFMessage, self.handle_get_tf_old, queue_size=50)

        self.tf_listener = tf.TransformListener()
        self.odom_broadcaster = tf.TransformBroadcaster()

        self.saved_initial_pose = False
        self.publish_first_tf()

        rospy.spin()
        pass


    def publish_first_tf(self) -> None:
        timestamp = rospy.Time.now()

        print("Printing first tf transform")

        position = Point(rospy.get_param("initial_pos_x", "0"), rospy.get_param("initial_pos_y", "0"), 0)
        orientation = quaternion_from_euler(0, 0, float(rospy.get_param("initial_pos_a", "0")))

        self.odom_broadcaster.sendTransform(
            (position.x, position.y, position.z),
            orientation,
            timestamp,
            "base_link",
            "odom"
        )

        self.odom_broadcaster.sendTransform(
            (0, 0, 0),
            quaternion_from_euler(0, 0, 0),
            timestamp,
            "laser_frame",
            "base_link"
        )
        pass


    def gt_write_kitti(self, data: Odometry) -> None:
        """ Get GT position and write it to kitti file. """
        out_file = self.gt_file

        trans = [data.pose.pose.position.x, data.pose.pose.position.y]
        rot = [data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w]
        th = euler_from_quaternion(rot)[2]

        matrix = np.array([[1, 0, 0, trans[0]], [0, 1, 0, trans[1]], [0, 0, 1, 0], [0, 0, 0, 1]])
        matrix_rot = np.array([[math.cos(th), -math.sin(th), 0, 0], [math.sin(th), math.cos(th), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        matrix = matrix @ matrix_rot

        kitti_line = matrix[:3, :].reshape(12)
        kitti_line_string = " ".join([str(i) for i in kitti_line])

        out_file.write(kitti_line_string)
        out_file.write('\n')
        pass


    def odom_write_kitti(self, timestamp) -> None:
        """ Get SLAM position and write it to kitti file. """
        out_file = self.odom_file

        self.tf_listener.waitForTransform("/world", "/base_link", timestamp, rospy.Duration(4.0))
        (trans, rot) = self.tf_listener.lookupTransform('/world', '/base_link', timestamp)

        th = euler_from_quaternion([0, 0, rot[2], rot[3]])[2]

        matrix = np.array([[1, 0, 0, trans[0]], [0, 1, 0, trans[1]], [0, 0, 1, 0], [0, 0, 0, 1]])
        matrix_rot = np.array([[math.cos(th), -math.sin(th), 0, 0], [math.sin(th), math.cos(th), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        matrix = matrix @ matrix_rot

        kitti_line = matrix[:3, :].reshape(12)
        kitti_line_string = " ".join([str(i) for i in kitti_line])

        out_file.write(kitti_line_string)
        out_file.write('\n')
        pass


    def finish_benchmarking(self) -> None:
        self.gt_file.close()
        self.odom_file.close()
        pass


    def publish_odometry(self, transform: TFMessage) -> None:
        """ Publish the robot movement over tf for the SLAM algorithm to use it. """
        self.odom_broadcaster.sendTransform(
            (transform.transform.translation.x, transform.transform.translation.y, 0.),
            (0, 0, transform.transform.rotation.z, transform.transform.rotation.w),
            transform.header.stamp,
            "base_link",
            "odom"
        )

        self.odom_broadcaster.sendTransform(
            (0, 0, 0),
            quaternion_from_euler(0, 0, 0),
            transform.header.stamp,
            "laser_frame",
            "base_link"
        )
        pass


    def handle_get_tf_old(self, data) -> None:
        transform = data.transforms[0]

        if transform.header.frame_id == "odom" and transform.child_frame_id == "base_link":
            self.publish_odometry(transform)

            # Get odom transform for kitti file
            self.odom_write_kitti(transform.header.stamp)
        pass


def main():
    try:
        Benchmarker()
    except rospy.ROSInterruptException:
        Benchmarker.finish_benchmarking()
        print('\n')
        pass


if __name__ == '__main__':
    main()