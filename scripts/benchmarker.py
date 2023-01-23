#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist, Point, Pose, Quaternion, Vector3, PointStamped, PoseStamped
from nav_msgs.msg import Odometry
from simple_robot_simulator.srv import Benchmark
from math import nan, sqrt
import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import numpy as np
import time
import math


# message = "0"
scan_angle = 270
scan_step = 3
scan_angle_rad = np.deg2rad(scan_angle)
scan_step_rad = np.deg2rad(scan_step)
scan_average = 1
laser_frequency = 1
scan_range_min = 0.001
scan_range_max = 1.5


class benchmarker:
    def __init__(self):
        rospy.init_node('benchmarker')

        self.kitti_path = rospy.get_param('~kitti_path', '/home/matic/Documents/Magistrska/Benchmarking')
        self.mapfile = rospy.get_param('~mapfile')

        # self.gt_file = open(self.kitti_path + "/ground_truth/" + self.mapfile + ".txt", 'w')
        self.odom_file = open(self.kitti_path + "/odom/" + self.mapfile + ".txt", 'w')

        #rospy.Subscriber('/clicked_point', PointStamped, self.get_clicked_point)
        # rospy.Subscriber('/ground_truth', Odometry, self.handle_get_gt, self.gt_file)
        rospy.Subscriber('/odom', Odometry, self.handle_get_odom, self.odom_file)
        self.tf_listener = tf.TransformListener()
        self.odom_broadcaster = tf.TransformBroadcaster()
        s = rospy.Service('get_benchmarking', Benchmark, self.handle_get_benchmarking)
        self.global_benchmark = 0
        self.local_benchmark = 0
        self.N = 0

        # (self.last_trans, self.last_rot) = self.tf_listener.lookupTransform('ground_truth', 'base_link', rospy.Time(0))
        self.publish_first_tf()

        rospy.spin()


    def publish_first_tf(self):
        timestamp = rospy.Time.now()

        print("Printing first tf transform")

        self.odom_broadcaster.sendTransform(
            (rospy.get_param("initial_pos_x", "0"), rospy.get_param("initial_pos_y", "0"), 0),
            quaternion_from_euler(0, 0, float(rospy.get_param("initial_pos_a", "0"))),
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


    def publish_tf(self, odom):
        print("publishing tf")

        timestamp = rospy.Time.now()

        self.odom_broadcaster.sendTransform(
            (odom.pose.pose.position.x, odom.pose.pose.position.y, 0),
            (0, 0, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w),
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


    def handle_get_gt(self, data, out_file):
        trans = [data.pose.pose.position.x, data.pose.pose.position.y]
        rot = [data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w]
        th = euler_from_quaternion(rot)[2]

        matrix = np.array([[1, 0, 0, trans[0]], [0, 1, 0, trans[1]], [0, 0, 1, 0], [0, 0, 0, 1]])
        matrix_rot = np.array([[math.cos(th), -math.sin(th), 0, 0], [math.sin(th), math.cos(th), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        matrix = matrix @ matrix_rot

        kitti_line = matrix[:3, :].reshape(12)
        kitti_line_string = " ".join([str(i) for i in kitti_line])

        print(kitti_line_string)

        out_file.write(kitti_line_string)
        out_file.write('\n')


    def handle_get_odom(self, data, out_file):
        (trans, rot) = self.tf_listener.lookupTransform('/base_link', '/world', rospy.Time.now())
        print(trans)
        print(rot)

        trans = [data.pose.pose.position.x, data.pose.pose.position.y]
        rot = [data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w]
        th = euler_from_quaternion(rot)[2]

        matrix = np.array([[1, 0, 0, trans[0]], [0, 1, 0, trans[1]], [0, 0, 1, 0], [0, 0, 0, 1]])
        matrix_rot = np.array([[math.cos(th), -math.sin(th), 0, 0], [math.sin(th), math.cos(th), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        matrix = matrix @ matrix_rot

        kitti_line = matrix[:3, :].reshape(12)
        kitti_line_string = " ".join([str(i) for i in kitti_line])

        print(kitti_line_string)

        out_file.write(kitti_line_string)
        out_file.write('\n')

        self.publish_tf(data)


    def handle_get_benchmarking(self, req):
        if req.type == 0:
            return self.global_benchmark / max(self.N, 1)
        else:
            return self.local_benchmark / max(self.N, 1)


    def finish_benchmarking(self):
        self.gt_file.close()
        self.odom_file.close()


if __name__ == '__main__':
    message = "0"

    try:
        benchmarker()
    except rospy.ROSInterruptException:
        benchmarker.finish_benchmarking()
        print('\n')
        pass