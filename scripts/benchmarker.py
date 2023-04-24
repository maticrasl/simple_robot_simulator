#!/usr/bin/env python
import rospy

import math
import numpy as np

from geometry_msgs.msg import Point, TransformStamped, Vector3
from nav_msgs.msg import Odometry
import os
import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from tf2_msgs.msg import TFMessage
from typing import List


class Benchmarker:
    def __init__(self):
        rospy.init_node('benchmarker')

        self.benchmarking_path = rospy.get_param('~kitti_path', '/home/matic/Documents/Magistrska/Benchmarking')
        self.mapfile = rospy.get_param('~mapfile')
        self.generate_gt = int(rospy.get_param('~generate_gt', "0"))

        self.transformations: List[TransformStamped] = []

        self.kitti_path = f"{self.benchmarking_path}/kitti_files/{self.mapfile}/"
        os.makedirs(self.kitti_path, exist_ok=True)

        if self.generate_gt == 1:
            self.gt_file = open(self.kitti_path + "ground_truth.txt", 'w')
            rospy.Subscriber('/ground_truth', Odometry, self.gt_write_kitti, queue_size=50)
        else:
            rospy.Subscriber('/tf_old', TFMessage, self.handle_get_tf_old, queue_size=50)
            rospy.Subscriber('/tf', TFMessage, self.handle_get_tf, queue_size=50)

        self.tf_listener = tf.TransformListener()
        self.odom_broadcaster = tf.TransformBroadcaster()

        rospy.on_shutdown(self.finish_benchmarking)

        self.publish_first_tf()

        rospy.spin()


    def publish_first_tf(self) -> None:
        timestamp = rospy.Time.now()

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


    def odom_write_kitti(self, transform_matrix: np.ndarray) -> None:
        """ Get SLAM position and write it to kitti file. """
        out_file = self.odom_file

        kitti_line = transform_matrix[:3, :].reshape(12)
        kitti_line_string = " ".join([str(i) for i in kitti_line])

        out_file.write(kitti_line_string)
        out_file.write('\n')


    def publish_odometry(self, transform: TransformStamped) -> None:
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


    def handle_get_tf_old(self, data: TFMessage) -> None:
        transform: TransformStamped = data.transforms[0]

        if transform.header.frame_id == "odom" and transform.child_frame_id == "base_link":
            self.publish_odometry(transform)


    def handle_get_tf(self, data: TFMessage) -> None:
        transform: TransformStamped = data.transforms[0]

        if transform.header.frame_id == "odom" and transform.child_frame_id == "base_link":
            self.transformations.append(transform)
        elif transform.header.frame_id == "world" and transform.child_frame_id == "odom":
            if len(self.transformations) == 0:
                self.transformations.append(transform)
            elif self.transformations[-1].header.frame_id == "odom":
                self.transformations.append(transform)
            else:
                self.transformations[-1] = transform


    def generate_transform_matrix(self, translation: Vector3, th: float) -> np.ndarray:
        matrix = np.array([[1, 0, 0, translation.x], [0, 1, 0, translation.y], [0, 0, 1, 0], [0, 0, 0, 1]])
        matrix_rot = np.array([[math.cos(th), -math.sin(th), 0, 0], [math.sin(th), math.cos(th), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        matrix: np.ndarray = matrix @ matrix_rot
        return matrix


    def generate_transform_matrix_from_transforms(self, robot_pos: TransformStamped, odom_transform: TransformStamped) -> np.ndarray:
        robot_rotation = robot_pos.transform.rotation
        robot_translation = robot_pos.transform.translation
        robot_th = euler_from_quaternion([0, 0, robot_rotation.z, robot_rotation.w])[2]
        robot_matrix = self.generate_transform_matrix(robot_translation, robot_th)

        odom_rotation = odom_transform.transform.rotation
        odom_translation = odom_transform.transform.translation
        odom_th = euler_from_quaternion([0, 0, odom_rotation.z, odom_rotation.w])[2]
        odom_matrix = self.generate_transform_matrix(odom_translation, odom_th)

        matrix = odom_matrix @ robot_matrix
        return matrix


    def generate_transform_matrices_from_transforms(self) -> List[np.ndarray]:
        transform_matrices: List[np.ndarray] = []

        # Generate the default odom_transform (0)
        odom_transform: TransformStamped = TransformStamped()
        
        # Iterate through all transformations
        for i in range(len(self.transformations)):
            if self.transformations[i].child_frame_id == "odom":
                odom_transform = self.transformations[i]
            else:
                if i == len(self.transformations) - 1:
                    transform_matrices.append(self.generate_transform_matrix_from_transforms(self.transformations[i], odom_transform))
                elif self.transformations[i + 1].child_frame_id == "base_link":
                    transform_matrices.append(self.generate_transform_matrix_from_transforms(self.transformations[i], odom_transform))
                else:
                    transform_matrices.append(self.generate_transform_matrix_from_transforms(self.transformations[i], self.transformations[i + 1]))

        return transform_matrices


    def finish_benchmarking(self) -> None:
        if self.generate_gt:
            self.gt_file.close()
        else:
            self.odom_file = open(self.kitti_path + "odom.txt", 'w')
            # Calculate all of the kitti lines (matrices)
            transform_matrices: List[np.ndarray] = self.generate_transform_matrices_from_transforms()
            # Write all kitti lines to the kitti_file
            for transform_matrix in transform_matrices:
                self.odom_write_kitti(transform_matrix)

            self.odom_file.close()


def main():
    try:
        Benchmarker()
    except rospy.ROSInterruptException:
        print('\n')


if __name__ == '__main__':
    main()