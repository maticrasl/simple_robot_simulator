#!/usr/bin/env python
from cmath import inf
from json.encoder import INFINITY
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist, Point, Pose, Quaternion, Vector3, PointStamped
from simple_robot_driving.msg import SimpleRobotDriveMsg, ThreeAngles
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from math import sin, cos, sqrt, pi, acos, exp, floor
import tf
import numpy as np
from tf.transformations import quaternion_from_euler
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt


scan_angle = 270
scan_step = 3
scan_angle_rad = np.deg2rad(scan_angle)
scan_step_rad = np.deg2rad(scan_step)
scan_average = 1
scan_range_min = 0.001
scan_range_max = 1.5

# ODOMETRY GROUND_TRUTH ERRORS  (odometry errors in driving on actual position)
drive_dist_err_mean = 0.00                  # Mean of error in distance when driving
drive_dist_err_std = 0.00                   # Standard deviation of error in distance when driving
drive_ang_err_mean = 0.0                    # Mean of error in angle when driving
drive_ang_err_std = 0.00                    # Standard deviation of error in angle when driving
rot_dist_err_mean = 0.0                     # Mean of error in distance when rotating
rot_dist_err_std = 0.000                    # Standard deviation of error in distance when rotating
rot_ang_err_mean = 0.0                      # Mean of error in angle when rotating
rot_ang_err_std = 0.00                      # Standard deviation of error in angle when rotating

# ODOMETRY SIM ERRORS  (odometry errors, added to the actual position - errors in angle measurement)
sim_drive_ang_err_std = 0.00                # This gets multiplied by real meters driven
sim_rot_ang_err_std = 0.00                  # This gets multiplied by rads of real rotation


# Lines of the map
lines = np.array((1, 2, 2))


def str2float(str):
    return float(str)


# Rotates the vector by an angle
def rotate(vector, th):
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return np.matmul(R, vector)


# Loads the world from a file
def load_world(filename):
    global lines
    # Get the root node
    root = ET.parse(filename).getroot()
    print("Loading world ...")
    links = root.findall('model/link')
    lines = np.zeros((len(links) * 4, 2, 2))
    for i in range(len(links)):
        collision = links[i].find('collision')
        pose = np.array(list(map(str2float, links[i].find('pose').text.split(' '))))
        size = np.array(list(map(str2float, collision.find('geometry/box/size').text.split(' '))))
        # Create lines of the current wall
        lines[i * 4 + 0, 0, :] = np.reshape(rotate(np.array([[-size[0] / 2.0], [-size[1] / 2.0]]), pose[5]), lines[0, 0, :].shape) + pose[:2]
        lines[i * 4 + 0, 1, :] = np.reshape(rotate(np.array([[size[0] / 2.0], [-size[1] / 2.0]]), pose[5]), lines[0, 0, :].shape) + pose[:2]
        lines[i * 4 + 1, 0, :] = np.reshape(rotate(np.array([[size[0] / 2.0], [-size[1] / 2.0]]), pose[5]), lines[0, 0, :].shape) + pose[:2]
        lines[i * 4 + 1, 1, :] = np.reshape(rotate(np.array([[size[0] / 2.0], [size[1] / 2.0]]), pose[5]), lines[0, 0, :].shape) + pose[:2]
        lines[i * 4 + 2, 0, :] = np.reshape(rotate(np.array([[size[0] / 2.0], [size[1] / 2.0]]), pose[5]), lines[0, 0, :].shape) + pose[:2]
        lines[i * 4 + 2, 1, :] = np.reshape(rotate(np.array([[-size[0] / 2.0], [size[1] / 2.0]]), pose[5]), lines[0, 0, :].shape) + pose[:2]
        lines[i * 4 + 3, 0, :] = np.reshape(rotate(np.array([[-size[0] / 2.0], [size[1] / 2.0]]), pose[5]), lines[0, 0, :].shape) + pose[:2]
        lines[i * 4 + 3, 1, :] = np.reshape(rotate(np.array([[-size[0] / 2.0], [-size[1] / 2.0]]), pose[5]), lines[0, 0, :].shape) + pose[:2]


# Displays the lines and the scanned points in a pyplot
def draw_world(scanned_points, source_point):
    global lines
    print(lines.shape)
    for line in lines:
        plt.plot([line[0, 0], line[1, 0]], [line[0, 1], line[1, 1]], color="black")
    plt.scatter(np.array(scanned_points[:, 0]), np.array(scanned_points[:, 1]), color="blue")
    plt.scatter(source_point[0], source_point[1], color="red")
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.show()


# Returns the point that is the intersection of the two lines
def linexline(line1, line2):
    x1 = line1[0, 0]
    y1 = line1[0, 1]
    x2 = line1[1, 0]
    y2 = line1[1, 1]
    x3 = line2[0, 0]
    y3 = line2[0, 1]
    x4 = line2[1, 0]
    y4 = line2[1, 1]

    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))

    if u >= 0 and u <= 1.0 and t >= 0 and t <= 1.0:
        xi = ((x3 + u * (x4 - x3)) + (x1 + t * (x2 - x1))) / 2
        yi = ((y3 + u * (y4 - y3)) + (y1 + t * (y2 - y1))) / 2
    else:
        xi = np.inf
        yi = np.inf

    return np.array([xi, yi])


# Rotates the point by the given angle and returns the new point coordinates.
def rotate_point(point, angle):
    new_point = (point[0] * cos(angle) - point[1] * sin(angle), point[0] * sin(angle) + point[1] * cos(angle))
    return new_point


# Returns the eucledean distance between two points.
def get_distance(len1, len2):
    distance = np.sqrt((len1[0] - len2[0])**2 + (len1[1] - len2[1])**2)
    return distance


# Returns the angle between the two lines
def get_angle(line1, line2):
    vector1 = np.array([line1[1, 0] - line1[0, 0], line1[1, 1] - line1[0, 1]])
    vector2 = np.array([line2[1, 0] - line2[0, 0], line2[1, 1] - line2[0, 1]])
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    mod_of_vector1 = sqrt(vector1[0]**2 + vector1[1]**2) * sqrt(vector2[0]**2 + vector2[1]**2)
    angle = acos(dot_product / mod_of_vector1)
    return angle


# Returns the distances and wall angles of the perfect dense scan
def get_scan(source_point, robot_orientation):
    # Initiate angles
    ray_angles = np.deg2rad(np.arange(0.0, 360.0, 0.5))
    n = ray_angles.shape[0]

    # If there are no walls, return empyt scan
    if lines.shape[0] == 0:
        return np.zeros((n, 1)), np.zeros((n, 1))
    
    # Scan vectors of the sensor
    ray_basic_vector = np.array([[1], [0]])
    ray_vectors = np.zeros((n, 2))

    # Rotate the vectors in all directions by their corresponding angle
    ray_vector_shape = ray_vectors[0, :].shape
    for i in range(n):
        ray_vectors[i, :] = np.reshape(rotate(ray_basic_vector, ray_angles[i] + robot_orientation), ray_vector_shape)

    # Scan lines
    ray_lines = np.zeros((n, 2, 2))

    ray_lines[:, 0, 0] = source_point[0]
    ray_lines[:, 0, 1] = source_point[1]
    ray_lines[:, 1, 0] = source_point[0] + (ray_vectors[:, 0] * 100)
    ray_lines[:, 1, 1] = source_point[1] + (ray_vectors[:, 1] * 100)

    # Points in which the ray_lines intersect their first wall_line 
    intersection_points = np.zeros((n, 2))
    intersection_point_distances = np.zeros((n, 1))
    intersection_point_wall_angles = np.zeros((n, 1))

    # Search for intersection points
    for i in range(n):
        intersection_points[i] = linexline(ray_lines[i], lines[0])
        best_distance = get_distance(source_point, intersection_points[i])
        for j in range(lines.shape[0]):
            temp_ip = linexline(ray_lines[i], lines[j])
            if get_distance(source_point, temp_ip) <= best_distance:
                best_distance = get_distance(source_point, temp_ip)
                intersection_point_wall_angles[i] = get_angle(ray_lines[i, :, :], lines[j, :, :])
                intersection_points[i] = temp_ip.copy()
        intersection_point_distances[i] = best_distance

    # Display all intersection_points on the screen
    # plt.scatter(np.array(intersection_points[:, 0]), np.array(intersection_points[:, 1]), color="red")
    # for line in lines:
    #     plt.plot([line[0, 0], line[1, 0]], [line[0, 1], line[1, 1]], color="black")
    # plt.gca().set_aspect('equal', adjustable='datalim')
    # plt.show()

    # draw_world(intersection_points, source_point)

    return intersection_point_distances, intersection_point_wall_angles


# Processes the raw distances and angles into the expected sensor output and returns "sensor's" distances
def process_scan(distances, wall_angles):
    sim_distances = np.zeros(91)                                    # 91 values for 270° with a 3° step
    sim_angles = np.deg2rad(np.arange(-135.0, 138.0, 3.0))          # Corresponding angles for each distance

    # Process each distance individually
    for i in range(91):
        I = (i * 6) - 270 + 720                                     # Index of first used simulated distnce for current processed distance
        
        # Go through every simulated scan value, needed to construct the current processed distance
        sim_distances[i] = distances[I % 720]

    return sim_distances, sim_angles                                # Returns processed distances and angles (angles are not really used in the ROS process, only to plot the points)


# Transform distances and source point into points in cartesian coordinates
def distances_to_points(distances, angles, source_point, robot_orientation):
    points = np.zeros((distances.shape[0], 2))

    for i in range(distances.shape[0]):
        points[i, 0] = distances[i] * np.cos(angles[i] + robot_orientation) + source_point[0]
        points[i, 1] = distances[i] * np.sin(angles[i] + robot_orientation) + source_point[1]

    return points


class simulator:
    def __init__(self):
        self.pos = np.array([float(rospy.get_param('initial_pos_x')), float(rospy.get_param('initial_pos_y')), float(rospy.get_param('initial_pos_a'))])                                                   # Simulated odometry position in a simulator
        self.ground_truth = np.array([float(rospy.get_param('initial_pos_x')), float(rospy.get_param('initial_pos_y')), float(rospy.get_param('initial_pos_a'))])                                          # Ground truth in a simulator

        # Topic subscribers:
        rospy.Subscriber('/make_drive', SimpleRobotDriveMsg, self.get_make_drive)
        rospy.Subscriber('/make_scan', String, self.get_make_scan)

        # Topic publishers:
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size = 50)
        self.scan_pub = rospy.Publisher('sim_scan', LaserScan, queue_size=50)
        self.odom_pub = rospy.Publisher('sim_odom', Odometry, queue_size=50)
        self.three_angles_pub = rospy.Publisher('sim_angles', ThreeAngles, queue_size=10)
        self.ground_truth_pub = rospy.Publisher('ground_truth', Odometry, queue_size=50)
        self.collision_pub = rospy.Publisher('sim_collision', String, queue_size=5)
        self.odom_broadcaster = tf.TransformBroadcaster()

        # Initiate the node:
        rospy.init_node('simple_robot_simulator', anonymous=True)

        load_world(rospy.get_param('~mapfile'))

        print("Started the simulator!")
        self.start_publishing_ground_truth()


    # Called every time the dist_ang message is published by the move_base node (local planner).
    def get_make_drive(self, data):
        rotation = data.rotation
        distance = data.distance

        print("SIMULATOR RECEIVED: rot(", rotation, "), dist(", distance, ")")

        # Returned angles (starting, after_rotation, after_translation)
        out_angles = [0.0, 0.0, 0.0]
        out_angles[0] = self.pos[2]             # Starting angle

        # Rotate, add error and aditional measurement error to pos and real_pos

        # Position increment in translation during rotation
        pos_increment = [
            np.random.normal(rot_dist_err_mean, rot_dist_err_std) * rotation,
            np.random.normal(rot_dist_err_mean, rot_dist_err_std) * rotation,
            rotation + (np.random.normal(rot_ang_err_mean, rot_ang_err_std) * rotation)
        ]
        self.ground_truth[:] += pos_increment[:]
        self.pos[2] += pos_increment[2] + np.random.normal(0.0, np.abs(sim_rot_ang_err_std * rotation))         # self.pos[2] also gets an aditional, "measurement" error
        # print("\tGround truth after rotation:\t", self.ground_truth)

        out_angles[1] = self.pos[2]             # Measured angle after rotation

        # Determine the final angle, then translate with the average of the two angles

        # Calculation of the ending angle of the ground_truth
        real_ending_angle = self.ground_truth[2] + (np.random.normal(drive_ang_err_mean, drive_ang_err_std) * distance)
        travel_angle = (self.ground_truth[2] + real_ending_angle) / 2.0                                         # Travel angle is the average of the two
        
        # Position increment during translation
        pos_increment = [
            distance * cos(travel_angle) * np.random.normal(1.0 + drive_dist_err_mean, drive_dist_err_std),
            distance * sin(travel_angle) * np.random.normal(1.0 + drive_dist_err_mean, drive_dist_err_std),
            0.0
        ]

        # Test if there was a collision during the current movement
        if self.test_for_collision(pos_increment, travel_angle):
            self.publish_colision()
            # return

        self.ground_truth[:] += pos_increment[:]
        self.pos[2] += (real_ending_angle - self.ground_truth[2]) + (np.random.normal(0.0, sim_drive_ang_err_std) * distance)
        self.ground_truth[2] = real_ending_angle
        print("\tGround truth after movement:\t", self.ground_truth)

        out_angles[2] = self.pos[2]             # Measured angle after translation

        # Publish the message with the three angles.
        self.publish_three_angles(out_angles)


    # Performs scan and publishes the sensor data
    def get_make_scan(self, data):
        distances, wall_angles = get_scan(self.ground_truth[:2], self.ground_truth[2])      # Get real distances of the surroundings
        sim_distances, sim_angles = process_scan(distances, wall_angles)                    # Turn real distances into the simulated sensor values
        sim_points = distances_to_points(sim_distances, sim_angles, self.ground_truth[:2], self.ground_truth[2])
        # draw_world(sim_points, self.ground_truth[:2])
        self.publish_laser_scan(sim_distances)                                              # Publish the simulated scan distances to the ROS topic


    # Publish three movement angles to the topic (function is called after the move)
    def publish_three_angles(self, out_angles):
        three_angles = ThreeAngles()
        three_angles.first = out_angles[0]
        three_angles.second = out_angles[1]
        three_angles.third = out_angles[2]
        self.three_angles_pub.publish(three_angles)
        

    # Publishes the simulated laser scan
    def publish_laser_scan(self, distances):
        global scan_angle_rad
        global scan_step_rad
        global scan_range_min
        global scan_range_max

        # print(distances.shape)

        scan = LaserScan()
        scan.header.stamp = rospy.Time.now()
        scan.header.frame_id = 'laser_frame'
        scan.angle_min = -scan_angle_rad / 2.0
        scan.angle_max = scan_angle_rad / 2.0
        scan.angle_increment = scan_step_rad
        scan.time_increment = 0.00001
        scan.range_min = scan_range_min
        scan.range_max = scan_range_max

        scan.ranges = []
        scan.intensities = []
        for d in distances:
            scan.ranges.append(d)

        self.scan_pub.publish(scan)
        return


    # Periodically publishes the ground truth
    def start_publishing_ground_truth(self):
        r = rospy.Rate(5.0)

        # Infinite loop, until the program gets shut down
        while not rospy.is_shutdown():
            # print(":")
            timestamp = rospy.Time.now()
            odom_quat = quaternion_from_euler(0, 0, self.ground_truth[2])
            
            # Publish the transform over tf
            self.odom_broadcaster.sendTransform(
                (self.ground_truth[0], self.ground_truth[1], 0.0),
                odom_quat,
                timestamp,
                "ground_truth",
                "map"
            )

            # Publish the odometry message over ROS
            odom = Odometry()
            odom.header.stamp = timestamp
            odom.header.frame_id = "map"

            odom.pose.pose = Pose(Point(self.ground_truth[0], self.ground_truth[1], 0.), Quaternion(*odom_quat))
            odom.child_frame_id = "ground_truth"
            odom.twist.twist = Twist(Vector3(0, 0, 0), Vector3(0, 0, 0))

            self.ground_truth_pub.publish(odom)
            r.sleep()


    # Publishes odometry (not used)
    def publish_odometry(self):
        odom_quat = quaternion_from_euler(0, 0, self.pos[2])
        timestamp = rospy.Time.now()

        self.odom_broadcaster.sendTransform(
            (0, 0, 0),
            quaternion_from_euler(0, 0, 0),
            timestamp,
            "laser_frame",
            "base_link"
        )

        # Publish the odometry message over ROS
        odom = Odometry()
        odom.header.stamp = timestamp
        odom.header.frame_id = "odom"

        odom.pose.pose = Pose(Point(self.pos[0], self.pos[1], 0.), Quaternion(*odom_quat))
        odom.child_frame_id = "base_link"
        odom.twist.twist = Twist(Vector3(0, 0, 0), Vector3(0, 0, 0))

        self.odom_pub.publish(odom)


    # Tests if there was a collision during the move
    def test_for_collision(self, pos_increment, travel_angle):
        travel_points = np.array([[0.035, 0.065], [0.035, -0.065], [-0.08, -0.065], [-0.08, 0.065]])            # Robot points
        for i in range(travel_points.shape[0]):
            travel_points[i] = rotate_point(travel_points[i], travel_angle)                                     # Rotated robot points
            # print(travel_points, "\n", self.ground_truth)
            travel_points[i, 0] += self.ground_truth[0]                                                         # Rotated translated robot points
            travel_points[i, 1] += self.ground_truth[1]                                                         # Rotated translated robot points

        travel_points[:2, 0] += pos_increment[0]
        travel_points[:2, 1] += pos_increment[1]                                                                # Travel points

        # Bounding lines of the robot during the move
        travel_lines = np.array([[travel_points[0], travel_points[1]], [travel_points[1], travel_points[2]], [travel_points[2], travel_points[3]], [travel_points[3], travel_points[0]]])

        for travel_l in travel_lines:
            for line in lines:
                int_point = linexline(travel_l, line)
                if int_point[0] != inf:
                    for line in lines:
                        plt.plot([line[0, 0], line[1, 0]], [line[0, 1], line[1, 1]], color="black")
                    for travel_l in travel_lines:
                        plt.plot([travel_l[0, 0], travel_l[1, 0]], [travel_l[0, 1], travel_l[1, 1]], color="green")
                    plt.gca().set_aspect('equal', adjustable='datalim')
                    plt.scatter(int_point[0], int_point[1], color="red")
                    print("COLLISION! ", int_point)
                    plt.show()
                    return True
        return False


    # Publishes the collision message to the ros topic
    def publish_colision(self):
        self.collision_pub.publish("COLLISION")


if __name__ == '__main__':
    # source_point = [0.5, 1.5]
    # robot_orientation = pi
    # distances, wall_angles = get_scan(source_point, robot_orientation)
    # sim_distances, sim_angles = process_scan(distances, wall_angles)
    # scanned_points = distances_to_points(sim_distances, sim_angles, source_point, robot_orientation)
    # draw_world(scanned_points, source_point)
    # temp_pts = np.zeros((5, 2))
    # draw_world(temp_pts, source_point)
    try:
        simulator()
    except rospy.ROSInterruptException:
        print("QUITTING\n")
        pass
    