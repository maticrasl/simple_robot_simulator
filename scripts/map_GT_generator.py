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
from PIL import Image


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


# Returns the bounding box (min. and max. coordinates of lines)
def get_bounding_box():
    global lines
    bounding_box = lines[0].copy()

    bounding_box[0, 0] = np.min(lines[:, :, 0])
    bounding_box[0, 1] = np.min(lines[:, :, 1])
    bounding_box[1, 0] = np.max(lines[:, :, 0])
    bounding_box[1, 1] = np.max(lines[:, :, 1])

    return bounding_box


# Calculates and returns the image size in pixels (W x H)
def get_image_size(resolution, x_size, y_size):
    return np.array([x_size / resolution, y_size / resolution])


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


# Transform distances and source point into points in cartesian coordinates
def distances_to_points(distances, angles, source_point, robot_orientation):
    points = np.zeros((distances.shape[0], 2))

    for i in range(distances.shape[0]):
        points[i, 0] = distances[i] * np.cos(angles[i] + robot_orientation) + source_point[0]
        points[i, 1] = distances[i] * np.sin(angles[i] + robot_orientation) + source_point[1]

    return points


class generator:
    def __init__(self):
        # Initiate the node:
        rospy.init_node('simple_robot_generator', anonymous=True)

        self.map_resolution = rospy.get_param('~map_resolution')
        self.out_file = rospy.get_param('~out_file')
        self.image_frame = np.array([[-10, -10], [10, 10]])

        load_world(rospy.get_param('~mapfile'))

        print("Started generating Ground Truth map!")

        print(get_bounding_box())

        self.init_grid()

        self.pixelate_lines()

        pixel_plot = plt.imshow(self.grid, cmap='gray')
        plt.show(pixel_plot)
        image = Image.fromarray(self.grid)
        image.show()
        image.save(self.out_file)


    # Initiates grid datastructure as boolean 2D array
    def init_grid(self):
        self.map_size = 10 * 2                                                      # Width and height of the map (m)
        self.map_size_px = int(self.map_size / self.map_resolution)                 # Width and height of the map (px)
        self.grid_origin = self.map_size_px / 2                                     # Pixel index at map position (0, 0)
        self.grid = np.ones((self.map_size_px, self.map_size_px)).astype(bool)     # Grid
        

    # Transforms line's coordinates to grid indices
    def lines_to_grid(self, lines):
        out_lines = lines.copy()
        out_lines[:, :, 1] *= -1
        return out_lines / self.map_resolution + self.grid_origin
        # return np.floor(out_lines / self.map_resolution + self.grid_origin)


    # Bresenham algorithm for line pixelization in the grid
    # Source: Strnad, D. Obrezovanje in rasterizacija, Slide 140.
    def bresenham(self, line):
        x1 = line[0, 0].copy().astype(int)
        x2 = line[1, 0].copy().astype(int)
        y1 = line[0, 1].copy().astype(int)
        y2 = line[1, 1].copy().astype(int)

        x = x1.copy()
        y = y1.copy()

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        s1 = np.sign(x2 - x1)
        s2 = np.sign(y2 - y1)

        if dy > dx:
            temp = dx.copy()
            dx = dy.copy()
            dy = temp.copy()
            interchange = 1
        else:
            interchange = 0
        e = 2 * dy - dx
        self.grid[y, x] = False
        
        for i in range(dx):
            while e >= 0:
                if interchange == 1:
                    x = x + s1
                else:
                    y = y + s2
                e = e - 2 * dx
            if interchange == 1:
                y = y + s2
            else:
                x = x + s1
            e = e + 2 * dy
            self.grid[y, x] = False


    # Marks grid pixels, intersected by the line, as occupied
    def pixelate_lines(self):
        global lines
        grid_lines = self.lines_to_grid(lines)

        for line in grid_lines:
            self.bresenham(line)


if __name__ == '__main__':
    # source_point = [0.5, 1.5]
    # robot_orientation = pi
    # distances, wall_angles = get_scan(source_point, robot_orientation)
    # sim_distances, sim_angles = process_scan(distances, wall_angles)
    # scanned_points = distances_to_points(sim_distances, sim_angles, source_point, robot_orientation)
    # draw_world(scanned_points, source_point)
    # temp_pts = np.zeros((5, 2))
    # draw_world(temp_pts, source_point)
    
    # filename = "/home/matic/Documents/Magistrska/maps/test_map.pgm"
    # image = Image.open(filename)
    # image.show()

    try:
        generator()
    except rospy.ROSInterruptException:
        print("QUITTING\n")
        pass
    