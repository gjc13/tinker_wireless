#!/usr/bin/env python

"""
/* Original work Team '"Moe" The Autonomous Lawnmower' - Auburn University
 * Modified work Copyright 2015 Institute of Digital Communication Systems - Ruhr-University Bochum
 * Modified by: Adrian Bauer
 *
 * This program is free software; you can redistribute it and/or modify it under the terms of
 * the GNU General Public License as published by the Free Software Foundation;
 * either version 3 of the License, or (at your option) any later version.
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 * You should have received a copy of the GNU General Public License along with this program;
 * if not, see <http://www.gnu.org/licenses/>.
 *
 * This file is a modified version of the original file taken from the au_automow_common ROS stack
 **/

This ROS node is responsible for planning a path.

This node takes the field shape as a geometry_msgs/PolygonStamped
and publishes the path as a set of visualization markers.

Once the path has been generated the node can, by configuration or
a service call, start feeding path waypoints as actionlib goals to move base.
"""

import roslib
import rospy
import tf
from tf.transformations import quaternion_from_euler as qfe
from actionlib import SimpleActionClient

import numpy as np
from math import radians

from geometry_msgs.msg import PolygonStamped, Point, PoseStamped, PointStamped
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from std_srvs.srv import Empty

import shapely.geometry as geo
from detect_rssi import detect_rssi


def get_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def to_heatmap(samples, origin_x, origin_y, width, height, resolution):
    r = 2.0
    hm = -np.ones((width, height), dtype=np.int8)
    rssi = np.array([s[1] for s in samples])
    if len(rssi) == 1:
        rssi[0] = 50
    else:
        rssi = 100 * (rssi - rssi.min()) / (rssi.max() - rssi.min())
    for i in range(height):
        for j in range(width):
            x = origin_x + j * resolution
            y = origin_y + i * resolution
            diss = [get_distance(s[0], (x, y)) for s in samples]
            weights = np.array([max(1e-3, r - d) if d < r else 0 for d in diss])
            if np.sum(weights) > 0:
                weights/= weights.sum()
                hm[i, j] = (weights * rssi).sum()
    data = hm.flatten()
    grid = OccupancyGrid()
    grid.header.stamp = rospy.Time.now()
    grid.header.frame_id = "map"
    grid.info.map_load_time = grid.header.stamp
    grid.info.resolution = resolution
    grid.info.width = width
    grid.info.height = height
    grid.info.origin.position.x = origin_x
    grid.info.origin.position.y = origin_y
    grid.data = data
    return grid


class PathPlannerNode(object):
    """
    This is a ROS node that is responsible for planning and executing
    the a path through the field.
    """
    def __init__(self):
        # Setup ROS node
        rospy.init_node('path_planner')

        # ROS params
        self.cut_spacing = rospy.get_param("~coverage_spacing", 0.5)

        # Setup publishers and subscribers
        rospy.Subscriber('clicked_point', PointStamped, self.point_callback)
        self.path_marker_pub = rospy.Publisher('visualization_marker',
                                               MarkerArray, queue_size=10,
                                               latch=True)
        self.heatmap_pub = rospy.Publisher('wifi_heatmap',
                                           OccupancyGrid, queue_size=1)

        # Setup initial variables
        self.field_shape = None
        self.field_frame_id = None
        self.path = None
        self.path_status = None
        self.path_markers = None
        self.start_path_following = False
        self.robot_pose = None
        self.goal_state = None
        self.current_destination = None
        self.testing = False
        self.current_distance = None
        self.previous_destination = None
        self.clear_costmaps = rospy.ServiceProxy('move_base/clear_costmaps', Empty)
        self.just_reset = False
        self.timeout = False
        self.clicked_points = []
        self.tf_listener = tf.TransformListener()
        self.move_base_client = SimpleActionClient('move_base', MoveBaseAction)
        self.move_base_client.wait_for_server()
        self.samples = []

        # Spin until shutdown or we are ready for path following
        rate = rospy.Rate(10.0)
        self.update_now()
        rospy.loginfo("Waiting for scan area input")
        while not rospy.is_shutdown():
            rate.sleep()
            if self.start_path_following:
                self.follow_path(self.clicked_points)
                self.clicked_points = []

    def get_xy(self):
        while True:
            try:
                (trans, rot) = self.tf_listener.lookupTransform('/map', '/base_link', rospy.Time(0))
                return trans[:2]
            except:
                continue

    def point_callback(self, msg):
        if self.start_path_following:
            return
        msg.header.stamp = rospy.Time(0)
        point = msg.point
        p_odom = self.tf_listener.transformPoint('/odom', msg)
        self.field_frame_id = msg.header.frame_id
        self.clicked_points.append([point.x, point.y, 0])
        self.visualize_path_as_marker(self.clicked_points) 
        if len(self.clicked_points) >= 2 and get_distance(self.clicked_points[-2], self.clicked_points[-1]) < 1:
            self.clicked_points = self.clicked_points[:-1]
            self.start_path_following = True

    def update_now(self):
        x, y = self.get_xy()
        rssi_sample = detect_rssi()
        self.samples.append(((x, y), rssi_sample['FUTURE_ANIME_5G']))
        rospy.loginfo('FUTURE_ANIME_5G: {}'.format(rssi_sample['FUTURE_ANIME_5G']))
        self.heatmap_pub.publish(to_heatmap(self.samples, -20, -20, 400, 400, 0.1))


    def follow_path(self, points):
        for p in points:
            self.move_to_point(p)
            self.update_now()


    def visualize_path_as_marker(self, path, path_status=None):
        """
        Publishes visualization Markers to represent the planned path.

        Publishes the path as a series of spheres connected by lines.
        The color of the spheres is set by the path_status parameter,
        which is a list of strings of which the possible values are in
        ['not_visited', 'visiting', 'visited'].
        """
        # Get the time
        now = rospy.Time.now()
        # If self.path_markers is None, initialize it
        if self.path_markers == None:
            self.path_markers = MarkerArray()
        self.path_markers = MarkerArray()
        line_strip_points = []
        # Create the waypoint markers
        for index, waypoint in enumerate(path):
            waypoint_marker = Marker()
            waypoint_marker.header.stamp = now
            waypoint_marker.header.frame_id = self.field_frame_id
            waypoint_marker.ns = "waypoints"
            waypoint_marker.id = index
            waypoint_marker.type = Marker.ARROW
            if index == 0:
                waypoint_marker.type = Marker.CUBE
            waypoint_marker.action = Marker.MODIFY
            waypoint_marker.scale.x = 0.5
            waypoint_marker.scale.y = 0.1
            waypoint_marker.scale.z = 0.1
            point = Point(waypoint[0], waypoint[1], 0)
            waypoint_marker.pose.position = point
            # Store the point for the line_strip marker
            line_strip_points.append(point)
            # Set the heading of the ARROW
            quat = qfe(0, 0, waypoint[2])
            waypoint_marker.pose.orientation.x = quat[0]
            waypoint_marker.pose.orientation.y = quat[1]
            waypoint_marker.pose.orientation.z = quat[2]
            waypoint_marker.pose.orientation.w = quat[3]
            # Color is based on path_status
            if path_status == None:
                waypoint_marker.color = ColorRGBA(1,0,0,0.5)
            else:
                status = path_status[index]
                if status == 'not_visited':
                    waypoint_marker.color = ColorRGBA(1,0,0,0.5)
                elif status == 'visiting':
                    waypoint_marker.color = ColorRGBA(0,1,0,0.5)
                elif status == 'visited':
                    waypoint_marker.color = ColorRGBA(0,0,1,0.5)
                else:
                    rospy.err("Invalid path status.")
                    waypoint_marker.color = ColorRGBA(1,1,1,0.5)
            # Put this waypoint Marker in the MarkerArray
            self.path_markers.markers.append(waypoint_marker)
        # Create the line_strip Marker which connects the waypoints
        line_strip = Marker()
        line_strip.header.stamp = now
        line_strip.header.frame_id = self.field_frame_id
        line_strip.ns = "lines"
        line_strip.id = 0
        line_strip.type = Marker.LINE_STRIP
        line_strip.action = Marker.ADD
        line_strip.scale.x = 0.1
        line_strip.color = ColorRGBA(0,0,1,0.5)
        line_strip.points = line_strip_points
        self.path_markers.markers.append(line_strip)
        # Publish the marker array
        self.path_marker_pub.publish(self.path_markers)

    def move_to_point(self, point):
        now_x, now_y = self.get_xy()
        #first turn towards
        dx = point[0] - now_x
        dy = point[1] - now_y
        from math import atan2, pi
        theta = atan2(dy, dx)
        self.move_to_pose((now_x, now_y, theta))
        self.move_to_pose((point[0], point[1], theta))

    def move_to_pose(self, waypoint):
        destination = MoveBaseGoal()
        destination.target_pose.header.frame_id = self.field_frame_id
        destination.target_pose.header.stamp = rospy.Time.now()
        destination.target_pose.pose.position.x = waypoint[0]
        destination.target_pose.pose.position.y = waypoint[1]
        quat = qfe(0, 0, waypoint[2])
        destination.target_pose.pose.orientation.x = quat[0]
        destination.target_pose.pose.orientation.y = quat[1]
        destination.target_pose.pose.orientation.z = quat[2]
        destination.target_pose.pose.orientation.w = quat[3]
        rospy.loginfo("Sending waypoint (%f, %f)@%f" % tuple(waypoint))
        self.move_base_client.send_goal(destination)
        self.move_base_client.wait_for_result(rospy.Duration.from_sec(30.0))
        self.move_base_client.cancel_all_goals()


if __name__ == '__main__':
    ppn = PathPlannerNode()
