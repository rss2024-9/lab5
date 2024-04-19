from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, Pose, TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros import TransformBroadcaster
from sensor_msgs.msg import LaserScan

from tf_transformations import euler_from_quaternion, quaternion_from_euler
from threading import Lock
from time import time

import numpy as np
import logging

from rclpy.node import Node
import rclpy

assert rclpy


class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")

        self.declare_parameter('simulation', True)
        self.simulation = self.get_parameter('simulation').get_parameter_value().bool_value

        self.get_logger().info(f"Running in {'Simulation' if self.simulation else 'Real Life'}")

        self.declare_parameter('particle_filter_frame', "default")
        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value

        # Particles initialization constants
        self.declare_parameter('num_particles', 1000)
        self.declare_parameter('particle_spread', 1.0)

        self.num_particles = self.get_parameter('num_particles').get_parameter_value().integer_value
        self.particle_spread = self.get_parameter('particle_spread').get_parameter_value().double_value
        self.particles = np.zeros((self.num_particles, 3))

        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.
        
        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('scan_topic', "/scan")

        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value

        self.laser_sub = self.create_subscription(LaserScan, scan_topic, self.laser_callback, 1)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odom_callback, 1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self.pose_callback, 1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.
        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)

        # Visualization
        self.viz_pub = self.create_publisher(PoseArray, "/particles", 1)
        self.viz_timer = self.create_timer(1/20, self.visualize_particles)

        # Initialize the models
        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)

        # Synchronization primitive
        self.lock = Lock()

        # Transformations from "/laser" to "/base_link"
        # For some reason, laser scans are published on this frame in the robot
        if not self.simulation:
            self.tf_static_pub = StaticTransformBroadcaster(self)

            # Create an identity transformation, which is technically incorrect because
            # the LiDAR isn't exactly at the car's position but... good enough
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = "base_link"
            t.child_frame_id = "laser"

            self.tf_static_pub.sendTransform(t)

        # Outside of simulation, the robot model isn't updated so we do it ourselves
        if not self.simulation:
            self.tf_pub = TransformBroadcaster(self)


        # arbitrary standard dev distance for exp eval
        self.std_dev = 0.0
        self.exp_eval = True
        with open('particle_std_dev.txt', 'w') as f:
            f.truncate(0)

        self.get_logger().info("=============+READY+=============")
    
    def laser_callback(self, scan: LaserScan):
        """
        From the instructions:
        Whenever you get sensor data use the sensor model to compute the particle probabilities.
        Then resample the particles based on these probabilities.

        Anytime the particles are update (either via the motion or sensor model), determine the
        "average" (term used loosely) particle pose and publish that transform.
        """
        with self.lock:
            probabilities = self.sensor_model.evaluate(self.particles, np.array(scan.ranges))
            
            # Workaround for when the map isn't ready yet
            if probabilities is None:
                return
            
            # Temperature and normalization
            probabilities **= 0.4
            probabilities /= np.sum(probabilities)

            # Resampling
            idx = np.random.choice(self.num_particles, self.num_particles, p=probabilities)
            self.particles = self.particles[idx, :]

            self.publish_transform()

    def odom_callback(self, odom: Odometry):
        """
        From the instructions:
        Whenever you get odometry data use the motion model to update the particle positions.

        Anytime the particles are update (either via the motion or sensor model), determine the
        "average" (term used loosely) particle pose and publish that transform.
        """
        now = odom.header.stamp.sec + odom.header.stamp.nanosec * 1e-9
        try:
            dt = now - self.last_odom_stamp
        except AttributeError:
            dt = 0
        self.last_odom_stamp = now

        velocity = odom.twist.twist.linear
        dx, dy = velocity.x * dt, velocity.y * dt
        dtheta = odom.twist.twist.angular.z * dt

        if not self.simulation:
            dx *= -1
            dy *= -1
            dtheta *= -1

        with self.lock:
            self.particles = self.motion_model.evaluate(self.particles, np.array([dx, dy, dtheta]))
            self.publish_transform()

    def publish_transform(self):
        """
        NOTE: This function must be called with an ownership of `self.lock` to prevent race conditions

        From the instructions:
        Anytime the particles are update (either via the motion or sensor model), determine the
        "average" (term used loosely) particle pose and publish that transform.
        """
        x = np.average(self.particles[:, 0]).item()
        y = np.average(self.particles[:, 1]).item()

        theta = self.particles[:, 2]
        theta = np.arctan2(np.average(np.sin(theta)), np.average(np.cos(theta))).item()

        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "map"
        odom.child_frame_id = "base_link_pf" if self.simulation else "base_link"
        
        odom.pose.pose = ParticleFilter.pose_to_msg(x, y, theta)

        self.odom_pub.publish(odom)

        if not self.simulation:
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = "map"
            t.child_frame_id = "base_link"

            t.transform.translation.x = odom.pose.pose.position.x
            t.transform.translation.y = odom.pose.pose.position.y
            t.transform.translation.z = odom.pose.pose.position.z
            t.transform.rotation.x = odom.pose.pose.orientation.x
            t.transform.rotation.y = odom.pose.pose.orientation.y
            t.transform.rotation.z = odom.pose.pose.orientation.z
            t.transform.rotation.w = odom.pose.pose.orientation.w
            
            self.tf_pub.sendTransform(t)

    def pose_callback(self, msg: PoseWithCovarianceStamped):
        """
        From the instruction:
        You will also consider how you want to initialize your particles. We recommend that you
        should be able to use some of the interactive topics in rviz to set an initialize "guess"
        of the robot's location with a random spread of particles around a clicked point or pose.
        Localization without this type of initialization (aka the global localization or the
        kidnapped robot problem) is very hard.
        """
        x, y, theta = ParticleFilter.msg_to_pose(msg.pose.pose)

        with self.lock:
            self.particles = (np.random.random(self.particles.shape) - 0.5) * self.particle_spread
            self.particles += np.array([x, y, 0])

            self.particles[:, 2] = theta + (np.random.random(self.num_particles) - 0.5) * 0.1

        # for experimental evaluation
        with open('particle_std_dev.txt', 'w') as f:
            f.truncate(0)
        

    def visualize_particles(self):
        """
        Display the current state of the particles in RViz
        """
        with self.lock:
            msg = PoseArray()
            msg.header.frame_id = "map"
            msg.header.stamp = self.get_clock().now().to_msg()

            msg.poses.extend(ParticleFilter.pose_to_msg(x, y, t) for [x, y, t] in self.particles)
        
        self.viz_pub.publish(msg)

        # things for experimental evaluation
        std_dev = np.std(self.particles)

        if std_dev > self.std_dev and self.exp_eval == True:
            with open('particle_std_dev.txt', 'a') as f:
                f.write(f'{std_dev}\n')

        elif std_dev <= self.std_dev and std_dev != 0.0:
            self.exp_eval = False


    @staticmethod
    def pose_to_msg(x, y, theta):
        msg = Pose()

        msg.position.x = float(x)
        msg.position.y = float(y)
        msg.position.z = 0.0
        
        quaternion = quaternion_from_euler(0.0, 0.0, theta)
        msg.orientation.x = quaternion[0]
        msg.orientation.y = quaternion[1]
        msg.orientation.z = quaternion[2]
        msg.orientation.w = quaternion[3]

        return msg

    @staticmethod
    def msg_to_pose(msg: Pose):
        pos, ori = msg.position, msg.orientation

        x, y = pos.x, pos.y
        theta = euler_from_quaternion((ori.x, ori.y, ori.z, ori.w))[-1]

        return x, y, theta


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(ParticleFilter())
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()
