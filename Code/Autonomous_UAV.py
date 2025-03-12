#!/usr/bin/env python

# Importing all necessary libraries
import numpy as np
import rospy
import sys
import tf.transformations  # For converting Euler angles to quaternions
from mavros_msgs.srv import CommandBool, SetMode
from mavros_msgs.msg import State
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
from scipy.interpolate import BSpline
import time
import math
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from darknet_ros_msgs.msg import BoundingBoxes
import threading 

# ==========================
# Object Detection Class
# ==========================

class ObjectDetector:
    """
    This class handles real-time object detection using YOLOv3 (via darknet_ros).
    It also processes depth data (from realsense stereocamera) to estimate obstacle positions in 3D space.
    """
    def __init__(self, update_callback):
        """
        Initializes object detection and subscribes to necessary ROS topics.
        """
        # update_callback: Callback function to send detected object data
        self.update_callback = update_callback

        # Convert ROS images to OpenCV format
        self.bridge = CvBridge()

        # Subscribe to RGB camera image
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)

         # Subscribe to bounding boxes from YOLO object detection
        self.bbox_sub = rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, self.bbox_callback)

        # Subscribe to depth camera image for distance estimation
        self.depth_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)
        self.bbox_data = None
        self.depth_data = None

    def image_callback(self, data):

        """ Receives and processes RGB camera images  """

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        #cv2.imshow("Image window", cv_image)
        #cv2.waitKey(3)

    def bbox_callback(self, data):
        """ Receives bounding boxes from YOLO object detection. """
        #rospy.loginfo("bounding box data received")    
        #rospy.loginfo("Bounding box data received with {} objects".format(len(data.bounding_boxes)))
        self.bbox_data = data
        self.process_data()

    def depth_callback(self, data):
        """ Receives depth camera data for distance estimation. """
        #rospy.loginfo("depth data received")    
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
            depth = np.array(cv_image, dtype=np.float32)
        except CvBridgeError as e:
            print(e)

        self.depth_data = depth

    def process_data(self):
        """ Process bounding box and depth data to determine the object's 3D position."""
        if self.bbox_data is not None and self.depth_data is not None and len(self.depth_data) > 0:
            #rospy.loginfo("Processing data 2")                   
            detection_results = []
            for box in self.bbox_data.bounding_boxes:
                # Only consider detections above 75% confidence
                if box.probability > 0.75:
                    # Camera focal length
                    fx, fy = 621.6918, 619.9693

                    # Camera principal point
                    cx, cy = 315.1213, 237.9996
                    center_x = (box.xmin + box.xmax) * 0.5
                    center_y = (box.ymin + box.ymax) * 0.5

                    corner_1_x = int(box.xmin)-1
                    corner_1_y = int(box.ymin)-1

                    corner_2_x = int(box.xmax)-1
                    corner_2_y = int(box.ymin)-1

                    corner_3_x = int(box.xmin)-1
                    corner_3_y = int(box.ymax)-1

                    corner_4_x = int(box.xmax)-1
                    corner_4_y = int(box.ymax)-1
                    
                    # Get depth value at the object's center
                    x, y = int(center_x), int(center_y)
                    depth_value = self.depth_data[y][x]
 
                    depth_value_c1 = self.depth_data[corner_1_y][corner_1_x]
                    depth_value_c2 = self.depth_data[corner_2_y][corner_2_x]
                    depth_value_c3 = self.depth_data[corner_3_y][corner_3_x]
                    depth_value_c4 = self.depth_data[corner_4_y][corner_4_x]
                    i=0
                    j=0

                    # If depth not found in the center pixels, search the nearby pixels for depth value
                    if depth_value == 0:
                        while i < 20:
                            x = x+i
                            depth_value = self.depth_data[y][x]
                            if depth_value != 0:
                                break
                            i=i+1
                        center_x = (box.xmin + box.xmax) * 0.5
                        while j < 20:
                            y=y+j
                            depth_value = self.depth_data[y][x]
                            if depth_value != 0:
                                break
                            j=j+1            
                        print "Depth not available at the center pixel"
                    
                    # Depth value at the corner of the bounding box
                    # Used to obtain the width and height of the bounding box
                    i=0
                    j=0
                    if depth_value_c1==0:
                        while i < 10:
                            corner_1_x = corner_1_x+i
                            depth_value_c1 = self.depth_data[corner_1_y][corner_1_x]
                            if depth_value_c1 != 0:
                                break
                            i=i+1
                        
                        while j < 10:
                            corner_1_y=corner_1_y+j
                            depth_value_c1 = self.depth_data[corner_1_y][corner_1_x]
                            if depth_value_c1 != 0:
                                break
                            j=j+1            
                        print "Depth for C1 not found"    

                    i=0
                    j=0
                    if depth_value_c2==0:
                        while i < 10:
                            corner_2_x = corner_2_x+i
                            depth_value_c2 = self.depth_data[corner_2_y][corner_2_x]
                            if depth_value_c2 != 0:
                                break
                            i=i+1
                        
                        while j < 10:
                            corner_2_y=corner_2_y+j
                            depth_value_c2 = self.depth_data[corner_2_y][corner_2_x]
                            if depth_value_c2 != 0:
                                break
                            j=j+1            
                        print "Depth for C2 not found"        

                    i=0
                    j=0
                    if depth_value_c3==0:
                        while i < 10:
                            corner_3_x = corner_3_x+i
                            depth_value_c3 = self.depth_data[corner_3_y][corner_3_x]
                            if depth_value_c3 != 0:
                                break
                            i=i+1
                        
                        while j < 10:
                            corner_3_y=corner_3_y+j
                            depth_value_c3 = self.depth_data[corner_3_y][corner_3_x]
                            if depth_value_c3 != 0:
                                break
                            j=j+1            
                        print "Depth for C3 not found"            

                    i=0
                    j=0
                    if depth_value_c4==0:
                        while i < 10:
                            corner_4_x = corner_4_x+i
                            depth_value_c4 = self.depth_data[corner_4_y][corner_4_x]
                            if depth_value_c4 != 0:
                                break
                            i=i+1
                        
                        while j < 10:
                            corner_4_y=corner_4_y+j
                            depth_value_c4 = self.depth_data[corner_4_y][corner_4_x]
                            if depth_value_c4 != 0:
                                break
                            j=j+1            
                        print "Depth for C4 not found"        
                        
                    # Convert depth to real-world coordinates
                    pos_x = (center_x-cx)*depth_value*0.1/fx
                    pos_y = (center_y-cy)*depth_value*0.1/fy

                    pos_c1x = (corner_1_x-cx)*depth_value_c1*0.1/fx
                    pos_c1y = (corner_1_y-cy)*depth_value_c1*0.1/fy

                    pos_c2x = (corner_2_x-cx)*depth_value_c2*0.1/fx
                    pos_c2y = (corner_2_y-cy)*depth_value_c2*0.1/fy

                    pos_c3x = (corner_3_x-cx)*depth_value_c3*0.1/fx
                    pos_c3y = (corner_3_y-cy)*depth_value_c3*0.1/fy

                    pos_c4x = (corner_4_x-cx)*depth_value_c4*0.1/fx
                    pos_c4y = (corner_4_y-cy)*depth_value_c4*0.1/fy

                    width = (pos_c4x)-(pos_c3x)
                    height = (pos_c4y)-(pos_c2y)
            
                    
                    detection_result = {
                    "class": box.Class,
                    "probability": box.probability,
                    "pos_x": pos_x,  
                    "pos_y": pos_y,  
                    "depth": depth_value,  
                    "width": width,
                    "height": height,
                     }
            
                    detection_results.append(detection_result)

            # If no objects detected, send empty detection        
            if len(detection_results) == 0:  
                detection_result = {
                "class": 0,
                "probability": 0,
                "pos_x": 0,  
                "pos_y": 0,  
                "depth": 0,  
                "width": 0,
                "height": 0,
                }
                detection_results.append(detection_result)
            self.update_callback(detection_results)

# ========================================
# UAV Offboard Control Class using MAVROS
# ========================================
# 		
class OffboardControl:
    """
    This class handles the UAV's flight control, waypoint navigation, and obstacle avoidance.
    It integrates with MAVROS for flight control and receives object detection data for obstacle avoidance.
    """
    def __init__(self, mocap_topic, object1_topic):
        rospy.init_node('offboard_control', anonymous=True)

        # Object detection system
        self.detected_objects_info = []  # This will store the detection info
        self.object_detector = ObjectDetector(self.detected_objects_info_update)

        # MAVROS publishers for UAV control
        self.vision_pose_pub = rospy.Publisher('/mavros/vision_pose/pose', PoseStamped, queue_size=10)
        self.local_position_pub = rospy.Publisher('/mavros/setpoint_position/local', PoseStamped, queue_size=10)

        # MAVROS service clients for arming and mode switching
        self.arming_client = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        self.set_mode_client = rospy.ServiceProxy('/mavros/set_mode', SetMode)

        # Subscribers for MOCAP (Motion Capture) and object detection topics
        self.mocap_subscriber = rospy.Subscriber(mocap_topic, PoseStamped, self.mocap_callback)
        self.object1_subscriber = rospy.Subscriber(object1_topic, PoseStamped, self.object1_callback)

        self.local_position_subscriber = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.local_position_callback)

        # # UAV position and orientation
        self.mocap_position = [0.0, 0.0, 0.0]
        self.mocap_orientation = [0.0, 0.0, 0.0, 1.0]

        # State information
        self.current_state = State()
        
        # File for storing MOCAP (Motion Capture) data
        self.mocap_data_file = open("mocap_data.txt", "w")

        # File for storing local position data
        self.local_position_data_file = open("local_position_data.txt", "w")

        # File for storing trajectory data
        self.traj_position_data_file = open("traj_position_data.txt", "w")

        self.nn_input_pub = rospy.Publisher('/nn_input', Float32MultiArray, queue_size=10)
        self.nn_output_sub = rospy.Subscriber('/nn_output', Float32MultiArray, self.nn_callback)
        self.nn_output = None  

    def detected_objects_info_update(self, data):
        """ Updates detected objects' information from the ObjectDetector class. """
        self.detected_objects_info = data
        
    def start_object_detection(self):
        # Use a separate thread for the callback system for object detection
        self.object_detection_thread = threading.Thread(target=self.run_object_detection)
        self.object_detection_thread.start()

    def run_object_detection(self):
        rospy.spin()  # Keeps Python from exiting until this node is stopped

    def local_position_callback(self, msg):
        """ Updates UAV's local position data. """
        position = msg.pose.position
        orientation = msg.pose.orientation
        data_str = "{},{},{},{},{},{},{},{}\n".format(
            position.x, position.y, position.z,
            orientation.x, orientation.y, orientation.z, orientation.w,
            msg.header.stamp
        )
        self.local_position_data_file.write(data_str)

    def mocap_callback(self, msg):
        """ Updates UAV position from motion capture system. """
        self.mocap_position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        self.mocap_orientation = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]

    def object1_callback(self, msg):
        self.object1_position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        self.object1_orientation = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]

    def arm(self):
        """ Arms the UAV. """
        self.send_mocap_data()        
        rospy.wait_for_service('/mavros/cmd/arming')
        try:
            response = self.arming_client(True)
            return response.success
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)
            return False
        
    def disarm(self):
        """Disarms the UAV"""
        rospy.wait_for_service('/mavros/cmd/arming')
        try:
            response = self.arming_client(False)  # Attempt to disarm the drone
            rospy.loginfo("Disarm service response: Success: {}, Result: {}".format(response.success, response.result))   
            if response.success:
                rospy.loginfo("Drone disarmed successfully.")
            else:
                rospy.logerr("Failed to disarm the drone.")
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)

    def set_offboard_mode(self):
        """ Switches Pixhawk to offboard mode for autonomous control. """
        self.send_mocap_data()    
        rospy.wait_for_service('/mavros/set_mode')
        try:
            response = self.set_mode_client(custom_mode='OFFBOARD')
            return response.mode_sent
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)
            return False
        
    def set_stabilized_mode(self):
        """Switches Pixhawk to Stabilized mode"""
        rospy.wait_for_service('/mavros/set_mode')
        try:
            response = self.set_mode_client(custom_mode='STABILIZED')
            return response.mode_sent
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)
            return False           
            
    def set_autoland_mode(self):
        """Land mode for Pixhawk"""
        rospy.wait_for_service('/mavros/set_mode')
        try:
            response = self.set_mode_client(custom_mode='LAND')
            return response.mode_sent
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)
            return False                               

    def send_mocap_data(self):
        """Sending state estimation informtaion (position & orientation) to Pixhawk"""
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.pose.position.x = self.mocap_position[0]
        msg.pose.position.y = self.mocap_position[1]
        msg.pose.position.z = self.mocap_position[2]
        msg.pose.orientation.x = self.mocap_orientation[0]
        msg.pose.orientation.y = self.mocap_orientation[1]
        msg.pose.orientation.z = self.mocap_orientation[2]
        msg.pose.orientation.w = self.mocap_orientation[3]
        self.vision_pose_pub.publish(msg)

        # Write MOCAP data to file
        #data_str = "{},{},{},{},{},{},{},{}\n".format(
        #msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
        #msg.pose.orientation.x, msg.pose.orientation.y,
        #msg.pose.orientation.z, msg.pose.orientation.w,
        #msg.header.stamp
        #)
        #self.mocap_data_file.write(data_str)


    def send_setpoint(self, x_pos, y_pos, altitude, yaw_degrees):
        """ Sends a setpoint (position and orientation) for the UAV to follow. """
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.pose.position.x = x_pos
        msg.pose.position.y = y_pos
        msg.pose.position.z = altitude

        # Convert yaw from degrees to radians and then to quaternion
        yaw_radians = np.radians(yaw_degrees)
        quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw_radians)
        msg.pose.orientation.x = quaternion[0]
        msg.pose.orientation.y = quaternion[1]
        msg.pose.orientation.z = quaternion[2]
        msg.pose.orientation.w = quaternion[3]

        self.local_position_pub.publish(msg)

    def nn_callback(self, msg):
        """NN callback for obtaining the collision free trajectory"""
        self.nn_output = np.array(msg.data)  # Store the NN output
        #rospy.loginfo("Received NN Output: {}".format(self.nn_output))

    def reconstruct_trajectory(self, tf, rate=5, order=4):
        """Reconstructs the full trajectory using B-spline coefficients."""
        self.send_mocap_data()
        if self.nn_output is None:
            rospy.logerr("No NN output received yet!")
            return None, None, None

        # Compute total N based on rate and total trajectory duration
        N = max(int(rate * tf*3), 30)  # Ensuring a minimum of 30 points for accuracy
        rospy.loginfo("Total points (N) computed: {} for rate: {} Hz and tf: {:.2f} sec".format(N, rate, tf))
        self.send_mocap_data()

        # Divide N into three parts proportionally based on segment times
        N1 = max(int(N * (1/3)), 10)  # At least 10 points per segment
        N2 = max(int(N * (1/3)), 10)
        N3 = max(N - (N1 + N2), 10)

        # Extract coefficients for x and y
        coeffs_x_ParTraj1 = self.nn_output[0:7]
        coeffs_y_ParTraj1 = self.nn_output[7:14]
        coeffs_x_ParTraj2 = self.nn_output[14:21]
        coeffs_y_ParTraj2 = self.nn_output[21:28]
        coeffs_x_ParTraj3 = self.nn_output[28:35]
        coeffs_y_ParTraj3 = self.nn_output[35:42]

        # Define breakpoints and uniform knot vectors
        breaks1 = np.linspace(0, tf/3, 5)
        breaks2 = np.linspace(tf/3, 2*tf/3, 5)
        breaks3 = np.linspace(2*tf/3, tf, 5)

        knots1 = np.concatenate(([breaks1[0]]*(order-1), breaks1, [breaks1[-1]]*(order-1)))
        knots2 = np.concatenate(([breaks2[0]]*(order-1), breaks2, [breaks2[-1]]*(order-1)))
        knots3 = np.concatenate(([breaks3[0]]*(order-1), breaks3, [breaks3[-1]]*(order-1)))
        self.send_mocap_data()

        # Define B-spline representations
        sp_x1, sp_y1 = BSpline(knots1, coeffs_x_ParTraj1, order-1), BSpline(knots1, coeffs_y_ParTraj1, order-1)
        sp_x2, sp_y2 = BSpline(knots2, coeffs_x_ParTraj2, order-1), BSpline(knots2, coeffs_y_ParTraj2, order-1)
        sp_x3, sp_y3 = BSpline(knots3, coeffs_x_ParTraj3, order-1), BSpline(knots3, coeffs_y_ParTraj3, order-1)
        self.send_mocap_data()

        # Generate time vectors
        tVec1 = np.linspace(0, tf/3, N1, endpoint=False)
        tVec2 = np.linspace(tf/3, 2*tf/3, N2, endpoint=False)
        tVec3 = np.linspace(2*tf/3, tf, N3)
        self.send_mocap_data()

        # Evaluate splines
        x1, y1 = sp_x1(tVec1), sp_y1(tVec1)
        x2, y2 = sp_x2(tVec2), sp_y2(tVec2)
        x3, y3 = sp_x3(tVec3), sp_y3(tVec3)
        self.send_mocap_data()

        # Combine all trajectory segments
        x_traj = np.concatenate((x1[:-1], x2[:-1], x3))
        y_traj = np.concatenate((y1[:-1], y2[:-1], y3))
        t_traj = np.concatenate((tVec1[:-1], tVec2[:-1], tVec3))

        return x_traj, y_traj, t_traj
        
    def save_trajectory(self, x_traj, y_traj, filename="trajectory_output.txt"):
        """Save trajectory points to a text file."""
        np.savetxt(filename, np.column_stack((x_traj, y_traj)), fmt="%.6f", delimiter=",")
        print("Trajectory saved to:", filename)    
        self.send_mocap_data()

    def run(self):
        time.sleep(0.5) 
        self.send_mocap_data()
        start_time_1 = rospy.Time.now()
        rate = rospy.Rate(5)  # rate for ros
 
        # Send state estimation data
        self.send_mocap_data()

        # Initiate Pixhawk to stabilized mode
        self.set_stabilized_mode()       
        
        # Arm the UAV
        if self.arm():
            rospy.loginfo("Drone armed successfully.")
        else:
            rospy.logerr("Failed to arm the drone.")
            return
            
        # Save the present location for takeoff
        self.send_mocap_data()
        x = self.mocap_position[0]
        y = self.mocap_position[1] 
        z =  self.mocap_position[2]
        print(x,y)

        takeoff_altitude = 1.75  # Desired takeoff altitude in meters
        yaw_angle = 90  # Yaw angle in degrees
        
        # Takeoff sequence
        rospy.loginfo("Drone is armed, taking off to {} meters.".format(takeoff_altitude))
        i=0
        self.send_mocap_data()
        while not rospy.is_shutdown() and i < 50:
            self.send_mocap_data()
            if i >= 0 and i<10:
                self.send_setpoint(x,y,takeoff_altitude, yaw_angle)
                if self.set_offboard_mode():
                    rospy.loginfo("Offboard mode set successfully.")
                    #print(np.absolute(self.mocap_position[2] - takeoff_altitude)
                else:
                    rospy.logerr("Failed to set offboard mode.")

            if i>=10 and i<50:
                self.send_setpoint(x,y,takeoff_altitude, yaw_angle)
                self.send_mocap_data()
            
            i=i+1
            rate.sleep()
  
        self.send_mocap_data()
         
        while not self.detected_objects_info and not rospy.is_shutdown():
            rospy.logwarn("Waiting for object detection...")
            rospy.sleep(0.1)

        print("DEBUG: Object detection list is now populated:", self.detected_objects_info)
        
        obj = self.detected_objects_info[0]
        self.send_mocap_data()

        if self.detected_objects_info:
            rospy.loginfo("Detected Objects! Proceed with caution")
            object_list = []
            for obj in self.detected_objects_info:
                """Converting the obstacle positoin to the MOCAP (Inertial) frame from the body frame"""
                #rospy.loginfo("Class: {} | Probability: {:.2f}".format(obj['class'], obj['probability'])) 
                #rospy.loginfo("Position (X, Y, Depth): ({:.2f}, {:.2f}, {:.2f})".format(obj["pos_x"], obj["pos_y"], obj["depth"]*0.1))       
                collision_depth =  obj['depth']*0.001
                width = obj['width']*0.01
                height = obj['height']*0.01
                x_quad = self.mocap_position[0]
                y_quad = self.mocap_position[1] 
                z_quad = self.mocap_position[2]

                self.send_mocap_data()
                
                # Obstacle Localization
                print(1,x_quad+obj['pos_x']*0.01, y_quad+collision_depth+0.1, z_quad-obj['pos_y']*0.01, width, height, collision_depth) 
                x_obj =  x_quad+obj['pos_x']*0.01
                y_obj = y_quad+collision_depth 

                object_list.append((x_obj, y_obj))
        else:
            rospy.logwarn("No objects detected. Continuing with default trajectory.")
        
        self.send_mocap_data()
        
        nn_input_msg = Float32MultiArray()

        # Boundary conditions and obstacle location information to the NN for 
        # obtaining the collision free trajectory
        nn_input_msg.data = [1,-3.2, -1, 3.55, 0.5, -1, 0.54, -0.5, 1.65, 0.54,7.1]
        self.nn_input_pub.publish(nn_input_msg)
        rospy.loginfo("Sent input to NN: {}".format(nn_input_msg.data))
        self.send_mocap_data()
        timeout = rospy.Time.now() + rospy.Duration(0.5)  # 0.5 seconds timeout
        while rospy.Time.now() < timeout:
            self.send_mocap_data()
            self.nn_input_pub.publish(nn_input_msg)
            if self.nn_output is not None:  # Check if NN output is received
                rospy.loginfo("NN Output received! Exiting loop...")
                break  # Exit loop immediately

            rospy.loginfo("Waiting for NN output...")
            rate.sleep()
        x_obj1_true = self.object1_position[0]
        y_obj1_true = self.object1_position[1] 
        
        xNN_i, yNN_i, xNN_f, yNN_f, tNN_f = nn_input_msg.data[0], nn_input_msg.data[1], nn_input_msg.data[2], nn_input_msg.data[3], nn_input_msg.data[10]
        self.send_mocap_data()

 
        if self.nn_output is not None:
            rospy.loginfo("Reconstructing trajectory...")
            x_traj, y_traj, t_traj = self.reconstruct_trajectory(tf=tNN_f)

        if x_traj is None or len(x_traj) == 0:
            rospy.logerr("No valid trajectory computed. Skipping waypoints execution.")
            return

        if x_traj is not None:
            rospy.loginfo("Trajectory reconstruction successful.")
            self.save_trajectory(x_traj, y_traj, "nn_trajectory.txt")

        # Execute the obstacle avoidance path
        for i in range(len(x_traj)):
            if rospy.is_shutdown():
                break
            self.send_mocap_data()
            self.send_setpoint(x_traj[i], y_traj[i], takeoff_altitude, yaw_angle)
            rate.sleep()   
   
        # Landing sequence
        while not rospy.is_shutdown() and np.absolute(self.mocap_position[2]) > 0.25:
            self.send_setpoint(xNN_f,yNN_f,0, yaw_angle)  # Maintain yaw while landing
            self.send_mocap_data()
            rate.sleep()

        self.send_mocap_data()  
        self.set_autoland_mode()        
        self.disarm()
 
    def __del__(self):
        # Close the file when the object is destroyed
        if self.mocap_data_file:
            self.mocap_data_file.close()

        if self.local_position_data_file:
            self.local_position_data_file.close()
    

if __name__ == '__main__':
    print("DEBUG: Running main function...")
    mocap_topic = sys.argv[1] if len(sys.argv) > 1 else '/mocap_node/drone_blue_jetson/pose'
    object1_topic = '/mocap_node/box/pose'

    try:
        print("DEBUG: OffboardControl object created.")
        offboard_control = OffboardControl(mocap_topic, object1_topic)
        offboard_control.run()
    except rospy.ROSInterruptException:
        pass
