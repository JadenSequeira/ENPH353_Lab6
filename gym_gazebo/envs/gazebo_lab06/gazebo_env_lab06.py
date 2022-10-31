## @package envs
#  Gazebo_Lab06_Env
#
#  The following script sets up the ROS Gazebo environment for the training of the 
#  line following robot through Q-learning. The main funcitonality in this calss is setting the control of the robots
#  movements based on actions chosen, and creating a statespace based on the robots camera vision.


import cv2
import gym
import math
import rospy
import roslaunch
import time
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from sensor_msgs.msg import Image
from time import sleep
from gym.utils import seeding

## Gazebo_Lab06_Env Class creates an ROS and gym-gazebo environments for line following Q-learning
#
#  The class initializes the environment along with bridging the camera vision to openCV images for processing and analyzing.
#  The class can be used to process the image into a state in a designed statespace. It can also be used to move the robot and determine the reward through the step function, based on the aciton
#  chosen. Finally it can reset the robot in the environment and can also seed the robot at the start (randomly).
class Gazebo_Lab06_Env(gazebo_env.GazeboEnv):

    ## Initialization function iniitializes the ROS environment and robot along with Subsciption and Publishing
    #  The functino also intializes the gazebo physics engine.
    def __init__(self):
        # Launch the simulation with the given launchfile name
        LAUNCH_FILE = '/home/fizzer/enph353_gym-gazebo-noetic/gym_gazebo/envs/ros_ws/src/enph353_lab06/launch/lab06_world.launch'
        gazebo_env.GazeboEnv.__init__(self, LAUNCH_FILE)
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world',
                                              Empty)

        self.action_space = spaces.Discrete(3)  # F,L,R
        self.reward_range = (-np.inf, np.inf)
        self.episode_history = []

        self._seed()

        self.bridge = CvBridge()
        self.timeout = 0  # Used to keep track of images with no line detected

        self.lower_blue = np.array([97,  0,   0])
        self.upper_blue = np.array([150, 255, 255])

    ## The process_image function intakes an image and provides and returns a state based on the state spaced implemented
    #  More precisely, it analyzes the cv_image and computes the state array and episode termination condition.
    #  The state array is a list of 10 elements (in this case) indicating where in the
    #  The episode termination condition is triggered and outputted when the line is not detected for more than 30 frames.
    #  @param data the image array provided from subscribing to the robots camera
    def process_image(self, data):

        # Convert data to an openCV image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # Please note that the state space can be increased by dividing the picture into smaller subdivisions (different styles of alterations are possible like 2 layers)

        state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        done = False

        # Image processing to determine the center of mass fo the line for providing the state
        gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        gblur = cv2.GaussianBlur(gray, (5,5), 0)
        ret,binary = cv2.threshold(gblur,127,255, cv2.THRESH_BINARY_INV)
    
        
        # Calculate the center of mass of the image using Open cv moments
        M = cv2.moments(binary)
        if (M["m00"] > 0):
            cX = int(M["m10"]/M["m00"])
        

            rows, cols = binary.shape

            # Break image into 10 vertical columns (10 states)
            spacing = cols/10
            lowerbound = 0
            upperbound = lowerbound + spacing
            for i in range(10):
                
                if ( cX >= lowerbound and cX <= upperbound):
                    state[i] = 1
                    break

                lowerbound += spacing
                upperbound += spacing
                
        else:
            self.timeout += 1

        if (self.timeout > 30):
            done = True

        return state, done


    ## The seed function seeds the robot at the start of the episode
    #  @param seed (default = None) seed for the robot
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    ## The step function publishes an action to the robot depending on the action chosen and then provides the rewards gained
    #  @param action the action being taken (ie. left, right, forward)
    def step(self, action):

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        self.episode_history.append(action)

        vel_cmd = Twist()

        # the following agular and linear values can be changed to tune the learning of the robot
        if action == 0:  # FORWARD
            vel_cmd.linear.x = 0.4
            vel_cmd.angular.z = 0.0
        elif action == 1:  # LEFT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.5
        elif action == 2:  # RIGHT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = -0.25

        # publish the action to the robot
        self.vel_pub.publish(vel_cmd)

        # Wait for the next image for  the camera feed of the robot
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw', Image,
                                              timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        #Process the image after the action was taken
        state, done = self.process_image(data)

        # Set the rewards for your action. These can be changed to tune the learning of the robot (sometimes track dependant)
        if not done:
            if action == 0:  # FORWARD
                reward = 4
            elif action == 1:  # LEFT
                reward = 3
            else:
                reward = 2  # RIGHT
        else:
            reward = -200

        return state, reward, done, {}

    ## The reset function resets the robot (usually when its camera is off track)
    def reset(self):

        print("Episode history: {}".format(self.episode_history))
        self.episode_history = []
        print("Resetting simulation...")
        # Resets the state of the environment and returns an initial
        # observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            # reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            # resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        # read image data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw',
                                              Image, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        self.timeout = 0
        state, done = self.process_image(data)

        return state
