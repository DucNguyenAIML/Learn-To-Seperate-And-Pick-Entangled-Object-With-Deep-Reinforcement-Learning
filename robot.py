import select
import struct
import time
import os
import numpy as np
import utils
import cv2
from zmqRemoteApi import RemoteAPIClient
from logger import Logger

class Robot(object):
    def __init__(self, is_sim, obj_mesh_dir, num_obj, mainbox_limits, bufferbox_limits,
                 tcp_host_ip, tcp_port, rtc_host_ip, rtc_port, is_testing,
                 test_preset_cases, test_preset_file):
        self.is_sim = is_sim
        self.mainbox_limits = mainbox_limits
        self.buffer_limits = bufferbox_limits
        #self.final_limits = finalbox_limits
        self.main_workspace = True              # main_workspace = True -> Robot working in main bin
                                                # main_workspace = False -> Robot working in buffer bin
        self.set_new_home = False

        if self.is_sim == True:
            # Define colors for object meshes (Tableau palette)
            # Quick hack for reuse Andy's code xD
            self.color_space = np.asarray([#[78.0, 121.0, 167.0], # blue
                                           #[89.0, 161.0, 79.0], # green
                                           #[156, 117, 95], # brown
                                           #[242, 142, 43], # orange
                                           #[237.0, 201.0, 72.0], # yellow
                                           #[186, 176, 172], # gray
                                           #[255.0, 87.0, 89.0], # red
                                           #[176, 122, 161], # purple
                                           #[118, 183, 178], # cyan
                                           [255, 157, 167],
                                           [255, 157, 167],
                                           [255, 157, 167],
                                           [255, 157, 167],
                                           [255, 157, 167],
                                           [255, 157, 167],
                                           [255, 157, 167],
                                           [255, 157, 167],
                                           [255, 157, 167],
                                           [255, 157, 167]])/255.0 #pink
            # Read files in object mesh directory
            self.obj_mesh_dir = obj_mesh_dir
            self.num_obj = num_obj
            self.mesh_list = os.listdir(self.obj_mesh_dir)

            # Randomly choose objects to add to scene
            self.obj_mesh_ind = np.random.randint(0, len(self.mesh_list), size=self.num_obj)
            self.obj_mesh_color = self.color_space[np.asarray(range(self.num_obj)) % 10, :]

            # Make sure to have the server side running in CoppeliaSim:
            # Start simulation, and run this program.
            #
            # Connect to simulator
            self.sim_client = RemoteAPIClient()

            if self.sim_client == -1:
                print('Failed to connect to simulation (CoppeliaSim ZMQ API server). Exiting.')
                exit()
            else:
                print('Connected to simulation.')
                self.sim_sim = self.sim_client.getObject('sim')
                self.restart_sim()

            self.is_testing = is_testing
            self.test_preset_cases = test_preset_cases
            self.test_preset_file = test_preset_file

            # Setup virtual camera in simulation
            self.setup_sim_camera_main()
            self.setup_sim_camera_buffer()

            # If testing, read object meshes and poses from test case file
            if self.is_testing and self.test_preset_cases:
                file = open(self.test_preset_file, 'r')
                file_content = file.readlines()
                self.test_obj_mesh_files = []
                self.test_obj_mesh_colors = []
                self.test_obj_positions = []
                self.test_obj_orientations = []
                for object_idx in range(self.num_obj):
                    file_content_curr_object = file_content[object_idx].split()
                    self.test_obj_mesh_files.append(os.path.join(self.obj_mesh_dir,file_content_curr_object[0]))
                    self.test_obj_mesh_colors.append([float(file_content_curr_object[1]),float(file_content_curr_object[2]),float(file_content_curr_object[3])])
                    self.test_obj_positions.append([float(file_content_curr_object[4]),float(file_content_curr_object[5]),float(file_content_curr_object[6])])
                    self.test_obj_orientations.append([float(file_content_curr_object[7]),float(file_content_curr_object[8]),float(file_content_curr_object[9])])
                file.close()
                self.obj_mesh_color = np.asarray(self.test_obj_mesh_colors)

            # Add objects to simulation environment
            self.add_objects()

        elif self.is_sim == False:
            print("Not in simulation mode, exit!")

    def setup_sim_camera_main(self):
        # Get handle to camera
        self.cam_handle_main = self.sim_sim.getObject('/Vision_main')
        # Get camera pose and intrinsics in simulation
        cam_position_main = self.sim_sim.getObjectPosition(self.cam_handle_main, -1)
        cam_orientation_main = self.sim_sim.getObjectOrientation(self.cam_handle_main, -1)
        cam_trans_main = np.eye(4, 4)
        cam_trans_main[0:3, 3] = np.asarray(cam_position_main)
        cam_orientation_main = [-cam_orientation_main[0], -cam_orientation_main[1], -cam_orientation_main[2]]
        cam_rotm_main = np.eye(4, 4)
        cam_rotm_main[0:3, 0:3] = np.linalg.inv(utils.euler2rotm(cam_orientation_main))
        self.cam_pose_main = np.dot(cam_trans_main, cam_rotm_main)  # Compute rigid transformation representing camera pose
        self.cam_intrinsics_main = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
        self.cam_depth_scale_main = 1
        # Get background image
        self.bg_color_img_main, self.bg_depth_img_main = self.get_camera_data_main()
        self.bg_depth_img_main = self.bg_depth_img_main * self.cam_depth_scale_main
        
    def setup_sim_camera_buffer(self):
        # Get handle to camera
        self.cam_handle_buffer = self.sim_sim.getObject('/Vision_buffer')
        # Get camera pose and intrinsics in simulation
        cam_position_buffer = self.sim_sim.getObjectPosition(self.cam_handle_buffer, -1)
        cam_orientation_buffer = self.sim_sim.getObjectOrientation(self.cam_handle_buffer, -1)
        cam_trans_buffer = np.eye(4, 4)
        cam_trans_buffer[0:3, 3] = np.asarray(cam_position_buffer)
        cam_orientation_buffer = [-cam_orientation_buffer[0], -cam_orientation_buffer[1], -cam_orientation_buffer[2]]
        cam_rotm_buffer = np.eye(4, 4)
        cam_rotm_buffer[0:3, 0:3] = np.linalg.inv(utils.euler2rotm(cam_orientation_buffer))
        self.cam_pose_buffer = np.dot(cam_trans_buffer, cam_rotm_buffer)  # Compute rigid transformation representing camera pose
        self.cam_intrinsics_buffer = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
        self.cam_depth_scale_buffer = 1
        # Get background image
        self.bg_color_img_buffer, self.bg_depth_img_buffer = self.get_camera_data_buffer()
        self.bg_depth_img_buffer = self.bg_depth_img_buffer * self.cam_depth_scale_buffer


    def add_objects(self):
        # Add each object to robot workspace at x,y location and orientation (random or pre-loaded)
        self.object_handles = []
        sim_obj_handles = []
        for object_idx in range(len(self.obj_mesh_ind)):
            curr_mesh_file = os.path.join(self.obj_mesh_dir, self.mesh_list[self.obj_mesh_ind[object_idx]])
            if self.is_testing and self.test_preset_cases:
                curr_mesh_file = self.test_obj_mesh_files[object_idx]
            curr_shape_name = 'shape_%02d' % object_idx
            drop_x = (self.mainbox_limits[0][1] - self.mainbox_limits[0][0] - 0.2) * np.random.random_sample() + \
                     self.mainbox_limits[0][0] + 0.1
            drop_y = (self.mainbox_limits[1][1] - self.mainbox_limits[1][0] - 0.2) * np.random.random_sample() + \
                     self.mainbox_limits[1][0] + 0.1
            object_position = [drop_x, drop_y, 0.15]
            object_orientation = [2 * np.pi * np.random.random_sample(), 2 * np.pi * np.random.random_sample(),
                                  2 * np.pi * np.random.random_sample()]
            if self.is_testing and self.test_preset_cases:
                object_position = [self.test_obj_positions[object_idx][0], self.test_obj_positions[object_idx][1],
                                   self.test_obj_positions[object_idx][2]]
                object_orientation = [self.test_obj_orientations[object_idx][0],
                                      self.test_obj_orientations[object_idx][1],
                                      self.test_obj_orientations[object_idx][2]]
            object_color = [self.obj_mesh_color[object_idx][0], self.obj_mesh_color[object_idx][1],
                            self.obj_mesh_color[object_idx][2]]
            script_object_handle = self.sim_sim.getObject('/remoteApiCommandServer')
            # print(script_object_handle)
            script_handle = self.sim_sim.getScript(1, script_object_handle, '/remoteApiCommandServer')
            # print(f"Script handle {script_handle}")
            #time.sleep(0.5)
            ret_ints, ret_floats, ret_strings, ret_buffer = self.sim_sim.callScriptFunction('importShape',
                                                                                            script_handle,
                                                                                            [0, 0, 255, 0],
                                                                                            object_position + object_orientation + object_color,
                                                                                            [curr_mesh_file,
                                                                                             curr_shape_name])
            curr_shape_handle = ret_ints[0]
            self.object_handles.append(curr_shape_handle)
            # if not (self.is_testing and self.test_preset_cases):
            #     time.sleep(0.5)
        self.prev_obj_positions = []
        self.obj_positions = []


    def restart_sim(self):
        self.UR5_target_handle = self.sim_sim.getObject('/UR5_main_target')
        self.UR5_buffer_handle = self.sim_sim.getObject('/UR5_buffer_target')
        self.UR5_tip_handle = self.sim_sim.getObject('/UR5_tip')
        self.sim_sim.stopSimulation()

        # Set Pos and Ori for main target
        self.sim_sim.setObjectPosition(self.UR5_target_handle, -1, (-0.3, +0.45, +0.3))
        self.get_heightmap_position_main = self.sim_sim.getObjectPosition(self.UR5_target_handle, -1)
        self.sim_sim.setObjectOrientation(self.UR5_target_handle, -1, (+90 * np.pi/180, -22.5 * np.pi/180, +90 * np.pi/180))
        self.get_heightmap_orientation_main = self.sim_sim.getObjectOrientation(self.UR5_target_handle, -1)
        # Set Pos and Ori for buffer target
        self.sim_sim.setObjectPosition(self.UR5_buffer_handle, -1, (+0.3, +0.45, +0.3))
        self.sim_sim.setObjectOrientation(self.UR5_buffer_handle, -1, (+90 * np.pi/180, -22.5 * np.pi/180, +90 * np.pi/180))
        
        self.home_main_pos = self.sim_sim.getObjectPosition(self.UR5_target_handle, -1)
        self.home_buffer_pos = self.sim_sim.getObjectPosition(self.UR5_buffer_handle, -1)

        self.sim_sim.startSimulation()

        time.sleep(1)
        RG2_tip_handle = self.sim_sim.getObject('/UR5_tip')
        gripper_position = self.sim_sim.getObjectPosition(RG2_tip_handle, -1)

        while gripper_position[2] > 0.6:
            self.sim_sim.stopSimulation()
            time.sleep(1)
            self.sim_sim.startSimulation()
            gripper_position = self.sim_sim.getObjectPosition(RG2_tip_handle, -1)
            time.sleep(2)

    
    def check_sim(self):
        RG2_tip_handle = self.sim_sim.getObject('/UR5_tip')
        Circle_Workspace = self.sim_sim.getObject('/workspace_recommend_UR5')
        gripper_position = self.sim_sim.getObjectPosition(RG2_tip_handle, -1)
        UR5_workspace = self.sim_sim.getObjectPosition(Circle_Workspace, -1)
        rad2 = (((gripper_position[0] - UR5_workspace[0]) * (gripper_position[0] - UR5_workspace[0])) + 
                ((gripper_position[1] - UR5_workspace[1]) * (gripper_position[1] - UR5_workspace[1])) + 
                ((gripper_position[2] - UR5_workspace[2]) * (gripper_position[2] - UR5_workspace[2])))
        print("rad2: ", rad2)
        print("z gripper pos is: ", gripper_position[2])
        # sim_ok = gripper_position[0] > mainbox_limits[0][0] - 0.1 and gripper_position[0] < \
        #              mainbox_limits[0][1] + 0.1 and gripper_position[1] > mainbox_limits[1][0] - 0.1 and \
        #              gripper_position[1] < mainbox_limits[1][1] + 0.1 and gripper_position[2] > \
        #              mainbox_limits[2][0] and gripper_position[2] < mainbox_limits[2][1]
        if rad2 <= (1.7 * 1.7) and gripper_position[2] > 0.0001 and gripper_position[2] < 0.6:
            sim_ok = True
            print("The gripper is in workspace")
        else:
            sim_ok = False
        
        if not sim_ok:
            print('Simulation unstable. Restarting environment.')
            self.restart_sim()
            self.add_objects()


    
    def get_object_position(self):
        obj_positions = []
        for object_handle in self.object_handles:
            object_position = self.sim_sim.getObjectPosition(object_handle, -1)
            obj_positions.append(object_position)

        return obj_positions
    
    
    # # Determine whether the tip is in main or buffer
    # def main_or_buffer(self):
    #     UR5_tip = self.sim_sim.self.sim_sim.getObjectPosition('/UR5_tip', -1)
    #     UR5_main_target = self.sim_sim.getObjectPosition('/UR5_main_target', -1)
    #     UR5_buffer_target = self.sim_sim.getObjectPosition('/UR5_buffer_target', -1)

    # Get camera data of main workspace
    def get_camera_data_main(self):
        
        if self.is_sim == True:

            # Get color image from simulation
            raw_image_main, resolution = self.sim_sim.getVisionSensorImg(self.cam_handle_main, 0)
            count_main = len(raw_image_main)
            raw_image_main = struct.unpack("B" * int(count_main), raw_image_main)
            color_img_main = np.asarray(raw_image_main)
            color_img_main.shape = (resolution[1], resolution[0], 3)
            color_img_main = color_img_main.astype(np.float64) / 255
            color_img_main[color_img_main < 0] += 1
            color_img_main *= 255
            color_img_main = np.fliplr(color_img_main)
            color_img_main = color_img_main.astype(np.uint8)

            # Get depth image from simulation
            depth_buffer, resolution = self.sim_sim.getVisionSensorDepth(self.cam_handle_main)
            count = len(depth_buffer) / 4
            depth_buffer = struct.unpack("f" * int(count), depth_buffer)
            depth_img_main = np.asarray(depth_buffer)
            depth_img_main.shape = (resolution[1], resolution[0])
            depth_img_main = np.fliplr(depth_img_main)
            zNear = 0.01
            zFar = 10
            depth_img_main = depth_img_main * (zFar - zNear) + zNear
        
        else:
            print("Camera/Scanner not in simulation mode, please try again")
        
        return color_img_main, depth_img_main
    
    # get camera data of buffer workspace:
    def get_camera_data_buffer(self):
        
        if self.is_sim == True:

            # Get color image from simulation
            raw_image_buffer, resolution = self.sim_sim.getVisionSensorImg(self.cam_handle_buffer, 0)
            count_buffer = len(raw_image_buffer)
            raw_image_buffer = struct.unpack("B" * int(count_buffer), raw_image_buffer)
            color_img_buffer = np.asarray(raw_image_buffer)
            color_img_buffer.shape = (resolution[1], resolution[0], 3)
            color_img_buffer = color_img_buffer.astype(np.float64) / 255
            color_img_buffer[color_img_buffer < 0] += 1
            color_img_buffer *= 255
            color_img_buffer = np.fliplr(color_img_buffer)
            color_img_buffer = color_img_buffer.astype(np.uint8)

            # Get depth image from simulation
            depth_buffer, resolution = self.sim_sim.getVisionSensorDepth(self.cam_handle_buffer)
            count = len(depth_buffer) / 4
            depth_buffer = struct.unpack("f" * int(count), depth_buffer)
            depth_img_buffer = np.asarray(depth_buffer)
            depth_img_buffer.shape = (resolution[1], resolution[0])
            depth_img_buffer = np.fliplr(depth_img_buffer)
            zNear = 0.01
            zFar = 10
            depth_img_buffer = depth_img_buffer * (zFar - zNear) + zNear
        
        else:
            print("Camera/Scanner not in simulation mode, please try again")
        
        return color_img_buffer, depth_img_buffer


    # def close_gripper(self):
        
    #     if self.is_sim == True:
    #         gripper_motor_velocity = -0.5
    #         gripper_motor_force = 100
    #         RG2_gripper_handle = self.sim_sim.getObject('/openCloseJoint')
    #         gripper_joint_position = self.sim_sim.getJointPosition(RG2_gripper_handle)
    #         self.sim_sim.setJointTargetForce(RG2_gripper_handle, gripper_motor_force)
    #         self.sim_sim.setJointTargetVelocity(RG2_gripper_handle, gripper_motor_velocity)
    #         gripper_fully_closed = False
    #         while gripper_joint_position > -0.045:  # Block until gripper is fully closed
    #             new_gripper_joint_position = self.sim_sim.getJointPosition(RG2_gripper_handle)
    #             if new_gripper_joint_position >= gripper_joint_position:
    #                 return gripper_fully_closed
    #             gripper_joint_position = new_gripper_joint_position
    #         gripper_fully_closed = True
    #         return gripper_fully_closed
        
    #     else:
    #         print("Not in simulation mode, cannot close gripper")

    def close_gripper(self):
        if self.is_sim == True:
            gripper_motor_velocity = -0.5
            gripper_motor_force = 100
            RG2_gripper_handle = self.sim_sim.getObject('/openCloseJoint')
            self.gripper_joint_position = self.sim_sim.getJointPosition(RG2_gripper_handle)
            self.sim_sim.setJointTargetForce(RG2_gripper_handle, gripper_motor_force)
            self.sim_sim.setJointTargetVelocity(RG2_gripper_handle, gripper_motor_velocity)
            gripper_fully_closed = False
            while self.gripper_joint_position > -0.045:  # Block until gripper is fully closed
                new_gripper_joint_position = self.sim_sim.getJointPosition(RG2_gripper_handle)
                if new_gripper_joint_position >= self.gripper_joint_position:
                    return gripper_fully_closed
                self.gripper_joint_position = new_gripper_joint_position
            gripper_fully_closed = True
            return gripper_fully_closed
        
        else:
            print("Not in simulation mode, cannot close gripper")


    def check_closed_gripper(self):
        ############# Old Method ##########################
        # RG2_gripper_handle = self.sim_sim.getObject('/openCloseJoint')
        # new_gripper_joint_position = self.sim_sim.getJointPosition(RG2_gripper_handle)
        # if new_gripper_joint_position >= self.gripper_joint_position:
        #     return False
        # else:
        #     return True

        ############################# New method: Use Force Sensor ##############################
        # Read the force sensor and take the Z dimension value
        # If grasp successfully, the Z dimension force of gravity will increase
        force_sensor = self.sim_sim.getObject('/leftForceSensor')
        result, forceVector, torqueVector = self.sim_sim.readForceSensor(force_sensor)
        print("Force sensor value z-dimension: ", forceVector[2])
        if forceVector[2] <= -50:
            return False
        else:
            return True
        


    def open_gripper(self):

        if self.is_sim == True:
            gripper_motor_velocity = 0.5
            gripper_motor_force = 20
            RG2_gripper_handle = self.sim_sim.getObject('/openCloseJoint')
            gripper_joint_position = self.sim_sim.getJointPosition(RG2_gripper_handle)
            self.sim_sim.setJointTargetForce(RG2_gripper_handle, gripper_motor_force)
            self.sim_sim.setJointTargetVelocity(RG2_gripper_handle, gripper_motor_velocity)
            while gripper_joint_position < 0.03:  # Block until gripper is fully open
                gripper_joint_position = self.sim_sim.getJointPosition(RG2_gripper_handle)

        else:
            print("Not in simulation mode, cannot open gripper")
    

    def move_to(self, tool_position, tool_orientation):
        
        if self.is_sim == True:
            UR5_target_position = self.sim_sim.getObjectPosition(self.UR5_target_handle, -1)

            move_direction = np.asarray(
                [tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1],
                tool_position[2] - UR5_target_position[2]])
            move_magnitude = np.linalg.norm(move_direction)
            move_step = 0.02 * move_direction / move_magnitude      # Edit: 0.02 -> 0.005
            num_move_steps = int(np.floor(move_magnitude / 0.02))   # Edit: 0.02 -> 0.005

            for step_iter in range(num_move_steps):
                self.sim_sim.setObjectPosition(self.UR5_target_handle, -1, (
                UR5_target_position[0] + move_step[0], UR5_target_position[1] + move_step[1],
                UR5_target_position[2] + move_step[2]))
                UR5_target_position = self.sim_sim.getObjectPosition(self.UR5_target_handle, -1)
            self.sim_sim.setObjectPosition(self.UR5_target_handle, -1,
                                        (tool_position[0], tool_position[1], tool_position[2]))
            
            print("New position of Target is: ", [tool_position[0], tool_position[1], tool_position[2]])
        else:
            print("Not in simulation mode at function move_to")


    def grasp(self, position, heightmap_rotation_angle, mainbox_limits):
        print('Executing: grasp at (%f, %f, %f)' % (position[0], position[1], position[2]))

        if self.is_sim == True:
            # Compute tool orientation from heightmap rotation angle
            tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi / 2

            # Avoid collision with floor
            position = np.asarray(position).copy()
            position[2] = max(position[2] - 0.04, mainbox_limits[2][0] + 0.02)

            # Move gripper to location above grasp target
            grasp_location_margin = 0.15
            location_above_grasp_target = (position[0], position[1], position[2] + grasp_location_margin)

            # Compute gripper position and linear movement increments
            tool_position = location_above_grasp_target
            UR5_target_position = self.sim_sim.getObjectPosition(self.UR5_target_handle, -1)
            move_direction = np.asarray(
                [tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1],
                 tool_position[2] - UR5_target_position[2]])
            #print("move_direction: ", move_direction)
            move_magnitude = np.linalg.norm(move_direction)
            #print("move_magnitude: ", move_magnitude)
            move_step = 0.05 * move_direction / move_magnitude
            #print("move_step: ", move_step[0])
            num_move_steps = int(np.floor(move_direction[0] / move_step[0]))
            #print("num_move_step: ", num_move_steps)

            # Compute gripper orientation and rotation increments
            gripper_orientation = self.sim_sim.getObjectOrientation(self.UR5_target_handle, -1)
            print(f"gripper Orientation:{gripper_orientation}")
            rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
            num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1]) / rotation_step))

            # Simultaneously move and rotate gripper
            for step_iter in range(max(num_move_steps, num_rotation_steps)):
                self.sim_sim.setObjectPosition(self.UR5_target_handle, -1, (
                UR5_target_position[0] + move_step[0] * min(step_iter, num_move_steps),
                UR5_target_position[1] + move_step[1] * min(step_iter, num_move_steps),
                UR5_target_position[2] + move_step[2] * min(step_iter, num_move_steps)))
                self.sim_sim.setObjectOrientation(self.UR5_target_handle, -1, (
                np.pi / 2, gripper_orientation[1] + rotation_step * min(step_iter, num_rotation_steps), np.pi / 2))
            self.sim_sim.setObjectPosition(self.UR5_target_handle, -1,
                                           (tool_position[0], tool_position[1], tool_position[2]))
            self.sim_sim.setObjectOrientation(self.UR5_target_handle, -1, (np.pi / 2, tool_rotation_angle, np.pi / 2))

            # Ensure gripper is open
            print("Prepare open gripper.")
            self.open_gripper()

            # Approach grasp target
            print("Approach grasp target")
            self.move_to(position, None)

            # Close gripper to grasp target
            print("Close gripper to grasp target")
            gripper_full_closed = self.close_gripper()
            time.sleep(2)
            # Move gripper to location above grasp target
            # self.move_to(location_above_grasp_target, None)
            if self.main_workspace == True:
                #self.move_to([-0.3, +0.45, +0.3], None)
                self.move_to([-0.3, +0.25, +0.4], None)
            elif self.main_workspace == False:
                self.move_to(self.home_buffer_pos, None)
            time.sleep(2)

            # Check if grasp is successful
            gripper_full_closed = self.check_closed_gripper()
            self.grasp_success = not gripper_full_closed     # if gripper is fully close, then it's not a successful grasp.

            grasp_success = self.grasp_success
            # Move the grasped object elsewhere:
                # object_positions = np.asarray(self.get_obj_positions())
                # object_positions = object_positions[:, 2]
                # grasped_object_ind = np.argmax(object_positions)
                # grasped_object_handle = self.object_handles[grasped_object_ind]
                # self.sim_sim.setObjectPosition(grasped_object_handle, -1,
                #                                (-0.5, 0.5 + 0.05 * float(grasped_object_ind), 0.1))
            print("grasp_success = ", grasp_success)
            print("gripper fully close = ", gripper_full_closed)
            return grasp_success
        else:
            print("In real setting mode, please try again")


    def process_after_grasp(self):
         if (self.grasp_success == True) and (self.main_workspace == True):
                
                ################################# Threshold way - Bad behavior !!!! ######################
                # Get path of before heightmap_diff folder and name of heightmap inside
                heightmap_diff_folder = os.path.abspath('heightmap_diff')
                name_heightmap_before_grasp, name_heightmap_after_grasp = os.listdir(heightmap_diff_folder)
                
                # Load heightmap with cv2
                heightmap_before_grasp = cv2.imread(os.path.join(heightmap_diff_folder, name_heightmap_before_grasp), 0)
                heightmap_after_grasp = cv2.imread(os.path.join(heightmap_diff_folder, name_heightmap_after_grasp), 0)
                
                # Change to matrix
                heightmap_before_grasp = np.asmatrix(heightmap_before_grasp)
                heightmap_after_grasp = np.asmatrix(heightmap_after_grasp)

                # Calculate the Threshold
                self.threshold_grasp = np.matrix.sum(heightmap_before_grasp) - np.matrix.sum(heightmap_after_grasp)
                print("Diff value: ", self.threshold_grasp)

                # Delete the old heightmaps:
                # Turn on after check folder
                os.remove(os.path.join(heightmap_diff_folder, name_heightmap_before_grasp))
                os.remove(os.path.join(heightmap_diff_folder, name_heightmap_after_grasp))
                #################################### Force Torque Sensor Way ####################################
                # This method uses force sensor as a "weight sensor", measure the diff of mass between grasp 1 and 2 objects
                # (mass of grasp 2 is > 1)
                self.weightSensor = self.sim_sim.getObject('/connection')
                self.bit, self.weightVector, _ = self.sim_sim.readForceSensor(self.weightSensor)

    def push(self, position, heightmap_rotation_angle, mainbox_limits):
        print('Executing: push at (%f, %f, %f)' % (position[0], position[1], position[2]))

        if self.is_sim == True:

            # Compute tool orientation from heightmap rotation angle
            tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi / 2

            # Adjust pushing point to be on tip of finger
            position[2] = position[2] + 0.026

            # Compute pushing direction
            push_orientation = [1.0, 0.0]
            push_direction = np.asarray([push_orientation[0] * np.cos(heightmap_rotation_angle) - push_orientation[
                1] * np.sin(heightmap_rotation_angle),
                                         push_orientation[0] * np.sin(heightmap_rotation_angle) + push_orientation[
                                             1] * np.cos(heightmap_rotation_angle)])

            # Move gripper to location above pushing point
            pushing_point_margin = 0.1
            location_above_pushing_point = (position[0], position[1], position[2] + pushing_point_margin)

            # Compute gripper position and linear movement increments
            tool_position = location_above_pushing_point
            UR5_target_position = self.sim_sim.getObjectPosition(self.UR5_target_handle, -1)
            move_direction = np.asarray(
                [tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1],
                 tool_position[2] - UR5_target_position[2]])
            move_magnitude = np.linalg.norm(move_direction)
            move_step = 0.05 * move_direction / move_magnitude
            num_move_steps = int(np.floor(move_direction[0] / move_step[0]))

            # Compute gripper orientation and rotation increments
            gripper_orientation = self.sim_sim.getObjectOrientation(self.UR5_target_handle, -1)
            rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
            num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1]) / rotation_step))

            # Simultaneously move and rotate gripper
            for step_iter in range(max(num_move_steps, num_rotation_steps)):
                self.sim_sim.setObjectPosition(self.UR5_target_handle, -1, (
                UR5_target_position[0] + move_step[0] * min(step_iter, num_move_steps),
                UR5_target_position[1] + move_step[1] * min(step_iter, num_move_steps),
                UR5_target_position[2] + move_step[2] * min(step_iter, num_move_steps)))
                self.sim_sim.setObjectOrientation(self.UR5_target_handle, -1, (
                np.pi / 2, gripper_orientation[1] + rotation_step * min(step_iter, num_rotation_steps), np.pi / 2))
            self.sim_sim.setObjectPosition(self.UR5_target_handle, -1,
                                           (tool_position[0], tool_position[1], tool_position[2]))
            self.sim_sim.setObjectOrientation(self.UR5_target_handle, -1, (np.pi / 2, tool_rotation_angle, np.pi / 2))

            # Ensure gripper is closed
            self.close_gripper()

            # Approach pushing point
            self.move_to(position, None)

            # Compute target location (push to the right)
            push_length = 0.1
            target_x = min(max(position[0] + push_direction[0] * push_length, mainbox_limits[0][0]),
                           mainbox_limits[0][1])
            target_y = min(max(position[1] + push_direction[1] * push_length, mainbox_limits[1][0]),
                           mainbox_limits[1][1])
            push_length = np.sqrt(np.power(target_x - position[0], 2) + np.power(target_y - position[1], 2))

            # Move in pushing direction towards target location
            self.move_to([target_x, target_y, position[2]], None)

            # Move gripper to location above grasp target
            self.move_to([target_x, target_y, location_above_pushing_point[2]], None)

            push_success = True
        else:
            print("In real setting mode, please try again")

    def move_to_main(self):
        self.move_to(self.home_main_pos, None)

    def move_to_buffer(self):
        self.move_to(self.home_buffer_pos, None)

    def move_to_final(self):
        final_box_center = np.array([+0.45, -0.25, 0.3])
        self.move_to(final_box_center, None)