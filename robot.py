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
    def __init__(self, is_sim, obj_mesh_dir, num_obj, mainbox_limits, tcp_host_ip,
                 tcp_port, rtc_host_ip, rtc_port, is_testing,
                 test_preset_cases, test_preset_file):
        self.is_sim = is_sim
        self.mainbox_limits = mainbox_limits
        #self.buffer_limits = bufferbox_limits
        #self.final_limits = finalbox_limits
        
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
            self.setup_sim_camera()

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


    def setup_sim_camera(self):
        # Get handle to camera
        self.cam_handle = self.sim_sim.getObject('/Vision_sensor_persp')
        # Get camera pose and intrinsics in simulation
        cam_position = self.sim_sim.getObjectPosition(self.cam_handle, -1)
        cam_orientation = self.sim_sim.getObjectOrientation(self.cam_handle, -1)
        cam_trans = np.eye(4, 4)
        cam_trans[0:3, 3] = np.asarray(cam_position)
        cam_orientation = [-cam_orientation[0], -cam_orientation[1], -cam_orientation[2]]
        cam_rotm = np.eye(4, 4)
        cam_rotm[0:3, 0:3] = np.linalg.inv(utils.euler2rotm(cam_orientation))
        self.cam_pose = np.dot(cam_trans, cam_rotm)  # Compute rigid transformation representing camera pose
        self.cam_intrinsics = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
        self.cam_depth_scale = 1
        # Get background image
        self.bg_color_img, self.bg_depth_img = self.get_camera_data()
        self.bg_depth_img = self.bg_depth_img * self.cam_depth_scale


    def setup_3d_scanner(self):
        # Get handle to camera
        self.scanner_handle = self.sim_sim.getObject('/3D_Scanner')
        # Get camera pose and intrinsics in simulation
        scanner_position = self.sim_sim.getObjectPosition(self.scanner_handle, -1)
        scanner_orientation = self.sim_sim.getObjectOrientation(self.scanner_handle, -1)
        scanner_trans = np.eye(4, 4)
        scanner_trans[0:3, 3] = np.asarray(scanner_position)
        scanner_orientation = [-scanner_orientation[0], -scanner_orientation[1], -scanner_orientation[2]]
        scanner_rotm = np.eye(4, 4)
        scanner_rotm[0:3, 0:3] = np.linalg.inv(utils.euler2rotm(scanner_orientation))
        self.scanner_pose = np.dot(scanner_trans, scanner_rotm)  # Compute rigid transformation representing camera pose
        self.scanner_intrinsics = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
        self.scanner_depth_scale = 1
        # Get background image
        self.scanner_bg_color_img, self.scanner_bg_depth_img = self.get_camera_data()
        self.scanner_bg_depth_img = self.scanner_bg_depth_img * self.scanner_depth_scale


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
            time.sleep(2)
            ret_ints, ret_floats, ret_strings, ret_buffer = self.sim_sim.callScriptFunction('importShape',
                                                                                            script_handle,
                                                                                            [0, 0, 255, 0],
                                                                                            object_position + object_orientation + object_color,
                                                                                            [curr_mesh_file,
                                                                                             curr_shape_name])
            print("Object position is: ", object_position)

            curr_shape_handle = ret_ints[0]
            self.object_handles.append(curr_shape_handle)
            if not (self.is_testing and self.test_preset_cases):
                time.sleep(2)
        self.prev_obj_positions = []
        self.obj_positions = []


    def restart_sim(self):
        self.UR5_target_handle = self.sim_sim.getObject('/UR5_main_target')
        self.UR5_buffer_handle = self.sim_sim.getObject('/UR5_buffer_target')
        self.UR5_tip_handle = self.sim_sim.getObject('/UR5_tip')
        self.sim_sim.stopSimulation()

        # Set Pos and Ori for main target
        self.sim_sim.setObjectPosition(self.UR5_target_handle, -1, (-0.1088, +0.57714, +0.53764))
        self.sim_sim.setObjectOrientation(self.UR5_target_handle, -1, (0 * np.pi/180, -90 * np.pi/180, -33.999 * np.pi/180))

        # Set Pos and Ori for buffer target
        #self.sim_sim.setObjectPosition(self.UR5_buffer_handle, -1, (-0.57714, -0.1088, +0.53764))
        #self.sim_sim.setObjectOrientation(self.UR5_buffer_handle, -1, (+90 * np.pi/180, 0 * np.pi/180, 56.001 * np.pi/180))
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
        # sim_ok = gripper_position[0] > mainbox_limits[0][0] - 0.1 and gripper_position[0] < \
        #              mainbox_limits[0][1] + 0.1 and gripper_position[1] > mainbox_limits[1][0] - 0.1 and \
        #              gripper_position[1] < mainbox_limits[1][1] + 0.1 and gripper_position[2] > \
        #              mainbox_limits[2][0] and gripper_position[2] < mainbox_limits[2][1]
        if rad2 <= (1.7 * 1.7) and gripper_position[2] > 0.1:
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
    
    
    def get_camera_data(self):
        
        if self.is_sim == True:

            # Get color image from simulation
            raw_image, resolution = self.sim_sim.getVisionSensorImg(self.cam_handle, 0)
            count = len(raw_image)
            raw_image = struct.unpack("B" * int(count), raw_image)
            color_img = np.asarray(raw_image)
            color_img.shape = (resolution[1], resolution[0], 3)
            color_img = color_img.astype(np.float64) / 255
            color_img[color_img < 0] += 1
            color_img *= 255
            color_img = np.fliplr(color_img)
            color_img = color_img.astype(np.uint8)

            # Get depth image from simulation
            depth_buffer, resolution = self.sim_sim.getVisionSensorDepth(self.cam_handle)
            count = len(depth_buffer) / 4
            depth_buffer = struct.unpack("f" * int(count), depth_buffer)
            depth_img = np.asarray(depth_buffer)
            depth_img.shape = (resolution[1], resolution[0])
            depth_img = np.fliplr(depth_img)
            zNear = 0.01
            zFar = 10
            depth_img = depth_img * (zFar - zNear) + zNear
        
        else:
            print("Camera/Scanner not in simulation mode, please try again")
        
        return color_img, depth_img
    

    def close_gripper(self):
        
        if self.is_sim == True:
            gripper_motor_velocity = -0.5
            gripper_motor_force = 100
            RG2_gripper_handle = self.sim_sim.getObject('/openCloseJoint')
            gripper_joint_position = self.sim_sim.getJointPosition(RG2_gripper_handle)
            self.sim_sim.setJointTargetForce(RG2_gripper_handle, gripper_motor_force)
            self.sim_sim.setJointTargetVelocity(RG2_gripper_handle, gripper_motor_velocity)
            gripper_fully_closed = False
            while gripper_joint_position > -0.045:  # Block until gripper is fully closed
                new_gripper_joint_position = self.sim_sim.getJointPosition(RG2_gripper_handle)
                # print(gripper_joint_position)
                if new_gripper_joint_position >= gripper_joint_position:
                    return gripper_fully_closed
                gripper_joint_position = new_gripper_joint_position
            gripper_fully_closed = True
        
        else:
            print("Not in simulation mode, cannot close gripper")

    
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
            # self.UR5_target_handle = self.sim_sim.getObject('/UR5_main_target')
            UR5_target_position = self.sim_sim.getObjectPosition(self.UR5_target_handle, -1)
            self.get_heightmap_position_main = UR5_target_position  # Save location for robot return to get heightmap after each action in main bin
            move_direction = np.asarray(
                [tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1],
                 tool_position[2] - UR5_target_position[2]])
                # distance = tool position - ur5 main target
            move_magnitude = np.linalg.norm(move_direction)
            move_step = 0.02 * move_direction / move_magnitude
            num_move_steps = int(np.floor(move_magnitude / 0.02))

            for step_iter in range(num_move_steps):
                self.sim_sim.setObjectPosition(self.UR5_target_handle, -1, (
                UR5_target_position[0] + move_step[0], UR5_target_position[1] + move_step[1],
                UR5_target_position[2] + move_step[2]))
                UR5_target_position = self.sim_sim.getObjectPosition(self.UR5_target_handle, -1)
            self.sim_sim.setObjectPosition(self.UR5_target_handle, -1,
                                           (tool_position[0], tool_position[1], tool_position[2]))
            
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
            move_magnitude = np.linalg.norm(move_direction)
            move_step = 0.05 * move_direction / move_magnitude
            num_move_steps = int(np.floor(move_direction[0] / move_step[0]))

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

            # Move gripper to location above grasp target
            # self.move_to(location_above_grasp_target, None)

            # Move back to location get heightmap: (Not complete - Build for Mainbox only)
            self.move_to([UR5_target_position[0], UR5_target_position[1], UR5_target_position[2]], None)

            # Check if grasp is successful
            gripper_full_closed = self.close_gripper()
            grasp_success = not gripper_full_closed     # if gripper is fully close, then it's not a successful grasp.

            # Move the grasped object elsewhere
            if grasp_success:
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
                threshold_grasp = heightmap_before_grasp - heightmap_after_grasp
                
                # if grasp 2 or more tangled objects, then move to the buffer bin and drop
                # if threshold_grasp >= x then:
                self.move_to(self.UR5_buffer_handle)
                self.open_gripper()

                # if grasp only 1 object, then move to the result bin and drop
                # (Do something here)

                # object_positions = np.asarray(self.get_obj_positions())
                # object_positions = object_positions[:, 2]
                # grasped_object_ind = np.argmax(object_positions)
                # grasped_object_handle = self.object_handles[grasped_object_ind]
                # self.sim_sim.setObjectPosition(grasped_object_handle, -1,
                #                                (-0.5, 0.5 + 0.05 * float(grasped_object_ind), 0.1))
        else:
            print("In real setting mode, please try again")

    
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
            # self.move_to([target_x, target_y, location_above_pushing_point[2]], None)

            # Move back to location get heightmap: (Not complete - Build for Mainbox only)
            self.move_to(self.get_heightmap_position_main, None)

            push_success = True
        else:
            print("In real setting mode, please try again")