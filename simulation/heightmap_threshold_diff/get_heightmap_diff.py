import sys
import os
# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(current_dir)
main_dir =  os.path.dirname(parent_dir)
# heightmap_data_path = os.path.join(current_dir, "heightmap_threshold_data")
# Add the parent directory to sys.path
# print("current_dir: ", current_dir)
# print("parent_dir: ", parent_dir)
# print("main_dir: ", main_dir)
# print("heightmap data path: ", heightmap_data_path)
sys.path.append(main_dir)
###################################################################
# Now import zmqRemoteApi
from zmqRemoteApi import RemoteAPIClient
import time
import numpy as np
import utils
import struct
import cv2

def restart_sim():
    sim_sim.stopSimulation()
    sim_sim.startSimulation()
    time.sleep(1)

def add_obj():
    obj_mesh_dir =  os.path.join(main_dir, "objects\\blocks")
    mesh_list = os.listdir(obj_mesh_dir)
    obj_mesh_ind = np.random.randint(0, len(mesh_list), size=num_obj)
    object_handles = []
    for object_idx in range(len(obj_mesh_ind)):
        curr_mesh_file = os.path.join(obj_mesh_dir, mesh_list[obj_mesh_ind[object_idx]])
        curr_shape_name = 'shape_%02d' % object_idx
        drop_x = (mainbox_limits[0][1] - mainbox_limits[0][0] - 0.2) * np.random.random_sample() + \
                     mainbox_limits[0][0] + 0.1
        drop_y = (mainbox_limits[1][1] - mainbox_limits[1][0] - 0.2) * np.random.random_sample() + \
                     mainbox_limits[1][0] + 0.1
        object_position = [drop_x, drop_y, 0.15]
        # object_orientation = [2 * np.pi * np.random.random_sample(), 2 * np.pi * np.random.random_sample(),
        #                        2 * np.pi * np.random.random_sample()]
        script_object_handle = sim_sim.getObject('/Dummy')
        script_handle = sim_sim.getScript(1, script_object_handle, '/Dummy')
        time.sleep(2)
        ret_ints, ret_floats, ret_strings, ret_buffer = sim_sim.callScriptFunction('importShape',
                                                                                    script_handle,
                                                                                    [0, 0, 255, 0],
                                                                                    object_position,
                                                                                    [curr_mesh_file, curr_shape_name])

def get_camera_data():
        cam_handle = sim_sim.getObject('/Vision_sensor_persp')
        
        # Get color image from simulation
        raw_image, resolution = sim_sim.getVisionSensorImg(cam_handle, 0)
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
        depth_buffer, resolution = sim_sim.getVisionSensorDepth(cam_handle)
        count = len(depth_buffer) / 4
        depth_buffer = struct.unpack("f" * int(count), depth_buffer)
        depth_img = np.asarray(depth_buffer)
        depth_img.shape = (resolution[1], resolution[0])
        depth_img = np.fliplr(depth_img)
        zNear = 0.01
        zFar = 10
        depth_img = depth_img * (zFar - zNear) + zNear
        return color_img, depth_img

def save_heightmaps(depth_heightmap, mode):
    name = int(input("Enter name for this picture: "))
    mode = 0
    #color_heightmap = cv2.cvtColor(color_heightmap, cv2.COLOR_RGB2BGR)
    #cv2.imwrite(os.path.join(heightmap_data_path, '%06d.%s.color.png' % (name, mode)), color_heightmap)
    depth_heightmap = np.round(depth_heightmap * 100000).astype(np.uint16) # Save depth in 1e-5 meters
    cv2.imwrite(os.path.join(heightmap_data_path, '%06d.%s.depth.png' % (name, mode)), depth_heightmap)

# def setup_sim_camera():
#     # Get handle to camera
#     cam_handle = sim_sim.getObject('/Vision_sensor_persp')
#     # Get camera pose and intrinsics in simulation
#     cam_position = sim_sim.getObjectPosition(cam_handle, -1)
#     cam_orientation = sim_sim.getObjectOrientation(cam_handle, -1)
#     cam_trans = np.eye(4, 4)
#     cam_trans[0:3, 3] = np.asarray(cam_position)
#     cam_orientation = [-cam_orientation[0], -cam_orientation[1], -cam_orientation[2]]
#     cam_rotm = np.eye(4, 4)
#     cam_rotm[0:3, 0:3] = np.linalg.inv(utils.euler2rotm(cam_orientation))
#     cam_pose = np.dot(cam_trans, cam_rotm)  # Compute rigid transformation representing camera pose
#     cam_intrinsics = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
#     # Get background image
#     bg_color_img, bg_depth_img = get_camera_data()
#     bg_depth_img = bg_depth_img * cam_depth_scale

# Declare some global things:
num_obj = int(input("Input number of objects: "))
mainbox_limits = np.asarray([[-0.324, 0.124], [0.276, 0.724], [-0.0001, 0.6]])
cam_depth_scale = 1
heightmap_data_path = os.path.join(current_dir, "heightmap_threshold_data")
sim_client = RemoteAPIClient() # Connect to CoppeliaSim


##################################################################################
if sim_client == -1:
    print('Failed to connect to simulation (CoppeliaSim ZMQ API server). Exiting.')
    exit()
else:
    print('Connected to simulation.')
    sim_sim = sim_client.getObject('sim')
    restart_sim()
    add_obj()
####################################################################################

cam_handle = sim_sim.getObject('/Vision_sensor_persp')    
cam_position = sim_sim.getObjectPosition(cam_handle, -1)
cam_orientation = sim_sim.getObjectOrientation(cam_handle, -1)
cam_trans = np.eye(4, 4)
cam_trans[0:3, 3] = np.asarray(cam_position)
cam_orientation = [-cam_orientation[0], -cam_orientation[1], -cam_orientation[2]]
cam_rotm = np.eye(4, 4)
cam_rotm[0:3, 0:3] = np.linalg.inv(utils.euler2rotm(cam_orientation))
cam_pose = np.dot(cam_trans, cam_rotm)  # Compute rigid transformation representing camera pose
cam_intrinsics = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
heightmap_resolution = 0.002

#####################################################################################################
## WARNING: U SHOULD BEGIN WITH THE 0000.0.depth.png first. Or the "counter" will make the program fail.
flag = True
counter = 0
while flag == True:
    command_string = input("Press y to save heightmap, press n to exit: ")
    if command_string == "y" or command_string == "Y":
        color_img, depth_img = get_camera_data()
        depth_img = depth_img * cam_depth_scale
        color_heightmap, depth_heightmap = utils.get_heightmap(color_img, depth_img, cam_intrinsics, cam_pose, mainbox_limits, heightmap_resolution)
        valid_depth_heightmap = depth_heightmap.copy()
        valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0
        save_heightmaps(valid_depth_heightmap, 0)
        counter += 1

        if counter == 2:
            counter = 0 # reset counter for next loop
            print("---- MODE ----")
            print("[1]: Grasp 1 object")
            print("[2]: Grasp 2 object")
            mode_save = int(input("Enter 1 or 2 to select mode: "))
            name_file = ""
            
            # If not select mode yet, then go to select mode:
            if name_file == "":
                if mode_save == 1:
                    name_file = "grasp_1_obj.txt"
                elif mode_save == 2:
                    name_file = "grasp_2_obj.txt"
                    print(name_file)
                else:
                    print("Only 1 or 2 is acceptable. Error in typing")
                    exit()

            # Prepare the path:
            path_to_text = os.path.join(current_dir, name_file)
            heightmaps_name = os.listdir(heightmap_data_path)
            path_to_heightmap_0 = os.path.join(heightmap_data_path, heightmaps_name[0])
            path_to_heightmap_1 = os.path.join(heightmap_data_path, heightmaps_name[1])

            # Load heightmap with opencv, then convert it into matrix value:
            matrix_heightmap_0 = np.asmatrix(cv2.imread(path_to_heightmap_0, 0))
            print("sum of heightmap 0: ", np.matrix.sum(matrix_heightmap_0))
            matrix_heightmap_1 = np.asmatrix(cv2.imread(path_to_heightmap_1, 0))
            print("sum of heightmap 1: ", np.matrix.sum(matrix_heightmap_1))

            # Get the threshold difference values, then write to the file:
            diff_value = np.matrix.sum(matrix_heightmap_0) - np.matrix.sum(matrix_heightmap_1)
            print("Diff_value is: ", diff_value)
            with open(path_to_text, 'a') as f:
                f.write(''.join(str(diff_value)) + '\n')
    
    
    elif command_string == "n" or command_string == "N":
        print("Exit")
        sim_sim.stopSimulation()
        flag = False
    else:
        print("Unknown command")
