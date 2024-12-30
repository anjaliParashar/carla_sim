import sys
import os.path as path
sys.path.append(path.abspath(path.join(__file__ ,"../../")))
import carla
import math
import random
import time
import queue
import numpy as np
import cv2
from utils.carla_utils import build_projection_matrix,get_ground_truth
from utils.yolo_utils import yolo_detect

def run():
    client = carla.Client('localhost', 2000)
    world  = client.get_world()
    bp_lib = world.get_blueprint_library()

    # Get the map spawn points
    spawn_points = world.get_map().get_spawn_points()

    # spawn vehicle
    vehicle_bp =bp_lib.find('vehicle.lincoln.mkz_2020')
    vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

    # spawn camera
    camera_bp = bp_lib.find('sensor.camera.rgb')
    camera_init_trans = carla.Transform(carla.Location(z=2))
    cam_location = carla.Location(x=1.5, z=2.4)
    cam_rotation = carla.Rotation(pitch=-15)
    cam_transform = carla.Transform(cam_location, cam_rotation)
    camera = world.spawn_actor(camera_bp, cam_transform, attach_to=vehicle)

    # Get the attributes from the camera: Needed for getting ground truth bounding box to compare
    image_w = camera_bp.get_attribute("image_size_x").as_int()
    image_h = camera_bp.get_attribute("image_size_y").as_int()
    fov = camera_bp.get_attribute("fov").as_float()

    # Calculate the camera projection matrix to project from 3D -> 2D
    K = build_projection_matrix(image_w, image_h, fov)
    K_b = build_projection_matrix(image_w, image_h, fov, is_behind_camera=True)
    vehicle.set_autopilot(True)

    # Set up the simulator in synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True # Enables synchronous mode
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # Create a queue to store and retrieve the sensor data
    image_queue = queue.Queue()
    camera.listen(image_queue.put)

    idx = 0

    for i in range(50):
        vehicle_bp = random.choice(bp_lib.filter('vehicle'))
        npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
        if npc:
            npc.set_autopilot(True)

    while True:
        # Retrieve and reshape the image
        world.tick()
        image = image_queue.get()

        img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
        # Remove the alpha channel to get an RGB image
        rgb_image = img[:, :, :3]  # Extract only the first three channels (BGR)
        # Convert from BGRA to BGR (if needed)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        # Get the camera matrix 
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
        
        if idx%500==0:
            bb_boxes,actor_id = get_ground_truth(world,vehicle,img,world_2_camera,K)
            print("Ground Truth",bb_boxes,actor_id)
            classIDs,boxes,confidences = yolo_detect(bgr_image,img,idx,verbose=False,save=True)
            print("Predicted",classIDs,boxes)
        # Now draw the image into the OpenCV display window
        cv2.imshow('ImageWindowName',img)
        # Break the loop if the user presses the Q key
        if cv2.waitKey(1) == ord('q'):
            break
        idx+=1

    # Close the OpenCV display window when the game loop stops
    cv2.destroyAllWindows()
    return 

run()