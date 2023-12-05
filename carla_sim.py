# Code based on Carla examples, which are authored by 
# Computer Vision Center (CVC) at the Universitat Autonoma de Barcelona (UAB).
import sys
try:
    #sys.path.append(glob.glob('/home/baiting/CARLA_0.9.11/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
    #    sys.version_info.major,
    #    sys.version_info.minor,
    #    'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    sys.path.append('/home/baiting/CARLA_0.9.11/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg')
except IndexError:
    pass
import carla
from pathlib import Path
import pygame
from util.carla_util import *
from util.geometry_util import dist_point_linestring
import argparse
import cv2
from PIL import Image
from torchvision import transforms
import torch
from torchvision.transforms import RandomApply
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(10, 50)  # Adjust the sizes
        self.fc2 = nn.Linear(50, 20)  # Adjust the sizes
        self.fc3 = nn.Linear(20, 2)  # Output size

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_test_data(img, transform):
    #img = Image.open(img_path)
    img = Image.fromarray(img).convert('RGB')
    #img = Image.fromarray(img)
    img = transform(img)
    return img

def load_model(model_path):
    model = NeuralNetwork()  # Make sure this is the same architecture as used during training
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

def get_trajectory_from_lane_detector(lane_detector, image):
    # get lane boundaries using the lane detector
    image_arr = carla_img_to_array(image)

    poly_left, poly_right, img_left, img_right, left_coeffs, right_coeffs, coff_check_left, coff_check_right = lane_detector(image_arr)
    # https://stackoverflow.com/questions/50966204/convert-images-from-1-1-to-0-255
    img = img_left + img_right
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = img.astype(np.uint8)
    img = cv2.resize(img, (600, 400))

    # trajectory to follow is the mean of left and right lane boundary
    # note that we multiply with -0.5 instead of 0.5 in the formula for y below
    # according to our lane detector x is forward and y is left, but
    # according to Carla x is forward and y is right.
    x = np.arange(-2, 60, 1.0)
    y = -0.5 * (poly_left(x) + poly_right(x))
    # x,y is now in coordinates centered at camera, but camera is 0.5 in front of vehicle center
    # hence correct x coordinates
    x += 0.5
    trajectory = np.stack((x, y)).T
    return trajectory, img, left_coeffs, right_coeffs, coff_check_left, coff_check_right


def get_trajectory_from_map(CARLA_map, vehicle):
    # get 80 waypoints each 1m apart. If multiple successors choose the one with lower waypoint.id
    waypoint = CARLA_map.get_waypoint(vehicle.get_transform().location)
    list_waypoint = [waypoint]
    for _ in range(20):
        next_wps = waypoint.next(1.0)
        if len(next_wps) > 0:
            waypoint = sorted(next_wps, key=lambda x: x.id)[0]
        list_waypoint.append(waypoint)

    # transform waypoints to vehicle ref frame
    trajectory = np.array(
        [np.array([*carla_vec_to_np_array(x.transform.location), 1.]) for x in list_waypoint]
    ).T
    trafo_matrix_world_to_vehicle = np.array(vehicle.get_transform().get_inverse_matrix())

    trajectory = trafo_matrix_world_to_vehicle @ trajectory
    trajectory = trajectory.T
    trajectory = trajectory[:,:2]
    return trajectory

def send_control(vehicle, throttle, steer, brake,
                 hand_brake=False, reverse=False):
    throttle = np.clip(throttle, 0.0, 1.0)
    steer = np.clip(steer, -1.0, 1.0)
    brake = np.clip(brake, 0.0, 1.0)
    control = carla.VehicleControl(throttle, steer, brake, hand_brake, reverse)
    vehicle.apply_control(control)

def save_data(left_coeffs, right_coeffs, move_speed, speed, throttle, steer, filename="training_data.csv"):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Combine left and right coefficients, and control actions
        data = np.concatenate((left_coeffs, right_coeffs, [move_speed, speed, throttle, steer]))
        # Write to the file
        writer.writerow(data)

def main(fps_sim, mapid, weather_idx, showmap, model_type, max_speed):
    # Imports
    from lane_detection.openvino_lane_detector import OpenVINOLaneDetector
    from lane_detection.lane_detector import LaneDetector
    from lane_detection.camera_geometry import CameraGeometry
    from control.pure_pursuit import PurePursuitPlusPID

    actor_list = []
    pygame.init()

    display, font, clock, world = create_carla_world(pygame, mapid)

    weather_presets = find_weather_presets()
    #print(weather_presets[weather_idx])
    world.set_weather(weather_presets[weather_idx][0])
    #world.set_weather(weather_idx)
    controller = PurePursuitPlusPID()
    cross_track_list = []
    fps_list = []
    max_speed = 25
    try:
        CARLA_map = world.get_map()

        # create a vehicle
        blueprint_library = world.get_blueprint_library()
        veh_bp = random.choice(blueprint_library.filter('vehicle.audi.tt'))
        veh_bp.set_attribute('color','64,81,181')
        spawn_point = random.choice(CARLA_map.get_spawn_points())
        #print(spawn_point)
        #town 4
        #spawn_point = carla.Transform(carla.Location(x=-219, y=425, z=0.6), carla.Rotation())
        #spawn_point = carla.Transform(carla.Location(x=-350, y=420, z=0.6), carla.Rotation())


        #spawn_point = carla.Transform(carla.Location(x=35, y=204, z=0.6), carla.Rotation())
        #spawn_point = carla.Transform(carla.Location(x=35, y=204, z=0.6), carla.Rotation())
        vehicle = world.spawn_actor(veh_bp, spawn_point)
        actor_list.append(vehicle)

        # Show map
        if showmap:
            plot_map(CARLA_map, mapid, vehicle)

        startPoint = spawn_point
        startPoint = carla_vec_to_np_array(startPoint.location)

        # visualization cam (no functionality)
        camera_rgb = world.spawn_actor(
            blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(camera_rgb)
        sensors = [camera_rgb]
        world.unload_map_layer(carla.MapLayer.All)

        # Lane Detector Model
        # ---------------------------------
        cg = CameraGeometry()
        
        # TODO: Change model here
        if model_type == "openvino":
            lane_detector = OpenVINOLaneDetector()
        else:
            #lane_detector = LaneDetector(model_path=Path("lane_detection/Deeplabv3+(MobilenetV2).pth").absolute())
            #lane_detector = LaneDetector(model_path=Path("/home/baiting/Desktop/self-driving-carla/best_model.pth").absolute())
            lane_detector = LaneDetector(model_path=Path("/home/baiting/Desktop/self-driving-carla/best_model.pth").absolute())

        # Windshield cam
        camera_transforms = [carla.Transform(carla.Location(x=-4.5, z=2.2), carla.Rotation(pitch=-14.5)),
                                  carla.Transform(carla.Location(x=-4.0, z=2.2), carla.Rotation(pitch=-18.0))]
        cam_windshield_transform = carla.Transform(carla.Location(x=0.5, z=cg.height), carla.Rotation(pitch=-1*cg.pitch_deg))
        bp = blueprint_library.find('sensor.camera.rgb')
        fov = cg.field_of_view_deg
        bp.set_attribute('image_size_x', str(cg.image_width))
        bp.set_attribute('image_size_y', str(cg.image_height))
        bp.set_attribute('fov', str(fov))
        camera_windshield = world.spawn_actor(
            bp,
            cam_windshield_transform,
            attach_to=vehicle)
        actor_list.append(camera_windshield)
        sensors.append(camera_windshield)
        # ---------------------------------

        flag = True
        max_error = 0
        FPS = fps_sim
        # Create dummy images (let's assume grayscale for this example)
        height, width = 512, 1024
        img_left = np.random.randint(0, 256, (height, width), dtype=np.uint8)
        img_right = np.random.randint(0, 256, (height, width), dtype=np.uint8)

        # Add the images
        img = img_left + img_right

        # Normalize the image
        img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # Convert to uint8
        img = img.astype(np.uint8)

        # Resize the image
        img = cv2.resize(img, (600, 400))
        #cv2.namedWindow('Real-time Video', cv2.WINDOW_NORMAL)
        #Load trained policy and necessary data here
        policy = load_model('best_model.pth')
        scaler_X = joblib.load('scaler_X.pkl')
        scaler_y = joblib.load('scaler_y.pkl')
        # Create a synchronous mode context.
        with CarlaSyncMode(world, *sensors, fps=FPS) as sync_mode:
            while True:
                if should_quit():
                    return
                clock.tick()          
                
                # Advance the simulation and wait for the data. 
                tick_response = sync_mode.tick(timeout=2.0)

                snapshot, image_rgb, image_windshield = tick_response
                #try:
                #    trajectory, img = get_trajectory_from_lane_detector(lane_detector, image_windshield)
                #except:
                #    trajectory = get_trajectory_from_map(CARLA_map, vehicle)
                # Choose which image to display

                # Convert the image to a format suitable for OpenCV, if necessary

                # Display the image
                #image_to_display = carla_img_to_array(image_windshield)
                #img = Image.fromarray(image_to_display).convert('RGB')
                #cv2.imshow('Real-time Video', img)
                trajectory, img, left_coeffs, right_coeffs, coff_check_left, coff_check_right = get_trajectory_from_lane_detector(lane_detector, image_windshield)


                if (not coff_check_left) or (not coff_check_right):
                    flag == False
                    break
                max_curvature = get_curvature(np.array(trajectory))
                if max_curvature > 0.005 and flag == False:
                    move_speed = np.abs(max_speed - 100*max_curvature)
                else:
                    move_speed = max_speed

                speed = np.linalg.norm( carla_vec_to_np_array(vehicle.get_velocity()))
                #print(speed)
                throttle, steer = controller.get_control(trajectory, speed, desired_speed=move_speed, dt=1./FPS)
                # Assuming you have defined left_coeffs, right_coeffs, move_speed, and speed
                input_features = np.concatenate((left_coeffs, right_coeffs, [move_speed, speed]))

                # Reshape the input to 2D for the scaler (1 sample, many features)
                input_features_2d = input_features.reshape(1, -1)

                # Apply the standard scaler transformation
                scaled_input = scaler_X.transform(input_features_2d)

                # Convert scaled input to PyTorch tensor
                input_tensor = torch.tensor(scaled_input, dtype=torch.float32)
                with torch.no_grad():
                    # Load the saved scalers
                    predictions = policy(input_tensor)
                    predictions_numpy = predictions.numpy()
                    predictions_original_scale = scaler_y.inverse_transform(predictions_numpy)
                    throttle = predictions_original_scale[0][0]
                    steer = predictions_original_scale[0][1]
                send_control(vehicle, throttle, steer, 0)

                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                dist = dist_point_linestring(np.array([0,0]), trajectory)

                cross_track_error = int(dist)
                max_error = max(max_error, cross_track_error)
                if cross_track_error > 0:
                    cross_track_list.append(cross_track_error)
                waypoint = CARLA_map.get_waypoint(vehicle.get_transform().location)
                vehicle_loc = carla_vec_to_np_array(waypoint.transform.location)

                if np.linalg.norm(vehicle_loc-startPoint) > 20:
                    flag = False

                if np.linalg.norm(vehicle_loc-startPoint) < 20 and flag == False:
                    print('done.')
                    break
                
                if speed < 1 and flag == False:
                    print("----------------------------------------\nSTOP, car accident !!!")
                    break

                fontText = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.75
                fontColor = (255,255,255)
                lineType = 2
                laneMessage = "No Lane Detected"
                steerMessage = ""
                
                if dist < 0.75:
                    laneMessage = "Lane Tracking: Good"
                    #if coff_check_left and coff_check_right:
                    #    save_data(left_coeffs, right_coeffs, move_speed, speed, throttle, steer)
                else:
                    laneMessage = "Lane Tracking: Bad"

                cv2.putText(img, laneMessage,
                        (350,50),
                        fontText,
                        fontScale,
                        fontColor,
                        lineType)             

                if steer > 0:
                    steerMessage = "Right"
                else:
                    steerMessage = "Left"

                cv2.putText(img, "Steering: {}".format(steerMessage),
                        (400,90),
                        fontText,
                        fontScale,
                        fontColor,
                        lineType)

                steerMessage = ""
                laneMessage = "No Lane Detected"

                cv2.putText(img, "X: {:.2f}, Y: {:.2f}".format((vehicle_loc[0]), vehicle_loc[1], vehicle_loc[2]),
                            (20,50),
                            fontText,
                            0.5,
                            fontColor,
                            lineType)

                cv2.imshow('Lane detect', img)
                cv2.waitKey(1)

                fps_list.append(clock.get_fps())

                # Draw the display pygame.
                draw_image(display, image_rgb)
                display.blit(
                    font.render('     FPS (real) % 5d ' % clock.get_fps(), True, (255, 255, 255)),
                    (8, 10))
                display.blit(
                    font.render('     FPS (simulated): % 5d ' % fps, True, (255, 255, 255)),
                    (8, 28))
                display.blit(
                    font.render('     speed: {:.2f} km/h'.format(speed*3.6), True, (255, 255, 255)),
                    (8, 46))
                display.blit(
                    font.render('     cross track error: {:03d} m'.format(cross_track_error*100), True, (255, 255, 255)),
                    (8, 64))
                display.blit(
                    font.render('     max cross track error: {:03d} m'.format(max_error), True, (255, 255, 255)),
                    (8, 82))

                pygame.display.flip()


    finally:
        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()
        print('mean cross track error: ',np.mean(np.array(cross_track_list)))
        print('mean fps: ',np.mean(np.array(fps_list)))
        pygame.quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs Carla simulation with your control algorithm.')
    parser.add_argument("--mapid", default = "4", help="Choose map from 1 to 5")
    parser.add_argument("--fps", type=int, default=20, help="Setting FPS")
    parser.add_argument("--weather", type=int, default=0, help="Check function find_weather in carla_util.py for mor information")
    parser.add_argument("--showmap", type=bool, default=False, help="Display Map")
    parser.add_argument("--model", default="openvino", help="Choose between OpenVINO model and PyTorch model")
    args = parser.parse_args()
    i = 0
    while i < 1:
        try:
            main(fps_sim = args.fps, mapid = args.mapid, weather_idx=args.weather, showmap=args.showmap, model_type=args.model, max_speed=i+1)

        except KeyboardInterrupt:
            print('\nCancelled by user. Bye!')
        i += 1
