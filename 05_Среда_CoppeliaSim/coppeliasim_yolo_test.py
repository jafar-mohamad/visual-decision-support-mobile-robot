from ultralytics import YOLO
import zmqRemoteApi
import numpy as np
import cv2
import time
import os

model = YOLO("E:/project/robot_project/runs/detect/train2/weights/best.pt")

client = zmqRemoteApi.RemoteAPIClient()
sim = client.getObject('sim')

camera = sim.getObject('/kinect/rgb')
left_motor = sim.getObject('/PioneerP3DX/leftMotor')
right_motor = sim.getObject('/PioneerP3DX/rightMotor')

forward_speed = 1.8
turn_speed = 1.3
turn_duration = 0.9
run_time = 60.0

conf_threshold = 0.5
min_box_area_ratio = 0.03
decision_cooldown = 1.2
frame_interval = 0.2

output_dir = "E:/project/robot_project/live_frames"
os.makedirs(output_dir, exist_ok=True)

last_turn_direction = 1
last_turn_time = 0.0

def move_robot(linear, angular):
    sim.setJointTargetVelocity(left_motor, linear - angular)
    sim.setJointTargetVelocity(right_motor, linear + angular)

def move_forward():
    move_robot(forward_speed, 0.0)

def stop_robot():
    move_robot(0.0, 0.0)

def turn_robot(direction):
    if direction > 0:
        move_robot(0.0, turn_speed)
    else:
        move_robot(0.0, -turn_speed)

def get_camera_image():
    img, resolution = sim.getVisionSensorImg(camera)
    img = np.frombuffer(img, dtype=np.uint8).reshape(resolution[1], resolution[0], 3)
    img = cv2.flip(img, 0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def choose_main_obstacle(image):
    results = model(image, verbose=False)
    h, w = image.shape[:2]

    x_min = int(w * 0.35)
    x_max = int(w * 0.65)
    y_min = int(h * 0.45)
    y_max = h

    annotated = results[0].plot()
    cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    candidates = []

    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if conf < conf_threshold:
            continue

        name = model.names[cls]

        if name not in ["human", "wall", "tree"]:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        box_area = box_w * box_h
        box_area_ratio = box_area / (w * h)

        if box_area_ratio < min_box_area_ratio:
            continue

        box_center_x = (x1 + x2) / 2.0
        box_center_y = (y1 + y2) / 2.0

        overlap_x1 = max(x1, x_min)
        overlap_y1 = max(y1, y_min)
        overlap_x2 = min(x2, x_max)
        overlap_y2 = min(y2, y_max)

        overlap_w = max(0, overlap_x2 - overlap_x1)
        overlap_h = max(0, overlap_y2 - overlap_y1)
        overlap_area = overlap_w * overlap_h

        if overlap_area <= 0:
            continue

        overlap_ratio = overlap_area / box_area

        if overlap_ratio < 0.25:
            continue

        center_distance = abs(box_center_x - (w / 2)) / (w / 2)
        vertical_priority = box_center_y / h

        score = (
            conf * 0.35 +
            box_area_ratio * 3.0 +
            overlap_ratio * 0.9 +
            vertical_priority * 0.8 -
            center_distance * 0.6
        )

        candidates.append({
            "name": name,
            "conf": conf,
            "score": score,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "center_x": box_center_x,
            "center_y": box_center_y
        })

    if not candidates:
        return None, annotated

    best = max(candidates, key=lambda c: c["score"])
    return best, annotated

sim.startSimulation()
time.sleep(1.5)

start_time = time.time()
frame_counter = 0

try:
    print("Simulation started")
    move_forward()

    while time.time() - start_time < run_time:
        frame = get_camera_image()
        obstacle, annotated = choose_main_obstacle(frame)

        output_path = os.path.join(output_dir, f"frame_{frame_counter:04d}.png")
        cv2.imwrite(output_path, annotated)

        now = time.time()

        if obstacle is None:
            print("No important obstacle in center -> move forward")
            move_forward()
            time.sleep(frame_interval)

        else:
            name = obstacle["name"]
            conf = obstacle["conf"]
            center_x = obstacle["center_x"]
            print(f"Main obstacle: {name} ({conf:.2f})")

            if now - last_turn_time < decision_cooldown:
                print("Cooldown active -> keep moving")
                move_forward()
                time.sleep(frame_interval)
            else:
                stop_robot()
                time.sleep(0.15)

                image_center = frame.shape[1] / 2.0

                if center_x < image_center:
                    direction = 1
                    print("Obstacle slightly left -> turn right")
                elif center_x > image_center:
                    direction = -1
                    print("Obstacle slightly right -> turn left")
                else:
                    direction = -last_turn_direction
                    print("Obstacle centered -> alternate turn direction")

                if abs(center_x - image_center) < frame.shape[1] * 0.08:
                    direction = -last_turn_direction
                    print("Obstacle near exact center -> alternate direction")

                turn_robot(direction)
                time.sleep(turn_duration)

                stop_robot()
                time.sleep(0.1)

                move_forward()

                last_turn_direction = direction
                last_turn_time = time.time()

        frame_counter += 1

    stop_robot()

finally:
    stop_robot()
    time.sleep(0.2)
    sim.stopSimulation()
    print("Finished 60-second live test.")