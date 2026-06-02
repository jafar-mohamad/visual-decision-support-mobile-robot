import os
import time
import cv2
import numpy as np
from ultralytics import YOLO
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

MODEL_PATH = r"E:/project/robot_project/runs/detect/experiment1/weights/best.pt"
SAVE_DIR = r"E:/project/robot_project/yolo_fusion_runtime2"
os.makedirs(SAVE_DIR, exist_ok=True)

client = RemoteAPIClient()
sim = client.getObject('sim')

left_motor = sim.getObject('/PioneerP3DX/leftMotor')
right_motor = sim.getObject('/PioneerP3DX/rightMotor')
camera = sim.getObject('/PioneerP3DX/kinect/rgb')

sensors = [sim.getObject(f'/PioneerP3DX/ultrasonicSensor[{i}]') for i in range(16)]

LEFT_GROUP = [1, 2]
RIGHT_GROUP = [5, 6]

forward_speed = 3.2
turn_speed = 3.0
backup_speed = -1.4
escape_speed = 2.5

front_block_thresh = 0.4
front_warn_thresh = 0.45
front_clear_thresh = 0.48

side_close_thresh = 0.18
side_preturn_thresh = 0.40

side_preturn_outer = 3.2
side_preturn_inner = 2.4

warn_outer = 3.0
warn_inner = 2.4

backup_time = 0.28
min_turn_time = 0.65
max_turn_time = 1.40
escape_time = 0.55

loop_dt = 0.06
max_runtime = 120.0

yolo_interval = 0.35
yolo_conf = 0.55
required_consecutive = 3

block_roi_x1 = 0.40
block_roi_x2 = 0.60
block_roi_y1 = 0.18
block_roi_y2 = 0.95
min_box_area_ratio = 0.08

human_stop_time = 1.2
visual_turn_time = 0.50
visual_turn_speed = 2

last_turn = "LEFT"

model = YOLO(MODEL_PATH)

last_yolo_time = 0.0
last_seen_label = None
same_count = 0
confirmed_blocking_label = None
confirmed_side = None

visual_state = "NONE"
visual_state_end = 0.0
visual_turn_dir = None

frame_index = 0

def read_sensor(index):
    result, distance, _, _, _ = sim.readProximitySensor(sensors[index])
    if result > 0:
        return distance
    return None

def read_group(indices):
    vals = []
    for i in indices:
        v = read_sensor(i)
        if v is not None:
            vals.append(v)
    if vals:
        return min(vals)
    return None

def read_front():
    s3 = read_sensor(3)
    s4 = read_sensor(4)

    if s3 is not None and s4 is not None:
        return min(s3, s4)

    if s3 is not None and s3 < 0.30:
        return s3

    if s4 is not None and s4 < 0.30:
        return s4

    return None

def set_speed(l, r):
    sim.setJointTargetVelocity(left_motor, l)
    sim.setJointTargetVelocity(right_motor, r)

def stop_robot():
    set_speed(0.0, 0.0)

def choose_turn(left_d, right_d, last_turn):
    if left_d is None and right_d is not None:
        return "LEFT"
    if right_d is None and left_d is not None:
        return "RIGHT"
    if left_d is None and right_d is None:
        return "RIGHT" if last_turn == "LEFT" else "LEFT"
    return "RIGHT" if left_d > right_d else "LEFT"

def get_rgb_image():
    img, resolution = sim.getVisionSensorImg(camera)
    img = np.frombuffer(img, dtype=np.uint8).reshape(resolution[1], resolution[0], 3)
    img = cv2.flip(img, 0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def run_yolo(frame):
    global last_seen_label, same_count, confirmed_blocking_label, confirmed_side, frame_index

    h, w = frame.shape[:2]
    x1_roi = int(w * block_roi_x1)
    x2_roi = int(w * block_roi_x2)
    y1_roi = int(h * block_roi_y1)
    y2_roi = int(h * block_roi_y2)

    results = model.predict(source=frame, conf=yolo_conf, imgsz=640, verbose=False)
    annotated = results[0].plot()

    cv2.rectangle(annotated, (x1_roi, y1_roi), (x2_roi, y2_roi), (0, 255, 255), 2)

    frame_index += 1
    save_path = os.path.join(SAVE_DIR, f"fusion_{frame_index:03d}.png")
    cv2.imwrite(save_path, annotated)

    boxes = results[0].boxes
    best_candidate = None

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            label = model.names[cls_id]

            if label not in ["human", "car", "tree"]:
                continue

            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = xyxy
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            area_ratio = ((x2 - x1) * (y2 - y1)) / float(w * h)

            inside_roi = (x1 < x2_roi and x2 > x1_roi and y1 < y2_roi and y2 > y1_roi)
            center_in_roi = (x1_roi <= cx <= x2_roi and y1_roi <= cy <= y2_roi)

            if inside_roi and center_in_roi and area_ratio >= min_box_area_ratio:
                score = conf + area_ratio
                side = "LEFT" if cx < (w / 2) else "RIGHT"
                if best_candidate is None or score > best_candidate["score"]:
                    best_candidate = {
                        "label": label,
                        "conf": conf,
                        "side": side,
                        "box": [x1, y1, x2, y2],
                        "score": score
                    }

    if best_candidate is None:
        last_seen_label = None
        same_count = 0
        confirmed_blocking_label = None
        confirmed_side = None
        return None, None

    label = best_candidate["label"]
    side = best_candidate["side"]

    if label == last_seen_label:
        same_count += 1
    else:
        last_seen_label = label
        same_count = 1

    if same_count >= required_consecutive:
        confirmed_blocking_label = label
        confirmed_side = side
        return confirmed_blocking_label, confirmed_side

    return None, None

sim.startSimulation()
time.sleep(1.0)

state = "FORWARD"
state_start = time.time()
state_end = 0.0

start_time = time.time()

try:
    while time.time() - start_time < max_runtime:
        now = time.time()

        if now - last_yolo_time >= yolo_interval:
            frame = get_rgb_image()
            blocking_label, blocking_side = run_yolo(frame)
            last_yolo_time = now

            if blocking_label == "human":
                visual_state = "HUMAN_STOP"
                visual_state_end = now + human_stop_time

            elif blocking_label in ["car", "tree"]:
                visual_danger = (
                  (front_d is not None and front_d < front_warn_thresh) or
                  (left_d is not None and left_d < side_preturn_thresh) or
                  (right_d is not None and right_d < side_preturn_thresh)
                )
                if visual_danger:
                 visual_state = "VISUAL_AVOID"
                 visual_state_end = now + visual_turn_time   
                visual_turn_dir = "RIGHT" if blocking_side == "LEFT" else "LEFT"

        left_d = read_group(LEFT_GROUP)
        front_d = read_front()
        right_d = read_group(RIGHT_GROUP)

        front_blocked = front_d is not None and front_d < front_block_thresh
        front_warning = front_d is not None and front_d < front_warn_thresh
        front_clear = front_d is None or front_d > front_clear_thresh

        left_too_close = left_d is not None and left_d < side_close_thresh
        right_too_close = right_d is not None and right_d < side_close_thresh

        if visual_state == "HUMAN_STOP":
            stop_robot()
            print(f"HUMAN_STOP | F={front_d} L={left_d} R={right_d}")
            if now >= visual_state_end:
                visual_state = "NONE"
            time.sleep(loop_dt)
            continue

        if visual_state == "VISUAL_AVOID":
            if visual_turn_dir == "LEFT":
                set_speed(-visual_turn_speed, visual_turn_speed)
                print(f"VISUAL_TURN_LEFT | F={front_d} L={left_d} R={right_d}")
            else:
                set_speed(visual_turn_speed, -visual_turn_speed)
                print(f"VISUAL_TURN_RIGHT | F={front_d} L={left_d} R={right_d}")

            if now >= visual_state_end:
                visual_state = "NONE"
            time.sleep(loop_dt)
            continue

        if state == "BACKUP":
            set_speed(backup_speed, backup_speed)
            if now >= state_end:
                turn_dir = choose_turn(left_d, right_d, last_turn)
                if turn_dir == "LEFT":
                    state = "TURN_LEFT"
                    last_turn = "LEFT"
                else:
                    state = "TURN_RIGHT"
                    last_turn = "RIGHT"
                state_start = now
                state_end = now + max_turn_time

        elif state == "TURN_LEFT":
            set_speed(-turn_speed, turn_speed)
            if (now - state_start) >= min_turn_time and front_clear:
                state = "ESCAPE_FORWARD"
                state_start = now
                state_end = now + escape_time
            elif now >= state_end:
                state = "ESCAPE_FORWARD"
                state_start = now
                state_end = now + escape_time

        elif state == "TURN_RIGHT":
            set_speed(turn_speed, -turn_speed)
            if (now - state_start) >= min_turn_time and front_clear:
                state = "ESCAPE_FORWARD"
                state_start = now
                state_end = now + escape_time
            elif now >= state_end:
                state = "ESCAPE_FORWARD"
                state_start = now
                state_end = now + escape_time

        elif state == "ESCAPE_FORWARD":
            set_speed(escape_speed, escape_speed)
            if now >= state_end:
                state = "FORWARD"

        else:
            if front_blocked or (left_too_close and right_too_close):
                state = "BACKUP"
                state_start = now
                state_end = now + backup_time

            elif right_d is not None and right_d < side_preturn_thresh and not front_warning:
                set_speed(side_preturn_inner, side_preturn_outer)
                state = "FORWARD"

            elif left_d is not None and left_d < side_preturn_thresh and not front_warning:
                set_speed(side_preturn_outer, side_preturn_inner)
                state = "FORWARD"

            elif front_warning:
                turn_dir = choose_turn(left_d, right_d, last_turn)
                if turn_dir == "LEFT":
                    set_speed(warn_outer, warn_inner)
                else:
                    set_speed(warn_inner, warn_outer)
                state = "FORWARD"

            else:
                set_speed(forward_speed, forward_speed)
                state = "FORWARD"

        print(f"{state} | F={front_d} L={left_d} R={right_d}")
        time.sleep(loop_dt)

    stop_robot()
    sim.stopSimulation()
    print("Finished")

except KeyboardInterrupt:
    stop_robot()
    sim.stopSimulation()
    print("Stopped by user")

except Exception as e:
    stop_robot()
    try:
        sim.stopSimulation()
    except:
        pass
    print("Error:", e)