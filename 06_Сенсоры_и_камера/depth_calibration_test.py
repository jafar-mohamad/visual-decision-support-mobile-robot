import time
import os
import cv2
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.getObject('sim')

left_motor = sim.getObject('/PioneerP3DX/leftMotor')
right_motor = sim.getObject('/PioneerP3DX/rightMotor')
depth_sensor = sim.getObject('/PioneerP3DX/kinect/depth')
rgb_sensor = sim.getObject('/PioneerP3DX/kinect/rgb')

forward_speed = 2.0
loop_sleep = 0.08
max_frames = 120

output_base = "E:/project/robot_project/depth_live_probe"
raw_dir = os.path.join(output_base, "raw_rgb")
vis_dir = os.path.join(output_base, "visualized")
depth_dir = os.path.join(output_base, "depth")

os.makedirs(raw_dir, exist_ok=True)
os.makedirs(vis_dir, exist_ok=True)
os.makedirs(depth_dir, exist_ok=True)

def set_speed(left, right):
    sim.setJointTargetVelocity(left_motor, left)
    sim.setJointTargetVelocity(right_motor, right)

def move_forward(speed):
    set_speed(speed, speed)

def stop_robot():
    set_speed(0.0, 0.0)

def get_rgb():
    img, res = sim.getVisionSensorImg(rgb_sensor)
    img = np.frombuffer(img, dtype=np.uint8).reshape(res[1], res[0], 3)
    img = cv2.flip(img, 0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def get_depth():
    depth_buffer, res = sim.getVisionSensorDepth(depth_sensor)
    depth = np.frombuffer(depth_buffer, dtype=np.float32).reshape(res[1], res[0])
    depth = cv2.flip(depth, 0)
    return depth

def stats(region):
    return {
        "mean": float(np.mean(region)),
        "min": float(np.min(region)),
        "q10": float(np.quantile(region, 0.10)),
        "q20": float(np.quantile(region, 0.20)),
    }

def get_regions(depth_img):
    h, w = depth_img.shape

    y1 = int(h * 0.62)
    y2 = int(h * 0.92)

    regions = {
        "far_left":  (int(w * 0.02), int(w * 0.12)),
        "left":      (int(w * 0.14), int(w * 0.28)),
        "center":    (int(w * 0.42), int(w * 0.58)),
        "right":     (int(w * 0.72), int(w * 0.86)),
        "far_right": (int(w * 0.88), int(w * 0.98)),
    }

    out = {}
    for name, (x1, x2) in regions.items():
        roi = depth_img[y1:y2, x1:x2]
        out[name] = stats(roi)

    return y1, y2, regions, out

def draw(rgb_img, depth_img, y1, y2, regions, out, frame_id):
    vis = rgb_img.copy()

    colors = {
        "far_left": (255, 128, 0),
        "left": (255, 0, 0),
        "center": (0, 255, 0),
        "right": (0, 0, 255),
        "far_right": (255, 0, 255),
    }

    y_text = 30
    for name, (x1, x2) in regions.items():
        color = colors[name]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis,
            f"{name}: q10={out[name]['q10']:.2f} q20={out[name]['q20']:.2f}",
            (20, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )
        y_text += 26

    cv2.putText(vis, f"frame={frame_id}", (20, y_text + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 255, 220), 2)

    depth_norm = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_norm.astype(np.uint8)
    depth_bgr = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)

    for name, (x1, x2) in regions.items():
        cv2.rectangle(depth_bgr, (x1, y1), (x2, y2), colors[name], 2)

    return vis, depth_bgr

sim.startSimulation()
time.sleep(1.0)

frame_id = 0

try:
    while frame_id < max_frames:
        move_forward(forward_speed)

        rgb_img = get_rgb()
        depth_img = get_depth()

        y1, y2, regions, out = get_regions(depth_img)
        vis, depth_vis = draw(rgb_img, depth_img, y1, y2, regions, out, frame_id)

        print(
            f"frame={frame_id} | "
            f"FL={out['far_left']['q20']:.2f} "
            f"L={out['left']['q20']:.2f} "
            f"C={out['center']['q20']:.2f} "
            f"R={out['right']['q20']:.2f} "
            f"FR={out['far_right']['q20']:.2f}"
        )

        cv2.imwrite(os.path.join(raw_dir, f"frame_{frame_id:04d}.png"), rgb_img)
        cv2.imwrite(os.path.join(vis_dir, f"frame_{frame_id:04d}.png"), vis)
        cv2.imwrite(os.path.join(depth_dir, f"frame_{frame_id:04d}.png"), depth_vis)

        frame_id += 1
        time.sleep(loop_sleep)

    stop_robot()
    sim.stopSimulation()
    print("Depth live probe finished.")

except Exception as e:
    stop_robot()
    try:
        sim.stopSimulation()
    except:
        pass
    print("Error:", e)