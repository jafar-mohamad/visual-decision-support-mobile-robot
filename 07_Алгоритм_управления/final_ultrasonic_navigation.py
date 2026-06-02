import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.getObject('sim')

left_motor = sim.getObject('/PioneerP3DX/leftMotor')
right_motor = sim.getObject('/PioneerP3DX/rightMotor')

sensors = [sim.getObject(f'/PioneerP3DX/ultrasonicSensor[{i}]') for i in range(16)]

LEFT_GROUP = [1, 2]
RIGHT_GROUP = [5, 6]

forward_speed = 3.2
turn_speed = 3.0
backup_speed = -1.4
escape_speed = 2.5

front_block_thresh = 0.5
front_warn_thresh = 0.38
front_clear_thresh = 0.48

side_close_thresh = 0.18
side_preturn_thresh = 0.4

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

last_turn = "LEFT"

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

sim.startSimulation()
time.sleep(1.0)

state = "FORWARD"
state_start = time.time()
state_end = 0.0

start_time = time.time()

try:
    while time.time() - start_time < max_runtime:
        now = time.time()

        left_d = read_group(LEFT_GROUP)
        front_d = read_front()
        right_d = read_group(RIGHT_GROUP)

        front_blocked = front_d is not None and front_d < front_block_thresh
        front_warning = front_d is not None and front_d < front_warn_thresh
        front_clear = front_d is None or front_d > front_clear_thresh

        left_too_close = left_d is not None and left_d < side_close_thresh
        right_too_close = right_d is not None and right_d < side_close_thresh

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
                    set_speed(warn_inner, warn_outer)
                else:
                    set_speed(warn_outer, warn_inner)
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