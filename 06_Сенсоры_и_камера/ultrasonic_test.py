import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.getObject('sim')

left_motor = sim.getObject('/PioneerP3DX/leftMotor')
right_motor = sim.getObject('/PioneerP3DX/rightMotor')

sensors = [sim.getObject(f'/PioneerP3DX/ultrasonicSensor[{i}]') for i in range(16)]

def read_sensor(sensor):
    result, distance, _, _, _ = sim.readProximitySensor(sensor)
    if result > 0:
        return distance
    return None

def set_speed(l, r):
    sim.setJointTargetVelocity(left_motor, l)
    sim.setJointTargetVelocity(right_motor, r)

sim.startSimulation()
time.sleep(1.0)

try:
    set_speed(2.0, 2.0)

    for _ in range(20):
        vals = [read_sensor(s) for s in sensors]
        print(vals)
        time.sleep(0.2)

    set_speed(0.0, 0.0)
    sim.stopSimulation()

except Exception as e:
    set_speed(0.0, 0.0)
    try:
        sim.stopSimulation()
    except:
        pass
    print("Error:", e)