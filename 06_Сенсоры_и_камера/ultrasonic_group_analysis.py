import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.getObject('sim')

sensors = [sim.getObject(f'/PioneerP3DX/ultrasonicSensor[{i}]') for i in range(16)]

def read_sensor(sensor):
    result, distance, _, _, _ = sim.readProximitySensor(sensor)
    if result > 0:
        return distance
    return None

sim.startSimulation()
time.sleep(1.0)

try:
    for step in range(30):
        vals = [read_sensor(s) for s in sensors]

        visible = []
        for i, v in enumerate(vals):
            if v is not None:
                visible.append(f"{i}:{v:.3f}")

        print(" | ".join(visible) if visible else "No detections")
        time.sleep(0.2)

    sim.stopSimulation()

except Exception as e:
    try:
        sim.stopSimulation()
    except:
        pass
    print("Error:", e)