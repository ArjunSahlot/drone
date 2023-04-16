import numpy as np
from mpu6050 import mpu6050
import time

# Constants
FACTOR = 0.98  # Take FACTOR% of the gyro data and (1-FACTOR)% of the accelerometer data (accounts for drift)
DAMPING = 0.9  # Retain DAMPING% of the previous rotation and factor in (1-DAMPING)% of the new rotation (accounts for noise)
CLOCK_SPEED = 250  # FPS of the gyroscope
ACCEL_OFFSET = [1.20358374, -2.68629782]  # Offset for my accelerometer

mpu = mpu6050(0x68)
mpu.set_gyro_range(mpu6050.GYRO_RANGE_500DEG)

def _calibrate(duration, offsets, counter):
    start = time.time()
    while time.time()-start < duration:
        offsets += list(mpu.get_gyro_data().values())
        counter += 1

    return offsets, counter

def calibrate(duration):
    offsets = np.zeros(3)
    counter = 0
    print("Calibrating...")
    for i in range(duration, 0, -1):
        print(f"{i} seconds left")
        offsets, counter = _calibrate(1, offsets, counter)
    
    print(f"Calibration done, {counter} samples taken")

    return offsets/counter

def rotation_by_accel():
    accel_data = np.array(list(mpu.get_accel_data().values()))
    total_accel = np.sqrt(np.sum(np.square(accel_data)))
    accel_rot = np.degrees(np.arcsin(accel_data[:2][::-1]/total_accel))
    accel_rot[1] *= -1
    accel_rot -= ACCEL_OFFSET
    return accel_rot


offsets = calibrate(3)

# First time it is all accelerometer data
final_rotation = rotation_by_accel()
pitch, roll = final_rotation
prev_gyro = [0, 0, 0]
prev_time = time.time()
start = time.time()
while True:
    gyro_data = list(mpu.get_gyro_data().values())-offsets
    accel_rot = rotation_by_accel()

    # Trapezoidal riemann sum
    pitch += 0.5*(gyro_data[0]+prev_gyro[0])*(time.time()-prev_time)
    roll += 0.5*(gyro_data[1]+prev_gyro[1])*(time.time()-prev_time)
    prev_time = time.time()
    prev_gyro = gyro_data
        
    pitch += roll * np.sin(np.radians(gyro_data[2]*(time.time()-prev_time)))
    roll -= pitch * np.sin(np.radians(gyro_data[2]*(time.time()-prev_time)))

    # Account for drift
    pitch = pitch*FACTOR + accel_rot[0]*(1-FACTOR)
    roll = roll*FACTOR + accel_rot[1]*(1-FACTOR)

    # Damping
    final_rotation = final_rotation*DAMPING + np.array([pitch, roll])*(1-DAMPING)

    print(f"Rotation: {final_rotation}")
