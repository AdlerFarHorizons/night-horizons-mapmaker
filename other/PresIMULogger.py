import sys, getopt

sys.path.append('.')
import RTIMU
import os
import os.path
import time
import math

from datetime import datetime

from ina219 import INA219
from ina219 import DeviceRangeError

SETTINGS_FILE = "RTIMULib"
TimeFormat = "%H:%M:%S %d.%m"
debug = False

#  computeHeight() - the conversion uses the formula:
#
#  h = (T0 / L0) * ((p / P0)**(-(R* * L0) / (g0 * M)) - 1)
#
#  where:
#  h  = height above sea level
#  T0 = standard temperature at sea level = 288.15
#  L0 = standard temperatur elapse rate = -0.0065
#  p  = measured pressure
#  P0 = static pressure = 1013.25
#  g0 = gravitational acceleration = 9.80665
#  M  = mloecular mass of earth's air = 0.0289644
#  R* = universal gas constant = 8.31432
#
#  Given the constants, this works out to:
#
#  h = 44330.8 * (1 - (p / P0)**0.190263)

#Initiate INA219
SHUNT_OHMS = 0.1
ina = INA219(SHUNT_OHMS)
ina.configure()

# print(round(ina.voltage(), 2))
# print(round(((ina.current()*(-1))/1000), 3))
# print(round(ina.power()/1000, 2))

def computeHeight(pressure):
    return 44330.8 * (1 - pow(pressure / 1013.25, 0.190263));
    
print("Using settings file " + SETTINGS_FILE + ".ini")
if not os.path.exists(SETTINGS_FILE + ".ini"):
  print("Settings file does not exist, will be created")

s = RTIMU.Settings(SETTINGS_FILE)
imu = RTIMU.RTIMU(s)
pressure = RTIMU.RTPressure(s)

print("IMU Name: " + imu.IMUName())
print("Pressure Name: " + pressure.pressureName())

if (not imu.IMUInit()):
    print("IMU Init Failed")
    sys.exit(1)
else:
    print("IMU Init Succeeded");

# this is a good time to set any fusion parameters

imu.setSlerpPower(0.02)
imu.setGyroEnable(True)
imu.setAccelEnable(True)
imu.setCompassEnable(True)

if (not pressure.pressureInit()):
    print("Pressure sensor Init Failed")
else:
    print("Pressure sensor Init Succeeded")

poll_interval = imu.IMUGetPollInterval()
#print("Recommended Poll Interval: %dmS\n" % poll_interval)

# Delay start by 15 seconds
time.sleep(15)

os.system("sudo echo 1 > /sys/class/gpio/gpio11/value &")

# Write Headers to the log file
file = open('/home/pi/PresIMULog.csv', 'a')
file.write("\n\n\n" + "CurrTimestamp" + "," + "TempC" + "," + "pressure" + "," + "mAltitude" + "," + "imuAccelX" + "," + "imuAccelY" + "," + "imuAccelZ" + "," + "imuGyroX" + "," + "imuGyroY" + "," + "imuGyroZ" + "," + "imuMagX" + "," + "imuMagY" + "," + "imuMagZ" + "," + "imuRoll" + "," + "imuPitch" + "," + "imuYaw" + "," + "VoltageV" + "," + "PiCurrentA" + "," + "PiPowerW" + "," + "\n")
file.close()

while True:
# Get timestamp
    CurrTimestamp = datetime.now()
#    CurrTimestamp = CurrTimestamp.strftime(TimeFormat)

    if imu.IMURead():
    # get all fused data from IMU and pressure/altitude sensor
        data = imu.getIMUData()
        (data["pressureValid"], data["pressure"], data["temperatureValid"], data["temperature"]) = pressure.pressureRead()
        gyro = data["gyro"]
        accel = data["accel"]
        compass = data["compass"]
        fusionPose = data["fusionPose"]

    # Write data to a log
        file = open('/home/pi/PresIMULog.csv', 'a')
        file.write(str(CurrTimestamp) + "," + str(data["temperature"]) + "," + str(data["pressure"]) + "," + str(computeHeight(data["pressure"])) + "," + str(accel[0]) + "," + str(accel[1]) + "," + str(accel[2]) + "," + str(gyro[0]) + "," + str(gyro[1]) + "," + str(gyro[2]) + "," + str(compass[0]) + "," + str(compass[1]) + "," + str(compass[2]) + "," + str(math.degrees(fusionPose[0])) + "," + str(math.degrees(fusionPose[1])) + "," + str(math.degrees(fusionPose[2])) + "," + str((round(ina.voltage(), 2))) + "," + str((round(((ina.current()*(-1))/1000), 3))) + "," + str((round(ina.power()/1000, 2))) + "\n")
        file.close()

    if debug:
    #DEBUG - Print all the results
        print("\nGyro Position: gx: %f  gy: %f  gz: %f" % ((gyro[0]), (gyro[1]), (gyro[2])))
        print("\nAccelerometer Position: ax: %f  ay: %f  az: %f" % ((accel[0]), (accel[1]), (accel[2])))
        print("\nMagnetometar Position: mx: %f  my: %f  mz: %f" % ((compass[0]), (compass[1]), (compass[2])))
        print("\nFused Position: r: %f  p: %f  y: %f" % (math.degrees(fusionPose[0]), math.degrees(fusionPose[1]), math.degrees(fusionPose[2])))
        if (data["pressureValid"]):
            print("Pressure: %f" % (data["pressure"]))
            print("Altitude: %f" % (computeHeight(data["pressure"])))
        if (data["temperatureValid"]):
            print("Temperature: %f" % (data["temperature"]))

    time.sleep(0.15)
