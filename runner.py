import picamera2
import time
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from gpiozero import AngularServo, DistanceSensor

model = tf.keras.models.load_model("82.keras")

camera = picamera2.Picamera2()
camera.start()

right_servo = AngularServo(27, min_pulse_width=0.10/100, max_pulse_width=0.2/100, frame_width=0.020, min_angle=0, max_angle=180)
left_servo = AngularServo(17, min_pulse_width=0.10/100, max_pulse_width=0.20/100, frame_width=0.020, min_angle=0, max_angle=90)

def right():
    right_servo.angle = 175
    left_servo.angle = 10


def left():
    right_servo.angle = 60
    left_servo.angle = 70

def straight():
    right_servo.angle = 175
    left_servo.angle = 70

class_names = {0: 'left', 1:'right', 2:'straight'}

try:
    while True:
        image = camera.capture_array()
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        resized_frame = cv2.resize(image, (480, 270))
    
        frame_arr = np.array(resized_frame)
    
        frame_array = frame_arr[np.newaxis, :, :, np.newaxis]
    
        frame_tensor = tf.convert_to_tensor(frame_array)
    
        predictions = model.predict(frame_tensor)
    
        prediction = class_names[np.argmax(predictions)]

        if prediction == 'left':
            left()
        elif prediction == 'right':
            right()
        elif prediction == 'straight':
            straight()   

        print(prediction)

        time.sleep(0.5)

        
except KeyboardInterrupt:
    pass

camera.stop()
