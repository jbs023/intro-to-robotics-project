import time
import pycozmo
import numpy as np

from track_object import TrackObject
from ml_object_detection import MLObjectDetector


class Robot():
    """Class to control Anki Cozmo Robot"""

    def __init__(self, cli):
        self.cli = cli

        # Need for object detection
        self.object_detector = MLObjectDetector("yolo-coco")
        self.object_detector_counter = 0
        self.latest_im = None
        self.current_step = False
        self.previous_step = False
        self.object_detected = False

        self.cliff_detected = False
        self.speed = 30
        self.accel = 15
        self.decel = 15

        cli.set_lift_height(10.0)
        cli.set_head_angle(0.0)
        time.sleep(1)

        # Enable camera.
        cli.enable_camera(color=True)
        time.sleep(1)

        # Run with 14 FPS. This is the frame rate of the robot camera.
        timer = pycozmo.util.FPSTimer(14)
        cli.add_handler(pycozmo.event.EvtCliffDetectedChange, self.on_cliff_detected)
        cli.add_handler(pycozmo.event.EvtNewRawCameraImage, self.on_camera_image)

    def on_gesture_left(self):
        """
        Backup a little, turn left.
        """
        # Speed has units of mmps.
        self.cli.drive_wheels(lwheel_speed=-self.speed, rwheel_speed=-self.speed, duration=0.5)
        self.cli.drive_wheels(lwheel_speed=-self.speed, rwheel_speed=self.speed, duration=2.5)

    def on_gesture_right(self):
        """
        Backup a little, turn right.
        """
        # Speed has units of mmps.
        self.cli.drive_wheels(lwheel_speed=-self.speed, rwheel_speed=-self.speed, duration=0.5)
        self.cli.drive_wheels(lwheel_speed=self.speed, rwheel_speed=-self.speed, duration=2.5)

    def on_gesture_in(self):
        """
        Follow a path around to the right.
        """
        coords = np.array([
            [0, 0],
            [300, 0],
            [300, 300],
            [0, 300],
        ])

        for coord_index in range(len(coords) - 1):
            from_x, from_y = coords[coord_index]
            to_x, to_y = coords[coord_index + 1]
            pkt = pycozmo.protocol_encoder.AppendPathSegLine(
                from_x=from_x, from_y=from_y,
                to_x=to_x, to_y=to_y,
                speed_mmps=self.speed, accel_mmps2=self.accel, decel_mmps2=self.decel)
            self.cli.conn.send(pkt)

        pkt = pycozmo.protocol_encoder.ExecutePath()
        self.cli.conn.send(pkt)
        time.sleep(30)

        # Turn right.
        self.cli.drive_wheels(lwheel_speed=self.speed, rwheel_speed=-self.speed, duration=2.5)

    def on_gesture_out(self):
        """
        Follow a path around to the left.
        """

        coords = np.array([
            [0, 0],
            [-300, 0],
            [-300, 300],
            [0, 300],
        ])

        for coord_index in range(len(coords) - 1):
            from_x, from_y = coords[coord_index]
            to_x, to_y = coords[coord_index + 1]
            pkt = pycozmo.protocol_encoder.AppendPathSegLine(
                from_x=from_x, from_y=from_y,
                to_x=to_x, to_y=to_y,
                speed_mmps=self.speed, accel_mmps2=self.accel, decel_mmps2=self.decel)
            self.cli.conn.send(pkt)

        pkt = pycozmo.protocol_encoder.ExecutePath()
        self.cli.conn.send(pkt)
        time.sleep(30)
        
        # Turn left.
        self.cli.drive_wheels(lwheel_speed=-self.speed, rwheel_speed=self.speed, duration=2.5)

    def default_behavior(self):
        self.cli.drive_wheels(lwheel_speed=self.speed, rwheel_speed=self.speed, duration=1.0)

    def on_cliff_detected(self, cli, state: bool):
        if state:
            self.cli.stop_all_motors()
            self.cliff_detected = True

    def on_camera_image(self, cli, new_im):
        """ Handle new images, coming from the robot. """
        self.latest_im = new_im
        self.object_detector_counter += 1
        if self.object_detector_counter % 14 == 0 and not (self.object_detected or self.cliff_detected):
            self.previous_step = self.current_step
            distance = self.object_detector.detect_object(self.latest_im)

            if distance != -1:
                self.current_step = True
            else:
                self.current_step = False

            #Stop robot if detected last time step and not detected this time 
            if self.previous_step == True and self.current_step == False:
                print(self.previous_step, self.current_step)
                self.cli.stop_all_motors()
                self.latest_im.show()
                self.current_step = False
                self.previous_step = False
                self.object_detected = True


def main():
    with pycozmo.connect() as cli:
        robot = Robot(cli)
        gesture_detector = TrackObject()

        # Loop until program terminates.
        cliff_counter = 0
        while True:
            if cliff_counter > 6:
                break

            gesture = None

            # Request gesture to navigate cliff.
            if robot.cliff_detected:
                robot.cliff_detected = False
                cliff_counter += 1
                while gesture not in ['terminate', 'left', 'right']:
                    gesture = gesture_detector.detect_gesture()
                    
            # Request gesture to navigate obstacle.
            elif robot.object_detected:
                robot.object_detected = False
                while gesture not in ['terminate', 'left', 'right', 'in', 'out']:
                    gesture = gesture_detector.detect_gesture()

            # Perform robot action based on gesture.
            if gesture == 'terminate':
                gesture_detector.teardown()
                break
            elif gesture == 'left':
                robot.on_gesture_left()
            elif gesture == 'right':
                robot.on_gesture_right()
            elif gesture == 'in':
                robot.on_gesture_in()
            elif gesture == 'out':
                robot.on_gesture_out()
            # Perform default behavior.
            else:
                # No cliff or obstacle detected.
                robot.default_behavior()
        
        print("Mission complete!")


if __name__ == "__main__":
    main()
