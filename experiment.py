import types
from typing import Optional
import pycozmo
import numpy as np
from functools import partial

from track_object import TrackObject


class Robot():
    """Class to control Anki Cozmo Robot"""
    def __init__(self, cli):
        self.cli = cli
        self.cliff_detected = False

        cli.turn_in_place = types.MethodType(self.turn_in_place, self.cli)
        cli.add_handler(pycozmo.event.EvtCliffDetectedChange,
                        partial(self.on_cliff_detected, self))

    def on_gesture_left(self):
        """
        Backup a little, turn left, go forward.
        """
        # Speed has units of mmps.
        speed = -10
        self.cli.drive_wheels(lwheel_speed=speed,
                              rwheel_speed=speed, duration=3.0)
        angle = -np.pi/2
        speed = 10
        duration = 3.0
        self.cli.drive_wheels(lwheel_speed=-speed,
                              rwheel_speed=speed, duration=duration)

    def on_gesture_right(self):
        """
        Backup a little, turn left, go forward.
        """
        # Speed has units of mmps.
        speed = -10
        self.cli.drive_wheels(lwheel_speed=speed,
                              rwheel_speed=speed, duration=3.0)
        angle = -np.pi/2
        speed = 10
        duration = 3.0
        self.cli.drive_wheels(lwheel_speed=speed,
                              rwheel_speed=-speed, duration=duration)

    def on_gesture_in():
        pass

    def on_gesture_out():
        pass

    def default_behavior(self):
        self.cli.drive_wheels(lwheel_speed=speed, rwheel_speed=speed, duration=1)
        

    def on_cliff_detected(self, state : bool):
        if state:
            self.cli.stop_all_motors()
            self.cliff_detected = True

def main():



    with pycozmo.connect() as cli:

        robot = Robot(cli)
        object_tracker = TrackObject()

        # Loop until program end.
        while True:

            if not robot.cliff_detected:
                # Run default robot b`ehavior.
                robot.default_behavior() # e.g. drive forward 10 seconds
            else:
                robot.cliff_detected = False
                # On cliff detected, loop over frames until a gesture is detected.
                gesture = object_tracker.detect_gesture()
                # Call robot stuff based on gesture.
                if gesture == 'terminate':
                    object_tracker.teardown()
                    break
                elif gesture == 'left':
                    robot.on_gesture_left()
                elif gesture == 'right':
                    robot.on_gesture_right()




if __name__ == "__main__":
    main()
