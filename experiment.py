import time
import pycozmo

from track_object import TrackObject


class Robot():
    """Class to control Anki Cozmo Robot"""
    def __init__(self, cli):
        self.cli = cli
        self.cliff_detected = False
        self.speed = 30

        cli.set_lift_height(0.5)
        cli.add_handler(pycozmo.event.EvtCliffDetectedChange, self.on_cliff_detected)

    def on_gesture_left(self):
        """
        Backup a little, turn left, go forward.
        """
        # Speed has units of mmps.
        self.cli.drive_wheels(lwheel_speed=-self.speed, rwheel_speed=-self.speed, duration=0.5)
        self.cli.drive_wheels(lwheel_speed=-self.speed, rwheel_speed=self.speed, duration=2.5)

    def on_gesture_right(self):
        """
        Backup a little, turn left, go forward.
        """
        # Speed has units of mmps.
        self.cli.drive_wheels(lwheel_speed=-self.speed, rwheel_speed=-self.speed, duration=0.5)
        self.cli.drive_wheels(lwheel_speed=self.speed, rwheel_speed=-self.speed, duration=2.5)

    def on_gesture_in():
        pass

    def on_gesture_out():
        pass

    def default_behavior(self):
        self.cli.drive_wheels(lwheel_speed=self.speed, rwheel_speed=self.speed, duration=1.0)
        

    def on_cliff_detected(self, cli, state : bool):
        if state:
            self.cli.stop_all_motors()
            self.cliff_detected = True

def main():
    with pycozmo.connect() as cli:
        robot = Robot(cli)
        
        height = (pycozmo.MAX_LIFT_HEIGHT.mm - pycozmo.MIN_LIFT_HEIGHT.mm)/2
        cli.set_lift_height(pycozmo.MAX_LIFT_HEIGHT.mm)
        cli.move_lift(10.0)
        time.sleep(1)
        object_tracker = TrackObject()

        cliff_counter = 0;

        # Loop until program end.
        while True:
            if cliff_counter > 2:
                break
            
            if not robot.cliff_detected:
                # Run default robot behavior.
                robot.default_behavior()
            else:
                # On cliff detected, loop over frames until a gesture is detected.
                gesture = object_tracker.detect_gesture()
                robot.cliff_detected = False
                cliff_counter += 1

                # Call robot stuff based on gesture.
                if gesture == 'terminate':
                    object_tracker.teardown()
                    break
                elif gesture == 'left':
                    robot.on_gesture_left()
                elif gesture == 'right':
                    robot.on_gesture_right()
        print("Done!")


if __name__ == "__main__":
    main()
