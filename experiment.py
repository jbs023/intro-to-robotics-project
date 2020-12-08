import time
import pycozmo

from track_object import TrackObject


class Robot():
    """Class to control Anki Cozmo Robot"""
    def __init__(self, cli):
        self.cli = cli
        self.cliff_detected = False
        self.speed = 30
        self.accel = 15
        self.decel = 15

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
        Backup a little, turn right, go forward.
        """
        # Speed has units of mmps.
        self.cli.drive_wheels(lwheel_speed=-self.speed, rwheel_speed=-self.speed, duration=0.5)
        self.cli.drive_wheels(lwheel_speed=self.speed, rwheel_speed=-self.speed, duration=2.5)

    def on_gesture_in():
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
                speed_mmps=self.speed, accel_mmps2=self.accel, decel_mmps2=self.decel
            )
            cli.conn.send(pkt)
        time.sleep(30)

        # Turn right.
        self.cli.drive_wheels(lwheel_speed=self.speed, rwheel_speed=-self.speed, duration=2.5)


    def on_gesture_out():
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
                speed_mmps=self.speed, accel_mmps2=self.accel, decel_mmps2=self.decel
            )
            cli.conn.send(pkt)
        time.sleep(30)

        # Turn left.
        self.cli.drive_wheels(lwheel_speed=-self.speed, rwheel_speed=self.speed, duration=2.5)


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

        # Loop until program terminates.
        while True:
            if cliff_counter > 6:
                break
            gesture = None
            obstacle_detected = False # detect_obstacle()

            # Request gesture to navigate cliff.
            if robot.cliff_detected:
                # On cliff detected, loop over frames until a gesture is detected.
                robot.cliff_detected = False
                cliff_counter += 1
                while gesture not in ['terminate', 'left', 'right']:
                    gesture = object_tracker.detect_gesture()
                    
            # Request gesture to navigate obstacle.
            elif obstacle_detected:
                while gesture not in ['terminate', 'in', 'out']:
                    gesture = object_tracker.detect_gesture()

            # Call robot stuff based on gesture.
            if gesture == 'terminate':
                object_tracker.teardown()
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
