import types
from typing import Optional
import pycozmo
import numpy as np

from track_object import TrackObject


class Robot():
    """Class to control Anki Cozmo Robot"""
    def __init__(self, cli):
        self.cli = cli

        cli.turn_in_place = types.MethodType(self.turn_in_place, self.cli)
        cli.add_handler(pycozmo.event.EvtCliffDetectedChange,
                        self.on_cliff_detected)

    def on_gesture_left(self):
        """
        Backup a little, turn left, go forward.
        """
        # Speed has units of mmps.
        speed = -10
        self.cli.drive_wheels(lwheel_speed=speed,
                              rwheel_speed=speed, duration=3.0)
        angle = np.pi/2
        # self.turn_in_place(angle_rad=angle, speed_rad_per_sec=np.pi/4)
        speed = 20
        self.cli.drive_wheels(lwheel_speed=speed, rwheel_speed=speed, duration=3.0)

    def on_gesture_right(self):
        """
        Backup a little, turn left, go forward.
        """
        # Speed has units of mmps.
        speed = -10
        self.cli.drive_wheels(lwheel_speed=speed,
                              rwheel_speed=speed, duration=3.0)
        angle = -np.pi/2
        # self.turn_in_place(angle_rad=angle, speed_rad_per_sec=np.pi/4)
        speed = 20
        self.cli.drive_wheels(lwheel_speed=speed, rwheel_speed=speed, duration=3.0)

    def on_gesture_in():
        pass

    def on_gesture_out():
        pass

    def turn_in_place(self, angle_rad: float, speed_rad_per_sec: float,
                      accel_rad_per_sec2: Optional[float] = 0.0, angle_tolerance_rad: Optional[float] = 0.0,
                      is_absolute: Optional[bool] = False) -> None:

        pkt = pycozmo.protocol_encoder.TurnInPlace(angle_rad=angle_rad, speed_rad_per_sec=speed_rad_per_sec,
                                                   accel_rad_per_sec2=accel_rad_per_sec2, angle_tolerance_rad=angle_tolerance_rad,
                                                   is_absolute=is_absolute)

        self.cli.conn.send(pkt)

    def on_cliff_detected(self, state: bool):
        if state:
            self.cli.stop_all_motors()


def main():
    with pycozmo.connect() as cli:
        robot = Robot(cli)
        object_tracker = TrackObject(robot=robot)

        # Note: is always listening for gestures. May be problematic.
        terminate = False
        while not terminate:
            terminate = object_tracker.next_frame()


if __name__ == "__main__":
    main()
