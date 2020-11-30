import time
import types
from typing import Optional

import numpy as np
import pycozmo

from track_object import TrackObject


def turn_in_place(self, angle_rad: float, speed_rad_per_sec: float,
                  accel_rad_per_sec2: Optional[float] = 0.0, angle_tolerance_rad: Optional[float] = 0.0,
                  is_absolute: Optional[bool] = False) -> None:

    pkt = pycozmo.protocol_encoder.TurnInPlace(angle_rad=angle_rad, speed_rad_per_sec=speed_rad_per_sec, 
                                       accel_rad_per_sec2=accel_rad_per_sec2, angle_tolerance_rad=angle_tolerance_rad,
                                       is_absolute=is_absolute)
    
    self.conn.send(pkt)


def on_cliff_detected(cli, state: bool):
    if state:
        cli.stop_all_motors()


def have_faith():

    with pycozmo.connect() as cli:

        def on_gesture_left(cli=cli):
            """
            Backup a little, turn left, go forward.
            """
            # Speed has units of mmps.
            speed = -10
            cli.drive_wheels(lwheel_speed=speed, rwheel_speed=speed, duration=1)
            angle = np.pi/2
            cli.turn_in_place(angle_rad=angle, speed_rad_per_sec=np.pi/4)
            speed = 20
            cli.drive_wheels(lwheel_speed=speed, rwheel_speed=speed)

        def on_gesture_right(cli=cli):
            """
            Backup a little, turn left, go forward.
            """
            # Speed has units of mmps.
            speed = -10
            cli.drive_wheels(lwheel_speed=speed, rwheel_speed=speed, duration=1)
            angle = -np.pi/2
            cli.turn_in_place(angle_rad=angle, speed_rad_per_sec=np.pi/4)
            speed = 20
            cli.drive_wheels(lwheel_speed=speed, rwheel_speed=speed)

        def on_gesture_in(cli=cli):
            pass

        def on_gesture_out(cli=cli):
            pass


        cli.turn_in_place = types.MethodType(turn_in_place, cli)
        cli.add_handler(pycozmo.event.EvtCliffDetectedChange, on_cliff_detected)

        object_tracker = TrackObject(
            on_gesture_left=on_gesture_left,
            on_gesture_right=on_gesture_right,
            on_gesture_in=on_gesture_in,
            on_gesture_out=on_gesture_out,
            )

        # Note: is always listening for gestures. May be problematic.
        terminate = False
        while not terminate:

            terminate = object_tracker.next_frame()


if __name__ == "__main__":
    have_faith()