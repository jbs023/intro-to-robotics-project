"""
This code is based on the work presented in this tutorial:
https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/
"""

from collections import deque
import time
import numpy as np
import cv2
import imutils
from imutils.video import VideoStream


class TrackObject:

    """Tracks an object in a live video or recorded video.

    Args:
        Required:

        robot (Robot object): Object that connects to the robot and controls it's actions
        video_src_index (int): ID of available cameras on users computer
    """

    def __init__(self, video_src_index=0):
        # Variable that may need to be more flexible
        self.video_src_index = video_src_index

        # This variables are fairly static
        self.color_lower_bound = (29, 86, 6)
        self.color_upper_bound = (64, 255, 255)
        self.draw_bounding_color = (0, 255, 255)
        self.draw_bounding_thickness = 2
        self.draw_center_color = (0, 0, 255)
        self.draw_center_thickness = -1
        self.draw_center_radius = 5
        self.num_trail_frames = 64
        self.draw_trail_max_thickness = 2.5
        self.draw_trail_color = (0, 0, 255)
        self.resize_to_width = 600
        self.resize_to_height = None
        self.blur_kernel_width = 11
        self.blur_kernel_height = 11
        self.video_file_path = None

        # Connect video_source.
        if self.video_file_path is None:
            self.video_source = VideoStream(src=video_src_index).start()
        else:
            self.video_source = cv2.VideoCapture(self.video_file_path)
        # Allows video_source to warm up. No idea what that means.
        time.sleep(2.0)

        self.trail = deque(maxlen=self.num_trail_frames)
        self.radii = deque(maxlen=self.num_trail_frames)

        self.last_lateral_gesture_text = None
        self.last_axial_gesture_text = None
        self.frames_since_lateral_gesture = 0
        self.frames_since_axial_gesture = 0
        self.lateral_gesture_text = None
        self.axial_gesture_text = None

    @staticmethod
    def detect_lateral_gesture(trail,
                               gesture_size_proportion=1,
                               vertical_strike_size_proportion=0.333,
                               max_vertical_strike_proportion=0.167,
                               max_avg_vertical_deltas_proportion=0.5,
                               lateral_threshold_proportion=0.95):
        """Detects a lateral gesture as the object moves across the screen.

        Args:
            trail (list): list of points, where each point represents the previous center of the object
            gesture_size_proportion (int, optional): Defaults to 1.
            vertical_strike_size_proportion (float, optional): Defaults to 1/3.
            max_vertical_strike_proportion (float, optional): Defaults to 1/6.
            max_avg_vertical_deltas_proportion (float, optional): Defaults to 1/2.
            lateral_threshold_proportion (float, optional): Defaults to 0.95.

        Returns:
            string: The general lateral direction the ball is moving.
        """
        # Scale inputs.
        gesture_size = int(gesture_size_proportion * trail.maxlen)
        vertical_strike_size = int(
            vertical_strike_size_proportion * gesture_size)
        max_vertical_strikes = int(
            max_vertical_strike_proportion * gesture_size)
        max_avg_vertical_deltas = int(max_avg_vertical_deltas_proportion *
                                      gesture_size)

        # Disqualify gesture if the trail is insufficiently long.
        if len(trail) < gesture_size or None in trail:
            return None
        recent_trail = np.array(trail)[:gesture_size]

        # Disqualify gesture based on vertical_deltas.
        vertical_deltas = np.abs(recent_trail[:-1, 1] - recent_trail[1:, 1])
        num_vertical_strikes = np.sum(vertical_deltas > vertical_strike_size)
        avg_vertical_deltas = np.average(vertical_deltas)

        if num_vertical_strikes > max_vertical_strikes or avg_vertical_deltas > max_avg_vertical_deltas:
            return None

        # Check lateral deltas.
        lateral_deltas = recent_trail[:-1, 0] - recent_trail[1:, 0]
        num_left = np.sum(lateral_deltas < 0)
        num_right = np.sum(lateral_deltas > 0)

        # Exchange left and right for camera perspective mirror.
        num_left, num_right = (num_right, num_left)
        num_nonzero = num_left + num_right

        if num_left / num_nonzero >= lateral_threshold_proportion and num_left > num_right:
            return "left"
        elif num_right / num_nonzero >= lateral_threshold_proportion and num_right > num_left:
            return "right"
        else:
            return None

    @staticmethod
    def detect_axial_gesture(radii,
                             gesture_size_proportion=1 / 2,
                             radial_threshold_proportion=0.9):
        """Detects an axial gesture when the radius of the minimum enclosing circle changes.

        Args:
            radii (list): list of values that represent size of the radius over time.
            gesture_size_proportion ([type], optional): Defaults to 1/2.
            radial_threshold_proportion (float, optional): Defaults to 0.9.

        Returns:
            string: A gesture proportional to how the radius is changing.
        """
        # Scale inputs.
        gesture_size = int(gesture_size_proportion * radii.maxlen)

        # Disqualify gesture if the trail is insufficiently long.
        if len(radii) < gesture_size or None in radii:
            return None
        recent_radii = np.array(radii)[:gesture_size]

        radial_deltas = recent_radii[:-1] - recent_radii[1:]
        num_in = np.sum(radial_deltas > 0)
        num_out = np.sum(radial_deltas < 0)
        num_nonzero = num_in + num_out

        if num_in / num_nonzero >= radial_threshold_proportion and num_in > num_out:
            return "in"
        elif num_out / num_nonzero >= radial_threshold_proportion and num_out > num_in:
            return "out"
        else:
            return None

    def next_frame(self):

        gesture = None

        # Read the current frame.
        if self.video_file_path is None:
            # If video_source is a imutils.video.VideoStream object, i.e. connected to a live camera
            frame = self.video_source.read()
        else:
            # If video_source is a cv2.VideoCapture object, i.e. points to a video file
            frame = self.video_source.read()[1]

        # If we failed to read a frame, the video_source has been exhausted or terminated.
        if frame is None:
            self.teardown()
            return True  # terminate => True

        # Resize frame.
        frame = imutils.resize(frame,
                               width=self.resize_to_width,
                               height=self.resize_to_height)
        # Blur frame.
        blurred_frame = cv2.GaussianBlur(frame,
                                         ksize=(self.blur_kernel_width,
                                                self.blur_kernel_height),
                                         sigmaX=0,
                                         sigmaY=0)
        # Convert blurred_frame from BGR to HSV.
        hsv_blurred_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

        # Construct a mask for colors within the color thresholds using hsv_blurred_frame.
        color_mask = cv2.inRange(hsv_blurred_frame, self.color_lower_bound,
                                 self.color_upper_bound)
        # Erode and dilate the color_mask.
        color_mask = cv2.erode(color_mask, kernel=None, iterations=2)
        color_mask = cv2.dilate(color_mask, kernel=None, iterations=2)

        # Find contours in color_mask.
        color_mask_contours = cv2.findContours(color_mask.copy(),
                                               mode=cv2.RETR_EXTERNAL,
                                               method=cv2.CHAIN_APPROX_SIMPLE)
        color_mask_contours = imutils.grab_contours(color_mask_contours)

        # Initialize center and radius to None in case it is not set.
        center = None
        valid_radius = None

        # If at least 1 contour was found:
        if len(color_mask_contours) >= 1:
            largest_contour = max(color_mask_contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)

            # If largest_contour is sufficiently large, find and record its center,
            # and draw the circle and center on the frame.
            if radius > 10:
                valid_radius = radius
                largest_contour_moments = cv2.moments(largest_contour)
                center = (int(largest_contour_moments["m10"] /
                              largest_contour_moments["m00"]),
                          int(largest_contour_moments["m01"] /
                              largest_contour_moments["m00"]))

                # Draw the circle.
                cv2.circle(frame,
                           center=(int(x), int(y)),
                           radius=int(radius),
                           color=self.draw_bounding_color,
                           thickness=self.draw_bounding_thickness)
                # Draw the center.
                cv2.circle(frame,
                           center=center,
                           radius=self.draw_center_radius,
                           color=self.draw_center_color,
                           thickness=self.draw_center_thickness)

        self.trail.appendleft(center)
        self.radii.appendleft(valid_radius)

        # Detect and draw gestures.
        # Lateral gesture.
        lateral_gesture = self.detect_lateral_gesture(self.trail)
        if lateral_gesture is not None:
            self.trail.clear()
            self.radii.clear()
            self.lateral_gesture_text = f"lateral: {lateral_gesture}"
            self.last_lateral_gesture_text = self.lateral_gesture_text
            self.frames_since_lateral_gesture = 0
            if lateral_gesture == 'left':
                gesture = 'left'
            elif lateral_gesture == 'right':
                gesture = 'right'
        else:
            self.frames_since_lateral_gesture += 1
            if self.frames_since_lateral_gesture < self.trail.maxlen / 2:
                self.lateral_gesture_text = self.last_lateral_gesture_text
            else:
                self.lateral_gesture_text = None

        # Print gesture to frame.
        position = (10, 50)
        cv2.putText(img=frame,
                    text=self.lateral_gesture_text,
                    org=position,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(209, 80, 0, 255),
                    thickness=3)

        # Radial gesture.
        axial_gesture = self.detect_axial_gesture(self.radii)
        if axial_gesture is not None:
            self.trail.clear()
            self.radii.clear()
            self.axial_gesture_text = f"axial: {axial_gesture}"
            self.last_axial_gesture_text = self.axial_gesture_text
            self.frames_since_axial_gesture = 0
            if axial_gesture == 'in' and self.on_gesture_in is not None:
                gesture = 'in'
            elif axial_gesture == 'out' and self.on_gesture_out is not None:
                gesture = 'out'
        else:
            self.frames_since_axial_gesture += 1
            if self.frames_since_axial_gesture < self.trail.maxlen / 2:
                self.axial_gesture_text = self.last_axial_gesture_text
            else:
                self.axial_gesture_text = None

        # Print gesture to frame.
        position = (10, 100)
        cv2.putText(img=frame,
                    text=self.axial_gesture_text,
                    org=position,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(209, 80, 0, 255),
                    thickness=3)

        # Draw trail.
        # Connect each point in trail to the next with increasing thickness. This traversal goes backwards in time.
        for point_index in range(len(self.trail) - 1):
            this_point = self.trail[point_index]
            next_point = self.trail[point_index + 1]
            # Do not draw to or from points that were not collected.
            if this_point is None or next_point is None:
                continue

            # Compute tapering thickness.
            draw_trail_thickness = int(
                np.sqrt(self.trail.maxlen / float(point_index + 1)) *
                self.draw_trail_max_thickness)
            # Draw line from this_point to next_point.
            cv2.line(frame, this_point, next_point, self.draw_trail_color,
                     draw_trail_thickness)

        # Display the frame.
        cv2.imshow("Frame", frame)

        # Accept possible keypress and respond.
        # Wait at least 1 millisecond and mask the key's code to get the last 4 digits.
        key = cv2.waitKey(1) & 0xFF

        # If the user pressed 'q', exit the loop and terminate.
        if key == ord("q"):
            self.teardown()
            return 'terminate'
        else:
            return gesture


    def detect_gesture(self):

        gesture = None
        while gesture is None:
            gesture = self.next_frame()

        return gesture


    def teardown(self):
        # If video_source is not connected to a video file, stop the camera video stream.
        # Else, release the video file.
        if self.video_file_path is None:
            self.video_source.stop()
        else:
            self.video_source.release()

        # Close all windows.
        cv2.destroyAllWindows()