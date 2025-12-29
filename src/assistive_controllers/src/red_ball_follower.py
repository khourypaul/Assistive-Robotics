#!/usr/bin/env python3
import math

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image


class RedBallFollower:
    def __init__(self):
        self.image_topic = rospy.get_param("~image_topic", "rgb_to_depth/image_raw")
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "cmd_vel_arm")
        self.debug_image = rospy.get_param("~debug_image", False)
        self.debug_image_topic = rospy.get_param("~debug_image_topic", "red_ball/debug_image")

        self.kx = float(rospy.get_param("~kx", 0.0015))
        self.ky = float(rospy.get_param("~ky", 0.0015))
        self.max_v = float(rospy.get_param("~max_v", 0.05))
        self.deadband_px = int(rospy.get_param("~deadband_px", 10))
        self.min_area = float(rospy.get_param("~min_area", 150.0))
        self.stop_on_lost = bool(rospy.get_param("~stop_on_lost", True))
        self.lost_timeout = float(rospy.get_param("~lost_timeout", 0.5))

        self.bridge = CvBridge()
        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.debug_pub = None
        if self.debug_image:
            self.debug_pub = rospy.Publisher(self.debug_image_topic, Image, queue_size=1)

        self.last_seen = rospy.Time(0)
        self.latest_cmd = Twist()
        self.have_target = False

        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(0.05), self.timer_cb)

    def image_cb(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        height, width = img.shape[:2]
        cx0, cy0 = width // 2, height // 2

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Red wraps in HSV, so combine low and high hue bands.
        mask1 = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.have_target = False
            return

        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < self.min_area:
            self.have_target = False
            return

        M = cv2.moments(c)
        if M["m00"] == 0:
            self.have_target = False
            return

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        ex = cx - cx0
        ey = cy - cy0
        if abs(ex) < self.deadband_px:
            ex = 0
        if abs(ey) < self.deadband_px:
            ey = 0

        cmd = Twist()
        cmd.linear.y = self._clamp(-self.kx * ex, self.max_v)
        cmd.linear.z = self._clamp(-self.ky * ey, self.max_v)

        self.latest_cmd = cmd
        self.have_target = True
        self.last_seen = rospy.Time.now()

        if self.debug_pub:
            cv2.circle(img, (cx, cy), 6, (0, 255, 0), -1)
            cv2.circle(img, (cx0, cy0), 6, (255, 0, 0), -1)
            debug_msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
            self.debug_pub.publish(debug_msg)

    def timer_cb(self, _):
        now = rospy.Time.now()
        if self.have_target and (now - self.last_seen).to_sec() <= self.lost_timeout:
            ##self.cmd_pub.publish(self.latest_cmd)
            return

        if self.stop_on_lost:
            self.cmd_pub.publish(Twist())

    @staticmethod
    def _clamp(value, limit):
        return max(-limit, min(limit, value))


def main():
    rospy.init_node("red_ball_follower")
    RedBallFollower()
    rospy.spin()


if __name__ == "__main__":
    main()
