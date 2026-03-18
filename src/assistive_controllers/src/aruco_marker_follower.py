#!/usr/bin/env python

"""
Author: (generated for Assistive Robotics project)
Node: aruco_marker_follower
Description:
    Detects a single ArUco marker in a camera image, estimates its 6-DOF pose
    in the camera frame, broadcasts the result to TF2, then commands the robot
    end-effector toward the marker using Cartesian velocity control.

    Motion pipeline (same as body_single_joint_follower2):
        This node  ->  cmd_vel  (geometry_msgs/Twist, ~25 Hz, in base frame)
                   ->  twist_mux
                   ->  oarbot_redundancy_resolver
                   ->  freq_adjuster_arm  (Twist -> kinova_msgs/PoseVelocity at 100 Hz)
                   ->  j2n6s300_*_driver/in/cartesian_velocity

    TF prerequisites (overhead NUC camera path):
        cage_rgb_camera_link -> map -> d*_tf_base_link -> root_*_arm
            -> j2n6s300_*_link_base -> ... -> j2n6s300_*_end_effector
        Requires: nuc_kinect_with_aruco.launch  +  oarbot_blue_arm_with_ft.launch

    TF prerequisites (on-robot arm camera path):
        j2n6s300_*_*_camera_link -> j2n6s300_*_link_base -> ... -> j2n6s300_*_end_effector
        Requires: tf_arm_camera_broadcaster  +  kinova driver

Parameters:
    ~marker_id              : int   - ArUco marker ID to track (default 0)
    ~marker_size_m          : float - Physical marker side length in metres (default 0.05)
    ~aruco_dict_type        : str   - OpenCV ArUco dict name, e.g. "DICT_4X4_50" (default)
    ~image_topic            : str   - Input camera image topic
    ~camera_info_topic      : str   - Camera calibration topic (provides K, D)
    ~camera_frame_id        : str   - Override camera TF frame; leave "" to read from
                                      camera_info header.frame_id
    ~base_frame_id          : str   - Robot base frame used for velocity commands
                                      (e.g. "d1_tf_base_link" for oarbot_blue)
    ~ee_frame_id            : str   - End-effector TF frame
    ~approach_offset_m      : float - Stand-off distance from marker along marker +Z axis
                                      (default 0.10 m).  The robot stops here rather than
                                      driving into the marker.
    ~cmd_vel_topic          : str   - Topic to publish Twist commands on (default "cmd_vel")
    ~pub_rate               : float - Control loop rate in Hz (default 25.0)
    ~K_lin                  : float - Proportional gain for linear velocity (default 1.0)
    ~K_ang                  : float - Proportional gain for angular velocity (0 = off)
    ~max_lin_vel            : float - Maximum linear speed m/s (default 0.05)
    ~max_ang_vel            : float - Maximum angular speed rad/s (default 0.3)
    ~position_err_thres     : float - Linear dead-zone m (default 0.01)
    ~orientation_err_thres  : float - Angular dead-zone rad (default 0.05)
    ~marker_lost_timeout    : float - Seconds without detection before stopping (default 1.0)
    ~enable_motion          : bool  - Whether to send velocity commands (default False)
    ~debug_image_view       : bool  - Publish annotated debug image (default False)
    ~debug_image_topic      : str   - Topic for debug image
    ~marker_tf_prefix       : str   - TF frame prefix for broadcast marker (default "aruco_marker_")
    ~tf_broadcast_marker    : bool  - Whether to broadcast marker frame to TF2 (default True)

Subscribes to:
    image_topic         (sensor_msgs/Image)
    camera_info_topic   (sensor_msgs/CameraInfo)

Publishes to:
    cmd_vel_topic       (geometry_msgs/Twist)
    debug_image_topic   (sensor_msgs/Image)   [optional]

Broadcasts to tf2:
    camera_frame_id  ->  aruco_marker_<marker_id>

Services provided:
    ~toggle_motion   (std_srvs/SetBool) - Enable/disable motion at runtime
"""

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError

import tf2_ros
import tf_conversions
import tf.transformations

import sensor_msgs.msg
import geometry_msgs.msg

from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse

# ── ArUco dictionary map (mirrors tf_overhead_camera_aruco_broadcaster.py) ──
ARUCO_DICT = {
    "DICT_4X4_50":       cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100":      cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250":      cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000":     cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50":       cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100":      cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250":      cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000":     cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50":       cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100":      cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250":      cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000":     cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50":       cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100":      cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250":      cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000":     cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


class ArucoMarkerFollower(object):
    """Detects an ArUco marker and moves the end-effector toward it."""

    def __init__(self):
        rospy.init_node('aruco_marker_follower', anonymous=False)

        # ── ArUco parameters ──────────────────────────────────────────────
        self.marker_id     = int(rospy.get_param('~marker_id', 0))
        self.marker_size_m = float(rospy.get_param('~marker_size_m', 0.05))
        dict_type          = rospy.get_param('~aruco_dict_type', 'DICT_4X4_50')

        if dict_type not in ARUCO_DICT:
            rospy.logfatal("[ArucoMarkerFollower] Unknown aruco_dict_type: '%s'", dict_type)
            raise SystemExit(1)

        self.aruco_dict   = cv2.aruco.Dictionary_get(ARUCO_DICT[dict_type])
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        # ── Camera parameters ─────────────────────────────────────────────
        self.image_topic        = rospy.get_param('~image_topic',       '/rgb/image_raw')
        self.camera_info_topic  = rospy.get_param('~camera_info_topic', '/rgb/camera_info')
        self.camera_frame_override = rospy.get_param('~camera_frame_id', '')

        # Set once camera_info arrives
        self.camera_matrix  = None  # 3x3 ndarray
        self.dist_coeffs    = None  # 1xN ndarray
        self.camera_frame_id = None

        # ── TF frame names ────────────────────────────────────────────────
        self.base_frame_id = rospy.get_param('~base_frame_id', 'd1_tf_base_link')
        self.ee_frame_id   = rospy.get_param('~ee_frame_id',   'j2n6s300_right_end_effector')
        prefix             = rospy.get_param('~marker_tf_prefix', 'aruco_marker_')
        self.marker_tf_name = prefix + str(self.marker_id)
        self.tf_broadcast_marker = bool(rospy.get_param('~tf_broadcast_marker', True))

        # ── Approach ──────────────────────────────────────────────────────
        # The target position is offset from the marker along the marker +Z axis
        # (ArUco Z points toward the camera, so this places the EE in front of
        # the marker on the camera side).
        self.approach_offset_m = float(rospy.get_param('~approach_offset_m', 0.10))

        # ── Control ───────────────────────────────────────────────────────
        self.pub_rate        = float(rospy.get_param('~pub_rate',        25.0))
        self.K_lin           = float(rospy.get_param('~K_lin',            1.0))
        self.K_ang           = float(rospy.get_param('~K_ang',            0.0))
        self.max_lin_vel     = float(rospy.get_param('~max_lin_vel',      0.05))
        self.max_ang_vel     = float(rospy.get_param('~max_ang_vel',      0.30))
        self.position_err_thres    = float(rospy.get_param('~position_err_thres',    0.010))
        self.orientation_err_thres = float(rospy.get_param('~orientation_err_thres', 0.050))
        self.marker_lost_timeout   = float(rospy.get_param('~marker_lost_timeout',   1.0))

        # ── Motion enable ─────────────────────────────────────────────────
        self.enable_motion = bool(rospy.get_param('~enable_motion', False))

        # ── Debug image ───────────────────────────────────────────────────
        self.debug_image_view  = bool(rospy.get_param('~debug_image_view',  False))
        self.debug_image_topic = rospy.get_param('~debug_image_topic', 'aruco_debug_image')

        # ── Command topic ─────────────────────────────────────────────────
        cmd_vel_topic = rospy.get_param('~cmd_vel_topic', 'cmd_vel')

        # ── Publishers ────────────────────────────────────────────────────
        self.pub_cmd_vel = rospy.Publisher(
            cmd_vel_topic, geometry_msgs.msg.Twist, queue_size=1)
        self.pub_debug_img = None
        if self.debug_image_view:
            self.pub_debug_img = rospy.Publisher(
                self.debug_image_topic, sensor_msgs.msg.Image, queue_size=1)

        # ── TF ────────────────────────────────────────────────────────────
        self.tf_buffer      = tf2_ros.Buffer()
        self.tf_listener    = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # ── cv_bridge ─────────────────────────────────────────────────────
        self.bridge = CvBridge()

        # ── State ─────────────────────────────────────────────────────────
        self.last_detection_time = None   # rospy.Time when marker was last seen

        # ── Services ─────────────────────────────────────────────────────
        rospy.Service('~toggle_motion', SetBool, self._toggle_motion_cb)

        # ── Subscribers ───────────────────────────────────────────────────
        rospy.Subscriber(
            self.camera_info_topic, sensor_msgs.msg.CameraInfo,
            self._camera_info_cb, queue_size=1)
        rospy.Subscriber(
            self.image_topic, sensor_msgs.msg.Image,
            self._image_cb, queue_size=1)

        # ── Control timer ─────────────────────────────────────────────────
        rospy.Timer(rospy.Duration(1.0 / self.pub_rate), self._control_loop)

        rospy.loginfo(
            "[ArucoMarkerFollower] Ready.  Tracking marker ID=%d  size=%.3f m  dict=%s",
            self.marker_id, self.marker_size_m, dict_type)
        rospy.loginfo(
            "[ArucoMarkerFollower] Base frame: '%s'  EE frame: '%s'",
            self.base_frame_id, self.ee_frame_id)
        rospy.loginfo(
            "[ArucoMarkerFollower] approach_offset=%.3f m  max_lin_vel=%.3f m/s",
            self.approach_offset_m, self.max_lin_vel)
        rospy.loginfo(
            "[ArucoMarkerFollower] Motion initially %s.  "
            "Toggle: rosservice call ~toggle_motion \"data: true\"",
            "ENABLED" if self.enable_motion else "DISABLED")

    # ─────────────────────────────────────────────────────────────────────────
    # Service callback
    # ─────────────────────────────────────────────────────────────────────────

    def _toggle_motion_cb(self, req):
        self.enable_motion = req.data
        state_str = "ENABLED" if req.data else "DISABLED"
        rospy.loginfo("[ArucoMarkerFollower] Motion %s", state_str)
        return SetBoolResponse(success=True,
                               message="Motion " + state_str.lower())

    # ─────────────────────────────────────────────────────────────────────────
    # camera_info callback – sets intrinsics once
    # ─────────────────────────────────────────────────────────────────────────

    def _camera_info_cb(self, msg):
        if self.camera_matrix is not None:
            return  # already have intrinsics

        self.camera_matrix = np.array(msg.K, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs   = np.array(msg.D, dtype=np.float64)

        if self.camera_frame_override:
            self.camera_frame_id = self.camera_frame_override
        else:
            self.camera_frame_id = msg.header.frame_id

        rospy.loginfo(
            "[ArucoMarkerFollower] Camera intrinsics received.  "
            "Camera frame: '%s'", self.camera_frame_id)

    # ─────────────────────────────────────────────────────────────────────────
    # Image callback – detect marker, broadcast TF
    # ─────────────────────────────────────────────────────────────────────────

    def _image_cb(self, msg):
        if self.camera_matrix is None:
            rospy.logwarn_throttle(
                5.0, "[ArucoMarkerFollower] Waiting for camera_info on '%s'...",
                self.camera_info_topic)
            return

        # Convert ROS image to OpenCV BGR
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("[ArucoMarkerFollower] CvBridge error: %s", str(e))
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params)

        marker_found = False

        if ids is not None:
            for i, mid in enumerate(ids.flatten()):
                if int(mid) != self.marker_id:
                    continue

                # ── Pose estimation in camera frame ───────────────────────
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners[i],
                    self.marker_size_m,
                    self.camera_matrix,
                    self.dist_coeffs)

                rvec_flat = rvec[0].flatten()   # (3,) Rodrigues vector
                tvec_flat = tvec[0].flatten()   # (3,) translation in metres

                # ── Broadcast marker frame to TF2 ─────────────────────────
                if self.tf_broadcast_marker and self.camera_frame_id:
                    t = geometry_msgs.msg.TransformStamped()
                    t.header.stamp    = msg.header.stamp
                    t.header.frame_id = self.camera_frame_id
                    t.child_frame_id  = self.marker_tf_name

                    t.transform.translation.x = float(tvec_flat[0])
                    t.transform.translation.y = float(tvec_flat[1])
                    t.transform.translation.z = float(tvec_flat[2])

                    R, _ = cv2.Rodrigues(rvec_flat)
                    R4 = np.eye(4)
                    R4[:3, :3] = R
                    q = tf_conversions.transformations.quaternion_from_matrix(R4)

                    t.transform.rotation.x = q[0]
                    t.transform.rotation.y = q[1]
                    t.transform.rotation.z = q[2]
                    t.transform.rotation.w = q[3]

                    self.tf_broadcaster.sendTransform(t)

                self.last_detection_time = rospy.Time.now()
                marker_found = True

                # ── Debug visualisation ───────────────────────────────────
                if self.debug_image_view:
                    cv2.aruco.drawDetectedMarkers(frame, corners)
                    cv2.aruco.drawAxis(
                        frame,
                        self.camera_matrix, self.dist_coeffs,
                        rvec_flat, tvec_flat,
                        self.marker_size_m * 0.5)
                    cv2.putText(
                        frame,
                        "ID:{} d={:.3f}m".format(self.marker_id, float(np.linalg.norm(tvec_flat))),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                break  # only handle the first occurrence of the target ID

        if not marker_found:
            rospy.logdebug_throttle(
                2.0, "[ArucoMarkerFollower] Marker ID=%d not detected", self.marker_id)

        # Publish debug image regardless of detection
        if self.debug_image_view and self.pub_debug_img is not None:
            try:
                self.pub_debug_img.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
            except CvBridgeError:
                pass

    # ─────────────────────────────────────────────────────────────────────────
    # Control loop (Timer callback at pub_rate Hz)
    # ─────────────────────────────────────────────────────────────────────────

    def _control_loop(self, event=None):
        # ── Guard: marker must have been seen recently ─────────────────────
        if self.last_detection_time is None:
            rospy.logwarn_throttle(
                10.0, "[ArucoMarkerFollower] No marker detected yet (ID=%d). "
                "Waiting...", self.marker_id)
            self._publish_stop()
            return

        age = (rospy.Time.now() - self.last_detection_time).to_sec()
        if age > self.marker_lost_timeout:
            rospy.logwarn_throttle(
                5.0, "[ArucoMarkerFollower] Marker ID=%d lost (%.1f s ago). "
                "Stopping.", self.marker_id, age)
            self._publish_stop()
            return

        # ── Guard: motion disabled ─────────────────────────────────────────
        if not self.enable_motion:
            return

        # ── TF lookups ────────────────────────────────────────────────────
        try:
            T_base2ee = self.tf_buffer.lookup_transform(
                self.base_frame_id, self.ee_frame_id,
                rospy.Time(0), rospy.Duration(0.05))

            T_base2marker = self.tf_buffer.lookup_transform(
                self.base_frame_id, self.marker_tf_name,
                rospy.Time(0), rospy.Duration(0.05))

        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(
                5.0, "[ArucoMarkerFollower] TF lookup failed: %s", str(e))
            self._publish_stop()
            return

        # ── Current end-effector position in base frame ────────────────────
        ee_pos = np.array([
            T_base2ee.transform.translation.x,
            T_base2ee.transform.translation.y,
            T_base2ee.transform.translation.z])

        # ── Marker pose in base frame ──────────────────────────────────────
        marker_pos = np.array([
            T_base2marker.transform.translation.x,
            T_base2marker.transform.translation.y,
            T_base2marker.transform.translation.z])

        q_marker = [
            T_base2marker.transform.rotation.x,
            T_base2marker.transform.rotation.y,
            T_base2marker.transform.rotation.z,
            T_base2marker.transform.rotation.w]
        R_marker = tf.transformations.quaternion_matrix(q_marker)[:3, :3]

        # ArUco Z axis points from marker toward camera (out of the marker face).
        # Target = marker_pos + offset * marker_Z_in_base
        # This places the EE in front of the marker on the camera side.
        marker_z_in_base = R_marker[:, 2]
        target_pos = marker_pos + self.approach_offset_m * marker_z_in_base

        # ── Position error in base frame ───────────────────────────────────
        pos_err = target_pos - ee_pos

        # Apply dead-zone per axis
        for i in range(3):
            if abs(pos_err[i]) < self.position_err_thres:
                pos_err[i] = 0.0

        # ── Proportional control → linear velocity ─────────────────────────
        vx = self.K_lin * pos_err[0]
        vy = self.K_lin * pos_err[1]
        vz = self.K_lin * pos_err[2]

        # Clamp to max linear speed (scale all axes uniformly)
        speed = np.linalg.norm([vx, vy, vz])
        if speed > self.max_lin_vel and speed > 1e-9:
            scale = self.max_lin_vel / speed
            vx *= scale
            vy *= scale
            vz *= scale

        # ── Optional orientation control ───────────────────────────────────
        wx, wy, wz = 0.0, 0.0, 0.0

        if self.K_ang > 1e-9:
            q_ee = [
                T_base2ee.transform.rotation.x,
                T_base2ee.transform.rotation.y,
                T_base2ee.transform.rotation.z,
                T_base2ee.transform.rotation.w]
            R_ee = tf.transformations.quaternion_matrix(q_ee)[:3, :3]

            # Desired orientation = marker orientation
            # (EE Z should align with marker Z pointing toward camera)
            R_err = np.dot(R_marker, R_ee.T)  # error in base frame

            trace = np.trace(R_err)
            cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
            angle = float(np.arccos(cos_angle))

            if angle > self.orientation_err_thres:
                sin_angle = np.sin(angle)
                if abs(sin_angle) > 1e-9:
                    ax = (R_err[2, 1] - R_err[1, 2]) / (2.0 * sin_angle)
                    ay = (R_err[0, 2] - R_err[2, 0]) / (2.0 * sin_angle)
                    az = (R_err[1, 0] - R_err[0, 1]) / (2.0 * sin_angle)
                    wx = self.K_ang * angle * ax
                    wy = self.K_ang * angle * ay
                    wz = self.K_ang * angle * az

                    ang_speed = np.linalg.norm([wx, wy, wz])
                    if ang_speed > self.max_ang_vel and ang_speed > 1e-9:
                        scale = self.max_ang_vel / ang_speed
                        wx *= scale
                        wy *= scale
                        wz *= scale

        # ── Publish Twist (geometry_msgs/Twist in robot base frame) ───────
        twist = geometry_msgs.msg.Twist()
        twist.linear.x  = float(vx)
        twist.linear.y  = float(vy)
        twist.linear.z  = float(vz)
        twist.angular.x = float(wx)
        twist.angular.y = float(wy)
        twist.angular.z = float(wz)
        self.pub_cmd_vel.publish(twist)

        err_norm = float(np.linalg.norm(pos_err))
        rospy.loginfo_throttle(
            2.0,
            "[ArucoMarkerFollower] pos_err=[%.3f, %.3f, %.3f] m (norm=%.3f)  "
            "vel=[%.3f, %.3f, %.3f] m/s",
            pos_err[0], pos_err[1], pos_err[2], err_norm, vx, vy, vz)

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _publish_stop(self):
        """Publish a zero-velocity Twist to halt the arm."""
        self.pub_cmd_vel.publish(geometry_msgs.msg.Twist())


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    try:
        node = ArucoMarkerFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
