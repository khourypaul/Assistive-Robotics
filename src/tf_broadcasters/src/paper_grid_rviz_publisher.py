#!/usr/bin/env python

from __future__ import print_function

import json
import os
import threading

import cv2
import numpy as np
import rospy
import tf2_ros
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


class PaperGridRvizPublisher(object):
    def __init__(self):
        rospy.init_node("paper_grid_rviz_publisher", anonymous=False)

        self.layout_json_path = os.path.expanduser(rospy.get_param("~layout_json_path", ""))
        if not self.layout_json_path:
            rospy.logfatal("~layout_json_path must be set.")
            raise RuntimeError("Missing layout_json_path")

        self.image_topic_name = rospy.get_param("~image_topic_name", "/nuc/rgb/image_raw")
        self.camera_info_topic_name = rospy.get_param("~camera_info_topic_name", "/nuc/rgb/camera_info")
        self.camera_frame_id = rospy.get_param("~camera_frame_id", "")
        self.target_frame_id = rospy.get_param("~target_frame_id", "")

        self.marker_topic_name = rospy.get_param("~marker_topic_name", "paper_grid_markers")
        self.pose_topic_name = rospy.get_param("~pose_topic_name", "paper_grid_pose")
        self.publish_debug_image = rospy.get_param("~publish_debug_image", True)
        self.debug_image_topic_name = rospy.get_param("~debug_image_topic_name", "paper_grid_debug_image")

        self.min_markers_for_pose = int(rospy.get_param("~min_markers_for_pose", 2))
        self.ransac_reprojection_error = float(rospy.get_param("~ransac_reprojection_error", 3.0))
        self.tf_lookup_timeout_sec = float(rospy.get_param("~tf_lookup_timeout_sec", 0.05))

        self.grid_line_width = float(rospy.get_param("~grid_line_width", 0.003))
        self.outline_line_width = float(rospy.get_param("~outline_line_width", 0.006))
        self.surface_alpha = float(rospy.get_param("~surface_alpha", 0.12))
        self.text_height = float(rospy.get_param("~text_height", 0.03))
        self.text_z_offset = float(rospy.get_param("~text_z_offset", 0.01))

        self.layout_data = self._load_layout(self.layout_json_path)
        self.layout_marker_corners = self._build_layout_marker_corners(self.layout_data)
        self.layout_marker_centers = self._build_layout_marker_centers(self.layout_marker_corners)
        self.layout_marker_ids = sorted(self.layout_marker_corners.keys())
        self.board_outline_local = self._build_board_outline(self.layout_marker_corners)

        layout_dict_name = self.layout_data.get("grid", {}).get("dict", "DICT_4X4_50")
        self.aruco_dict_name = rospy.get_param("~aruco_dict_name", layout_dict_name)
        self.aruco_dict = self._make_aruco_dict(self.aruco_dict_name)
        self.aruco_params = self._make_aruco_params()
        self.aruco_detector = None
        if hasattr(cv2.aruco, "ArucoDetector"):
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        self.bridge = CvBridge()
        self.marker_pub = rospy.Publisher(self.marker_topic_name, MarkerArray, queue_size=1)
        self.pose_pub = rospy.Publisher(self.pose_topic_name, PoseStamped, queue_size=1)
        self.debug_image_pub = None
        if self.publish_debug_image:
            self.debug_image_pub = rospy.Publisher(self.debug_image_topic_name, Image, queue_size=1)

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.K = None
        self.D = None
        self.camera_info_ready = False
        self.markers_visible = False
        self.data_lock = threading.Lock()

        self.camera_info_sub = rospy.Subscriber(
            self.camera_info_topic_name, CameraInfo, self._camera_info_cb, queue_size=1
        )
        self.image_sub = rospy.Subscriber(
            self.image_topic_name, Image, self._image_cb, queue_size=1
        )

        rospy.loginfo("paper_grid_rviz_publisher initialized.")
        rospy.loginfo("Layout markers loaded: %d", len(self.layout_marker_ids))
        rospy.loginfo("Aruco dictionary: %s", self.aruco_dict_name)

    def _load_layout(self, path):
        if not os.path.exists(path):
            raise IOError("Layout json not found: {}".format(path))
        with open(path, "r") as f:
            return json.load(f)

    def _build_layout_marker_corners(self, layout_data):
        out = {}
        for marker in layout_data.get("markers", []):
            marker_id = int(marker["id"])
            corners = np.asarray(marker["corners_m"], dtype=np.float64)
            if corners.shape != (4, 3):
                raise ValueError("Marker {} corners_m must be 4x3".format(marker_id))
            out[marker_id] = corners
        if not out:
            raise ValueError("Layout has no markers")
        return out

    def _build_layout_marker_centers(self, marker_corners):
        centers = {}
        for marker_id, corners in marker_corners.items():
            centers[marker_id] = np.mean(corners, axis=0)
        return centers

    def _build_board_outline(self, marker_corners):
        all_points = np.vstack(list(marker_corners.values()))
        min_x = float(np.min(all_points[:, 0]))
        max_x = float(np.max(all_points[:, 0]))
        min_y = float(np.min(all_points[:, 1]))
        max_y = float(np.max(all_points[:, 1]))
        z = 0.0
        return np.asarray(
            [
                [min_x, min_y, z],
                [max_x, min_y, z],
                [max_x, max_y, z],
                [min_x, max_y, z],
            ],
            dtype=np.float64,
        )

    def _make_aruco_dict(self, name):
        if name not in ARUCO_DICT:
            raise ValueError("Unsupported aruco dictionary: {}".format(name))
        dict_id = ARUCO_DICT[name]
        if hasattr(cv2.aruco, "getPredefinedDictionary"):
            return cv2.aruco.getPredefinedDictionary(dict_id)
        return cv2.aruco.Dictionary_get(dict_id)

    def _make_aruco_params(self):
        if hasattr(cv2.aruco, "DetectorParameters_create"):
            return cv2.aruco.DetectorParameters_create()
        return cv2.aruco.DetectorParameters()

    def _camera_info_cb(self, msg):
        with self.data_lock:
            self.K = np.asarray(msg.K, dtype=np.float64).reshape(3, 3)
            self.D = np.asarray(msg.D, dtype=np.float64)
            self.camera_info_ready = True
            if not self.camera_frame_id:
                self.camera_frame_id = msg.header.frame_id

    def _detect_markers(self, gray):
        if self.aruco_detector is not None:
            return self.aruco_detector.detectMarkers(gray)
        return cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

    def _image_cb(self, msg):
        with self.data_lock:
            if not self.camera_info_ready:
                return
            K = self.K.copy()
            D = self.D.copy()
            camera_frame_id = self.camera_frame_id

        if not camera_frame_id:
            rospy.logwarn_throttle(5.0, "Camera frame is empty; waiting for CameraInfo.")
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logwarn_throttle(5.0, "cv_bridge conversion failed: %s", str(exc))
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detect_markers(gray)

        if self.publish_debug_image and self.debug_image_pub is not None:
            debug_img = frame.copy()
            if ids is not None and len(ids) > 0 and hasattr(cv2.aruco, "drawDetectedMarkers"):
                cv2.aruco.drawDetectedMarkers(debug_img, corners, ids)
            self._publish_debug_image(debug_img, msg.header.stamp)

        if ids is None or len(ids) == 0:
            self._clear_markers_if_needed(msg.header.stamp, camera_frame_id)
            return

        ids = ids.flatten().tolist()
        object_points = []
        image_points = []
        matched_marker_ids = []

        for i, marker_id in enumerate(ids):
            if marker_id not in self.layout_marker_corners:
                continue
            marker_corners_local = self.layout_marker_corners[marker_id]
            marker_corners_image = np.asarray(corners[i], dtype=np.float64).reshape(4, 2)
            object_points.extend(marker_corners_local.tolist())
            image_points.extend(marker_corners_image.tolist())
            matched_marker_ids.append(marker_id)

        unique_count = len(set(matched_marker_ids))
        if unique_count < self.min_markers_for_pose:
            self._clear_markers_if_needed(msg.header.stamp, camera_frame_id)
            rospy.logwarn_throttle(
                2.0,
                "Detected %d layout markers, need at least %d.",
                unique_count,
                self.min_markers_for_pose,
            )
            return

        object_points = np.asarray(object_points, dtype=np.float64)
        image_points = np.asarray(image_points, dtype=np.float64)

        if object_points.shape[0] < 4:
            self._clear_markers_if_needed(msg.header.stamp, camera_frame_id)
            return

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            K,
            D,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=self.ransac_reprojection_error,
            confidence=0.99,
            iterationsCount=200,
        )
        if not success:
            self._clear_markers_if_needed(msg.header.stamp, camera_frame_id)
            rospy.logwarn_throttle(2.0, "solvePnPRansac failed for paper board.")
            return

        if inliers is not None and len(inliers) >= 4:
            inlier_idx = inliers.flatten()
            inlier_obj = object_points[inlier_idx]
            inlier_img = image_points[inlier_idx]
            success_refine, rvec_refine, tvec_refine = cv2.solvePnP(
                inlier_obj,
                inlier_img,
                K,
                D,
                rvec,
                tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if success_refine:
                rvec = rvec_refine
                tvec = tvec_refine

        R_co, _ = cv2.Rodrigues(rvec)
        t_co = np.asarray(tvec, dtype=np.float64).reshape(3, 1)

        stamp = msg.header.stamp if msg.header.stamp != rospy.Time(0) else rospy.Time.now()
        target_frame = self.target_frame_id if self.target_frame_id else camera_frame_id
        R_tc, t_tc = self._camera_to_target_transform(camera_frame_id, target_frame, stamp)
        if R_tc is None:
            return

        R_to = np.matmul(R_tc, R_co)
        t_to = np.matmul(R_tc, t_co) + t_tc

        self._publish_pose(stamp, target_frame, R_to, t_to)
        self._publish_markers(stamp, target_frame, R_to, t_to)
        self.markers_visible = True

    def _camera_to_target_transform(self, camera_frame_id, target_frame_id, stamp):
        if target_frame_id == camera_frame_id:
            return np.eye(3), np.zeros((3, 1))
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                target_frame_id,
                camera_frame_id,
                stamp,
                rospy.Duration(self.tf_lookup_timeout_sec),
            )
        except Exception as exc:
            rospy.logwarn_throttle(
                2.0,
                "TF lookup failed: %s -> %s (%s)",
                target_frame_id,
                camera_frame_id,
                str(exc),
            )
            return None, None

        q = tf_msg.transform.rotation
        t = tf_msg.transform.translation
        R = self._quat_to_rot(q.x, q.y, q.z, q.w)
        T = np.asarray([[t.x], [t.y], [t.z]], dtype=np.float64)
        return R, T

    @staticmethod
    def _quat_to_rot(x, y, z, w):
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z
        return np.asarray(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _rot_to_quat(R):
        m = R
        trace = float(m[0, 0] + m[1, 1] + m[2, 2])
        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (m[2, 1] - m[1, 2]) / s
            qy = (m[0, 2] - m[2, 0]) / s
            qz = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s
        return qx, qy, qz, qw

    def _transform_points(self, points_local, R_to, t_to):
        points_local = np.asarray(points_local, dtype=np.float64)
        points_world = np.matmul(R_to, points_local.T) + t_to
        return points_world.T

    def _publish_pose(self, stamp, frame_id, R_to, t_to):
        qx, qy, qz, qw = self._rot_to_quat(R_to)
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.pose.position.x = float(t_to[0, 0])
        msg.pose.position.y = float(t_to[1, 0])
        msg.pose.position.z = float(t_to[2, 0])
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        self.pose_pub.publish(msg)

    @staticmethod
    def _to_point(x, y, z):
        p = Point()
        p.x = float(x)
        p.y = float(y)
        p.z = float(z)
        return p

    def _publish_markers(self, stamp, frame_id, R_to, t_to):
        msg = MarkerArray()

        outline_world = self._transform_points(self.board_outline_local, R_to, t_to)

        surface = Marker()
        surface.header.stamp = stamp
        surface.header.frame_id = frame_id
        surface.ns = "paper_surface"
        surface.id = 0
        surface.type = Marker.TRIANGLE_LIST
        surface.action = Marker.ADD
        surface.pose.orientation.w = 1.0
        surface.scale.x = 1.0
        surface.scale.y = 1.0
        surface.scale.z = 1.0
        surface.color.r = 0.1
        surface.color.g = 0.5
        surface.color.b = 1.0
        surface.color.a = self.surface_alpha
        surface.points.append(self._to_point(*outline_world[0]))
        surface.points.append(self._to_point(*outline_world[1]))
        surface.points.append(self._to_point(*outline_world[2]))
        surface.points.append(self._to_point(*outline_world[0]))
        surface.points.append(self._to_point(*outline_world[2]))
        surface.points.append(self._to_point(*outline_world[3]))
        msg.markers.append(surface)

        outline = Marker()
        outline.header.stamp = stamp
        outline.header.frame_id = frame_id
        outline.ns = "paper_outline"
        outline.id = 1
        outline.type = Marker.LINE_STRIP
        outline.action = Marker.ADD
        outline.pose.orientation.w = 1.0
        outline.scale.x = self.outline_line_width
        outline.color.r = 1.0
        outline.color.g = 0.2
        outline.color.b = 0.2
        outline.color.a = 1.0
        for pt in outline_world:
            outline.points.append(self._to_point(*pt))
        outline.points.append(self._to_point(*outline_world[0]))
        msg.markers.append(outline)

        grid = Marker()
        grid.header.stamp = stamp
        grid.header.frame_id = frame_id
        grid.ns = "aruco_grid"
        grid.id = 2
        grid.type = Marker.LINE_LIST
        grid.action = Marker.ADD
        grid.pose.orientation.w = 1.0
        grid.scale.x = self.grid_line_width
        grid.color.r = 0.0
        grid.color.g = 1.0
        grid.color.b = 0.2
        grid.color.a = 0.95

        for marker_id in self.layout_marker_ids:
            corners_world = self._transform_points(self.layout_marker_corners[marker_id], R_to, t_to)
            edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
            for a, b in edges:
                grid.points.append(self._to_point(*corners_world[a]))
                grid.points.append(self._to_point(*corners_world[b]))
        msg.markers.append(grid)

        for i, marker_id in enumerate(self.layout_marker_ids):
            center_world = self._transform_points(
                np.asarray([self.layout_marker_centers[marker_id]]), R_to, t_to
            )[0]
            text = Marker()
            text.header.stamp = stamp
            text.header.frame_id = frame_id
            text.ns = "aruco_ids"
            text.id = 100 + i
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(center_world[0])
            text.pose.position.y = float(center_world[1])
            text.pose.position.z = float(center_world[2] + self.text_z_offset)
            text.pose.orientation.w = 1.0
            text.scale.z = self.text_height
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            text.text = str(marker_id)
            msg.markers.append(text)

        self.marker_pub.publish(msg)

    def _clear_markers_if_needed(self, stamp, frame_id):
        if not self.markers_visible:
            return
        clear = MarkerArray()
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = frame_id
        marker.action = Marker.DELETEALL
        clear.markers.append(marker)
        self.marker_pub.publish(clear)
        self.markers_visible = False

    def _publish_debug_image(self, cv_img, stamp):
        try:
            msg = self.bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
            msg.header.stamp = stamp
            self.debug_image_pub.publish(msg)
        except CvBridgeError:
            pass


if __name__ == "__main__":
    try:
        PaperGridRvizPublisher()
        rospy.spin()
    except Exception as exc:
        rospy.logfatal("paper_grid_rviz_publisher failed: %s", str(exc))
