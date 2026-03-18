#!/usr/bin/env python3

"""
Author: (Assistive Robotics project)
Node: arm_pose_sequencer
Description:
    Cycles a Kinova arm through a list of pre-defined joint configurations.
    Press SPACE / ENTER / N to go to the next config.
    Press P / B to go to the previous config.
    Press 0-9 to jump directly to a config by index.
    Press Q or ESC to quit.

    Joint angles in the YAML are in DEGREES (same convention as the Kinova
    driver's ArmJointAnglesGoal).

Parameters:
    ~action_address    : ArmJointAnglesAction server  (same as arm_joint_jogger)
    ~joint_state_topic : JointState feedback topic    (same as arm_joint_jogger)
    ~timeout           : seconds to wait for each move to complete (default 15.0)
    ~configurations    : list of {name, joints} dicts loaded from YAML

Subscribes to:
    joint_state_topic  (sensor_msgs/JointState)

Action client:
    action_address     (kinova_msgs/ArmJointAnglesAction)
"""

import sys
import select
import threading
import math

import rospy
import actionlib
import kinova_msgs.msg
from sensor_msgs.msg import JointState


class ArmPoseSequencer:
    def __init__(self):
        rospy.init_node('arm_pose_sequencer', anonymous=False)

        self.action_address = rospy.get_param(
            '~action_address',
            '/oarbot_blue/j2n6s300_right_driver/joints_action/joint_angles')
        self.joint_state_topic = rospy.get_param(
            '~joint_state_topic',
            '/oarbot_blue/j2n6s300_right_driver/out/joint_state')
        self.act_timeout = float(rospy.get_param('~timeout', 15.0))

        # Load configurations from parameter server
        raw = rospy.get_param('~configurations', [])
        if not raw:
            rospy.logfatal(
                "arm_pose_sequencer: no '~configurations' found in params. "
                "Load a YAML file with the configurations list.")
            sys.exit(1)

        self.configs = []
        for i, entry in enumerate(raw):
            if isinstance(entry, dict):
                name   = entry.get('name', 'Config {}'.format(i))
                joints = entry.get('joints', [])
            else:
                # plain list of angles
                name   = 'Config {}'.format(i)
                joints = list(entry)

            if len(joints) < 6:
                rospy.logfatal(
                    "arm_pose_sequencer: config '%s' has %d joints, need 6.",
                    name, len(joints))
                sys.exit(1)

            self.configs.append({'name': name, 'joints': [float(j) for j in joints[:6]]})

        rospy.loginfo("arm_pose_sequencer: loaded %d configurations.", len(self.configs))

        self.current_idx = -1   # -1 = haven't gone to any config yet

        # Current joint angles (degrees, from driver)
        self._angles      = None
        self._angles_lock = threading.Lock()

        # Subscribe to joint state
        rospy.Subscriber(
            self.joint_state_topic, JointState,
            self._joint_state_cb, queue_size=1)

        # Connect to action server
        rospy.loginfo("arm_pose_sequencer: connecting to '%s'...", self.action_address)
        self._client = actionlib.SimpleActionClient(
            self.action_address, kinova_msgs.msg.ArmJointAnglesAction)

        if not self._client.wait_for_server(rospy.Duration(5.0)):
            rospy.logerr(
                "arm_pose_sequencer: action server not available at '%s'",
                self.action_address)
            sys.exit(1)
        rospy.loginfo("arm_pose_sequencer: connected.")

        # Wait for first joint state
        deadline = rospy.Time.now() + rospy.Duration(5.0)
        rate = rospy.Rate(20)
        while self._angles is None and not rospy.is_shutdown():
            if rospy.Time.now() > deadline:
                rospy.logerr(
                    "arm_pose_sequencer: timed out waiting for joint state.")
                sys.exit(1)
            rate.sleep()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _joint_state_cb(self, msg):
        with self._angles_lock:
            self._angles = [math.degrees(p) for p in msg.position[:6]]

    def _get_angles(self):
        with self._angles_lock:
            return list(self._angles) if self._angles is not None else None

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    def _send_config(self, idx):
        cfg = self.configs[idx]
        j   = cfg['joints']

        goal = kinova_msgs.msg.ArmJointAnglesGoal()
        goal.angles.joint1 = j[0]
        goal.angles.joint2 = j[1]
        goal.angles.joint3 = j[2]
        goal.angles.joint4 = j[3]
        goal.angles.joint5 = j[4]
        goal.angles.joint6 = j[5]
        goal.angles.joint7 = 0.0

        print('\r\033[K  >> Sending: {} -> [{:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}]'.format(
            cfg['name'], j[0], j[1], j[2], j[3], j[4], j[5]), flush=True)

        self._client.send_goal(goal)
        finished = self._client.wait_for_result(rospy.Duration(self.act_timeout))

        if finished:
            print('  >> Done: {}'.format(cfg['name']), flush=True)
        else:
            print('  >> WARNING: move timed out ({:.0f}s)'.format(self.act_timeout),
                  flush=True)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _print_menu(self):
        n = len(self.configs)
        lines = [
            '',
            '╔══════════════════════════════════════════════════════════╗',
            '║              ARM POSE SEQUENCER                         ║',
            '╠══════════════════════════════════════════════════════════╣',
            '║  SPACE / ENTER / N  →  next config                      ║',
            '║  P / B              →  previous config                  ║',
            '║  0-9                →  jump to config by index          ║',
            '║  Q / ESC            →  quit                             ║',
            '╠══════════════════════════════════════════════════════════╣',
        ]

        for i, cfg in enumerate(self.configs):
            marker = '►' if i == self.current_idx else ' '
            j = cfg['joints']
            lines.append('║ {} {:2d}  {:20s}  [{:6.1f},{:6.1f},{:6.1f},{:6.1f},{:6.1f},{:6.1f}] ║'.format(
                marker, i, cfg['name'][:20], j[0], j[1], j[2], j[3], j[4], j[5]))

        lines.append('╠══════════════════════════════════════════════════════════╣')

        cur = self._get_angles()
        if cur:
            lines.append('║ Current  [{:6.1f},{:6.1f},{:6.1f},{:6.1f},{:6.1f},{:6.1f}]         ║'.format(
                cur[0], cur[1], cur[2], cur[3], cur[4], cur[5]))
        else:
            lines.append('║ Current  (unknown)                                       ║')

        lines.append('╚══════════════════════════════════════════════════════════╝')
        lines.append('  Press a key...')

        print('\n'.join(lines), flush=True)

    # ------------------------------------------------------------------
    # Keyboard loop
    # ------------------------------------------------------------------

    def run(self):
        try:
            import tty
            import termios
        except ImportError:
            rospy.logerr("arm_pose_sequencer: termios not available (Linux only).")
            sys.exit(1)

        n   = len(self.configs)
        fd  = sys.stdin.fileno()
        old = termios.tcgetattr(fd)

        self._print_menu()

        try:
            tty.setraw(fd)

            while not rospy.is_shutdown():
                rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not rlist:
                    continue

                key = sys.stdin.read(1)

                if key in ('q', '\x1b', '\x03'):          # Q, ESC, Ctrl-C
                    break

                elif key in (' ', '\r', '\n', 'n', 'N'):  # next
                    self.current_idx = (self.current_idx + 1) % n

                elif key in ('p', 'P', 'b', 'B'):         # previous
                    self.current_idx = (self.current_idx - 1) % n

                elif key.isdigit():                        # jump to index
                    idx = int(key)
                    if idx < n:
                        self.current_idx = idx
                    else:
                        print('\r\033[K  No config index {}'.format(idx), flush=True)
                        continue

                else:
                    continue

                # Restore terminal to print cleanly, send move, then go raw again
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
                self._send_config(self.current_idx)
                self._print_menu()
                tty.setraw(fd)

        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
            print('\narm_pose_sequencer: exiting.', flush=True)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    try:
        node = ArmPoseSequencer()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass
