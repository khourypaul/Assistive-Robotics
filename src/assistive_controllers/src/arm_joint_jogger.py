#!/usr/bin/env python3

"""
Author: Claude Code
Node: arm_joint_jogger
Description:
    Interactive terminal node for jogging Kinova arm joints incrementally.
    Uses the same ArmJointAnglesAction server as the GUI home button.

    Reads current joint angles from the Kinova driver output topic, adds the
    requested delta (in degrees), and sends the updated target via the action
    server.  Two modes are supported:

      keyboard (default):
        Interactive terminal.  Press 1-6 to select a joint, w/= to jog
        positive, s/- to jog negative, ]/. to increase step, [/, to
        decrease step, q or ESC to quit.

      single:
        Send one incremental command and exit.  Useful for scripting.
        rosrun assistive_controllers arm_joint_jogger.py \\
            _mode:=single _joint:=1 _delta:=5.0

Parameters:
    ~action_address   : ArmJointAnglesAction server address
    ~joint_state_topic: kinova_msgs/JointAngles output topic (degrees)
    ~mode             : 'keyboard' or 'single' (default: 'keyboard')
    ~joint            : joint number 1-6 (single mode only)
    ~delta            : delta in degrees (single mode only, default: 5.0)
    ~step_size        : initial step size in degrees for keyboard mode (default: 5.0)
    ~timeout          : action result wait timeout in seconds (default: 10.0)

Subscribes to:
    ~joint_state_topic  (kinova_msgs/JointAngles)

Action client:
    ~action_address  (kinova_msgs/ArmJointAnglesAction)
"""

import sys
import select
import threading

import math

import rospy
import actionlib
import kinova_msgs.msg
from sensor_msgs.msg import JointState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_import_termios():
    """Return (tty, termios) on Linux, None on failure (e.g. Windows)."""
    try:
        import tty
        import termios
        return tty, termios
    except ImportError:
        return None, None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ArmJointJogger:
    def __init__(self):
        rospy.init_node('arm_joint_jogger', anonymous=False)

        self.action_address = rospy.get_param(
            '~action_address',
            '/oarbot_blue/j2n6s300_right_driver/joints_action/joint_angles')

        self.joint_state_topic = rospy.get_param(
            '~joint_state_topic',
            '/oarbot_blue/j2n6s300_right_driver/out/joint_state')

        self.mode       = rospy.get_param('~mode',      'keyboard')
        self.step_size  = rospy.get_param('~step_size',  5.0)   # degrees
        self.act_timeout = rospy.get_param('~timeout',   10.0)  # seconds

        # Current joint angles in degrees (6-element list)
        self._angles      = None
        self._angles_lock = threading.Lock()

        # Subscribe to joint angle feedback
        self._sub = rospy.Subscriber(
            self.joint_state_topic,
            JointState,
            self._joint_angles_cb,
            queue_size=1)

        # Connect to action server
        rospy.loginfo("arm_joint_jogger: connecting to '%s'", self.action_address)
        self._client = actionlib.SimpleActionClient(
            self.action_address,
            kinova_msgs.msg.ArmJointAnglesAction)

        if not self._client.wait_for_server(rospy.Duration(5.0)):
            rospy.logerr(
                "arm_joint_jogger: action server not available at '%s'",
                self.action_address)
            sys.exit(1)

        rospy.loginfo("arm_joint_jogger: action server connected.")

        # Wait for first joint state reading
        rospy.loginfo(
            "arm_joint_jogger: waiting for joint state on '%s'...",
            self.joint_state_topic)
        deadline = rospy.Time.now() + rospy.Duration(5.0)
        rate = rospy.Rate(20)
        while self._angles is None and not rospy.is_shutdown():
            if rospy.Time.now() > deadline:
                rospy.logerr(
                    "arm_joint_jogger: timed out waiting for joint state "
                    "from '%s'", self.joint_state_topic)
                sys.exit(1)
            rate.sleep()

        rospy.loginfo("arm_joint_jogger: joint state received.")

    # ------------------------------------------------------------------
    # Subscriber callback
    # ------------------------------------------------------------------

    def _joint_angles_cb(self, msg):
        # sensor_msgs/JointState positions are in radians; convert to degrees
        # for use with ArmJointAnglesGoal (which expects degrees)
        with self._angles_lock:
            self._angles = [math.degrees(p) for p in msg.position[:6]]

    def _get_angles(self):
        with self._angles_lock:
            return list(self._angles) if self._angles is not None else None

    # ------------------------------------------------------------------
    # Core jog command
    # ------------------------------------------------------------------

    def send_jog(self, joint_idx, delta_deg, wait=False):
        """
        Jog joint (0-based index) by delta_deg degrees.

        Reads current angles, adds delta to the selected joint, then sends
        an ArmJointAnglesGoal.  Returns True on success.
        """
        angles = self._get_angles()
        if angles is None:
            rospy.logerr("arm_joint_jogger: no joint state available")
            return False

        angles[joint_idx] += delta_deg

        goal = kinova_msgs.msg.ArmJointAnglesGoal()
        goal.angles.joint1 = angles[0]
        goal.angles.joint2 = angles[1]
        goal.angles.joint3 = angles[2]
        goal.angles.joint4 = angles[3]
        goal.angles.joint5 = angles[4]
        goal.angles.joint6 = angles[5]
        goal.angles.joint7 = 0.0  # 7th DOF unused on J2N6S300

        rospy.loginfo(
            "arm_joint_jogger: jog joint %d by %+.2f° -> "
            "[%.1f, %.1f, %.1f, %.1f, %.1f, %.1f]",
            joint_idx + 1, delta_deg,
            angles[0], angles[1], angles[2],
            angles[3], angles[4], angles[5])

        self._client.send_goal(goal)

        if wait:
            finished = self._client.wait_for_result(
                rospy.Duration(self.act_timeout))
            if not finished:
                rospy.logwarn("arm_joint_jogger: action timed out")
            return finished

        return True

    # ------------------------------------------------------------------
    # Single-command mode
    # ------------------------------------------------------------------

    def _run_single(self):
        joint = int(rospy.get_param('~joint', 1))
        delta = float(rospy.get_param('~delta', 5.0))

        if not 1 <= joint <= 6:
            rospy.logerr(
                "arm_joint_jogger: invalid joint %d, must be 1-6", joint)
            sys.exit(1)

        rospy.loginfo(
            "arm_joint_jogger: single-command mode  joint=%d  delta=%.2f°",
            joint, delta)

        success = self.send_jog(joint - 1, delta, wait=True)
        if success:
            rospy.loginfo("arm_joint_jogger: move complete.")
        else:
            rospy.logwarn("arm_joint_jogger: move did not complete in time.")

    # ------------------------------------------------------------------
    # Keyboard mode
    # ------------------------------------------------------------------

    def _run_keyboard(self):
        tty_mod, termios_mod = _try_import_termios()
        if tty_mod is None:
            rospy.logerr(
                "arm_joint_jogger: termios not available – "
                "keyboard mode requires Linux.")
            sys.exit(1)

        selected_joint = 0  # 0-based
        step = self.step_size

        HELP = (
            "\n"
            "╔══════════════════════════════════════════════════════╗\n"
            "║             ARM JOINT JOGGER  (keyboard mode)        ║\n"
            "╠══════════════════════════════════════════════════════╣\n"
            "║  Select joint   :  1  2  3  4  5  6                 ║\n"
            "║  Jog +          :  w  or  =                         ║\n"
            "║  Jog -          :  s  or  -                         ║\n"
            "║  Increase step  :  ]  or  .                         ║\n"
            "║  Decrease step  :  [  or  ,                         ║\n"
            "║  Quit           :  q  or  ESC                       ║\n"
            "╚══════════════════════════════════════════════════════╝\n"
        )
        print(HELP, flush=True)

        def status_line():
            angles = self._get_angles()
            a_str = ("N/A" if angles is None
                     else "  ".join(f"J{i+1}:{v:7.2f}°" for i, v in enumerate(angles)))
            return (f"  Active joint: J{selected_joint+1} | "
                    f"Step: {step:.1f}°  |  {a_str}")

        # Print initial status
        print(status_line(), flush=True)

        fd = sys.stdin.fileno()
        old_settings = termios_mod.tcgetattr(fd)

        try:
            tty_mod.setraw(fd)

            while not rospy.is_shutdown():
                # Non-blocking key poll (50 ms timeout)
                rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not rlist:
                    continue

                key = sys.stdin.read(1)

                if key in ('q', '\x1b', '\x03'):   # q, ESC, Ctrl-C
                    break

                elif key in ('1', '2', '3', '4', '5', '6'):
                    selected_joint = int(key) - 1

                elif key in ('w', '=', '+'):
                    self.send_jog(selected_joint, step)

                elif key in ('s', '-'):
                    self.send_jog(selected_joint, -step)

                elif key in (']', '.'):
                    step = min(step + 1.0, 45.0)

                elif key in ('[', ','):
                    step = max(step - 1.0, 0.5)

                # Overwrite the status line in place
                print(f"\r\033[K{status_line()}", end='', flush=True)

        finally:
            termios_mod.tcsetattr(fd, termios_mod.TCSADRAIN, old_settings)
            print("\narm_joint_jogger: exiting.", flush=True)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self):
        if self.mode == 'single':
            self._run_single()
        else:
            self._run_keyboard()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    try:
        jogger = ArmJointJogger()
        jogger.run()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass
