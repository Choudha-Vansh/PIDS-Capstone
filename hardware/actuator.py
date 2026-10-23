"""
Actuator interface — the seam that turns a drive decision into real motion. Mirrors
the sensor abstraction: the control loop only knows about ActuatorInterface, so the
same loop runs safely on the laptop (NoOp) or drives the Duckiebot (ROS) unchanged.

SAFETY: the loop should start with the NoOp actuator (config.duckiebot.safety.dry_run:
true) and only switch to the Duckiebot actuator once the brake/accelerate decisions
look correct in the logs/overlay.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class ActuatorInterface(ABC):
    @abstractmethod
    def drive(self, v: float, omega: float) -> None:
        """Command linear speed v (m/s) and angular speed omega (rad/s)."""

    def brake(self) -> None:
        """Command a full stop."""
        self.drive(0.0, 0.0)

    def stop(self) -> None:
        """Alias for brake; also used on shutdown / watchdog."""
        self.brake()

    def close(self) -> None:
        self.stop()


class NoOpActuator(ActuatorInterface):
    """Safe dry-run actuator: logs intended commands, moves nothing."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.last = (0.0, 0.0)

    def drive(self, v: float, omega: float) -> None:
        self.last = (v, omega)
        if self.verbose:
            print(f"[DRY-RUN] drive v={v:.3f} m/s  omega={omega:.3f} rad/s")


class DuckiebotROSActuator(ActuatorInterface):
    """
    Publishes a Twist2DStamped on the Duckiebot's car-command topic over ROS.
    Lazy ROS imports so the module loads without a ROS install.
    """

    def __init__(self, cfg: dict):
        import rospy                                   # lazy
        from duckietown_msgs.msg import Twist2DStamped
        self._rospy = rospy
        self._Twist = Twist2DStamped

        d = cfg["duckiebot"]
        veh = d["vehicle_name"]
        topic = d["command"]["car_cmd_topic"].replace("{veh}", veh)
        if not rospy.core.is_initialized():
            rospy.init_node("adas_laptop_actuator", anonymous=True,
                            disable_signals=True)
        self._pub = rospy.Publisher(topic, Twist2DStamped, queue_size=1)
        self._topic = topic

    def drive(self, v: float, omega: float) -> None:
        msg = self._Twist()
        msg.header.stamp = self._rospy.Time.now()
        msg.v = float(v)
        msg.omega = float(omega)
        self._pub.publish(msg)


def build_actuator(cfg: dict) -> ActuatorInterface:
    """Factory: NoOp when safety.dry_run is true (default), else the Duckiebot actuator."""
    dry = cfg.get("duckiebot", {}).get("safety", {}).get("dry_run", True)
    if dry:
        return NoOpActuator()
    return DuckiebotROSActuator(cfg)
