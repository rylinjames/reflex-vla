"""ROS2 bridge for reflex serve.

Runs a ROS2 node that subscribes to image + state + task topics, runs
inference via ReflexServer, and publishes action chunks to a topic at a
configurable rate.

rclpy is NOT pip-installable. Install ROS2 (humble/iron/jazzy) via apt or
robostack before running:

    source /opt/ros/humble/setup.bash
    reflex ros2-serve <export_dir>

Default topic layout (override via CLI flags):
    subs:
      /camera/image_raw      sensor_msgs/msg/Image (rgb8)
      /joint_states          sensor_msgs/msg/JointState (positions → state vector)
      /reflex/task           std_msgs/msg/String (text instruction)
    pub:
      /reflex/actions        std_msgs/msg/Float32MultiArray (flat chunk × action_dim)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _require_rclpy():
    """Import rclpy + message modules or raise a helpful ImportError."""
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image, JointState
        from std_msgs.msg import Float32MultiArray, String
        return rclpy, Node, Image, JointState, String, Float32MultiArray
    except ImportError as exc:
        raise ImportError(
            "rclpy not available. The reflex ROS2 bridge requires a ROS2 install "
            "(humble, iron, or jazzy) via apt or robostack — rclpy is NOT "
            "pip-installable. Run:\n"
            "    source /opt/ros/humble/setup.bash  # or iron / jazzy\n"
            "    reflex ros2-serve <export_dir>\n"
            f"Underlying error: {exc}"
        ) from exc


def create_ros2_bridge_node(
    server: Any,
    *,
    image_topic: str = "/camera/image_raw",
    state_topic: str = "/joint_states",
    task_topic: str = "/reflex/task",
    action_topic: str = "/reflex/actions",
    rate_hz: float = 20.0,
    node_name: str = "reflex_vla",
) -> Any:
    """Build a ROS2 node that wraps ``server.predict()`` as pub/sub.

    The returned node subscribes to image + state + task topics, caches the
    latest message from each, and at ``rate_hz`` Hz invokes
    ``server.predict(image, instruction, state)`` and publishes the action
    chunk to ``action_topic`` as a flat Float32MultiArray.
    """
    rclpy, Node, Image, JointState, String, Float32MultiArray = _require_rclpy()

    class ReflexROS2Node(Node):
        def __init__(self) -> None:
            super().__init__(node_name)
            self._server = server
            self._last_image: np.ndarray | None = None
            self._last_state: list[float] | None = None
            self._last_task: str = ""
            self._inference_count = 0

            self.create_subscription(Image, image_topic, self._image_cb, 10)
            self.create_subscription(JointState, state_topic, self._state_cb, 10)
            self.create_subscription(String, task_topic, self._task_cb, 10)
            self._action_pub = self.create_publisher(Float32MultiArray, action_topic, 10)
            self._timer = self.create_timer(1.0 / max(0.1, rate_hz), self._tick)

            self.get_logger().info(
                f"reflex ros2 node '{node_name}' up: subs={image_topic} + "
                f"{state_topic} + {task_topic}, pub={action_topic} at "
                f"{rate_hz:.1f} Hz"
            )

        def _image_cb(self, msg: Any) -> None:
            """Decode sensor_msgs/Image → HxWx3 uint8 numpy array.

            Handles rgb8 and bgr8 encodings. Other encodings fall back to the
            raw reshape and may need conversion by the caller.
            """
            h, w = int(msg.height), int(msg.width)
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            encoding = getattr(msg, "encoding", "rgb8")
            if arr.size == h * w * 3:
                img = arr.reshape(h, w, 3)
                if encoding == "bgr8":
                    img = img[..., ::-1]
                self._last_image = img.copy()
            elif arr.size == h * w * 4:
                # rgba8 / bgra8 — drop alpha
                img = arr.reshape(h, w, 4)[..., :3]
                if encoding in ("bgra8", "bgr8"):
                    img = img[..., ::-1]
                self._last_image = img.copy()
            else:
                self.get_logger().warning(
                    f"unsupported image size/encoding: {arr.size} bytes, "
                    f"{h}x{w}, encoding={encoding}"
                )

        def _state_cb(self, msg: Any) -> None:
            self._last_state = [float(x) for x in msg.position]

        def _task_cb(self, msg: Any) -> None:
            self._last_task = str(msg.data)

        def _tick(self) -> None:
            if self._last_image is None or self._last_state is None:
                return
            try:
                result = self._server.predict(
                    image=self._last_image,
                    instruction=self._last_task,
                    state=self._last_state,
                )
            except Exception as exc:
                self.get_logger().error(f"predict failed: {exc}")
                return
            if isinstance(result, dict) and "error" in result:
                self.get_logger().warning(f"predict error: {result['error']}")
                return
            actions = result.get("actions") if isinstance(result, dict) else None
            if not actions:
                return
            out = Float32MultiArray()
            out.data = [float(v) for chunk in actions for v in chunk]
            self._action_pub.publish(out)
            self._inference_count += 1

    return ReflexROS2Node()


def run_ros2_bridge(
    export_dir: str | Path,
    *,
    device: str = "cuda",
    providers: list[str] | None = None,
    strict_providers: bool = True,
    safety_config: str | Path | None = None,
    image_topic: str = "/camera/image_raw",
    state_topic: str = "/joint_states",
    task_topic: str = "/reflex/task",
    action_topic: str = "/reflex/actions",
    rate_hz: float = 20.0,
    node_name: str = "reflex_vla",
) -> None:
    """Load the model, init rclpy, spin the bridge node until shutdown."""
    rclpy, _, _, _, _, _ = _require_rclpy()
    from reflex.runtime.server import ReflexServer

    server = ReflexServer(
        export_dir,
        device=device,
        providers=providers,
        strict_providers=strict_providers,
        safety_config=safety_config,
    )
    server.load()

    rclpy.init()
    node = None
    try:
        node = create_ros2_bridge_node(
            server,
            image_topic=image_topic,
            state_topic=state_topic,
            task_topic=task_topic,
            action_topic=action_topic,
            rate_hz=rate_hz,
            node_name=node_name,
        )
        rclpy.spin(node)
    except KeyboardInterrupt:
        logger.info("ros2 bridge interrupted by user")
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


__all__ = ["create_ros2_bridge_node", "run_ros2_bridge"]
