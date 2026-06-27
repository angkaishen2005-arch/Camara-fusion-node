#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import message_filters

# TF2 imports
from tf2_ros import Buffer, TransformListener, TransformException

import numpy as np

class CameraFusionNode(Node):
    def __init__(self):
        super().__init__('camera_fusion_node')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Cache for transformation matrices
        self.cached_transforms = {}

        self.pc_pub = self.create_publisher(PointCloud2, '/point_cloud', 10)

        self.sub_l = message_filters.Subscriber(self, PointCloud2, '/lidar_l/points')
        self.sub_r = message_filters.Subscriber(self, PointCloud2, '/lidar_r/points')
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_l, self.sub_r], 
            queue_size=10, 
            slop=0.1
        )
        self.ts.registerCallback(self.sync_callback)

        # --- Hysteresis / Persistence State Variables ---
        self.left_bad_frames = 0
        self.right_bad_frames = 0
        self.PERSISTENCE_LIMIT = 5  # Must see 5 consecutive frames to toggle state
        self.OBSTACLE_HEIGHT = 0.45
        self.MIN_OBSTACLE_POINTS = 1000

        self.get_logger().info("Python Camera Fusion Node (with Hysteresis) Started.")

    def extract_xyz(self, cloud_msg):
        """Extracts X, Y, Z instantly using native memory buffers."""
        if len(cloud_msg.data) == 0:
            return np.empty((0, 3), dtype=np.float32)

        dtype = np.dtype({
            'names': ['x', 'y', 'z'],
            'formats': ['<f4', '<f4', '<f4'],
            'offsets': [0, 4, 8], 
            'itemsize': cloud_msg.point_step
        })

        pts = np.frombuffer(cloud_msg.data, dtype=dtype)
        
        # Filter out NaN and Inf values efficiently
        valid_mask = np.isfinite(pts['x']) & np.isfinite(pts['y']) & np.isfinite(pts['z'])
        pts = pts[valid_mask]

        return np.column_stack((pts['x'], pts['y'], pts['z']))

    def get_cached_matrix(self, frame_id, target_frame):
        """Fetches or calculates the transformation matrix to avoid repeated TF math."""
        if frame_id not in self.cached_transforms:
            tf_msg = self.tf_buffer.lookup_transform(
                target_frame, frame_id, rclpy.time.Time()
            )
            
            t = tf_msg.transform.translation
            q = tf_msg.transform.rotation

            translation = np.array([t.x, t.y, t.z])
            x, y, z, w = q.x, q.y, q.z, q.w
            
            rotation_matrix = np.array([
                [1 - 2*(y**2 + z**2),     2*(x*y - w*z),     2*(x*z + w*y)],
                [    2*(x*y + w*z), 1 - 2*(x**2 + z**2),     2*(y*z - w*x)],
                [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
            ])

            self.cached_transforms[frame_id] = (rotation_matrix, translation)
            self.get_logger().info(f"Cached TF for frame: {frame_id}")

        return self.cached_transforms[frame_id]

    def sync_callback(self, msg_l, msg_r):
        target_frame = 'base_link'

        try:
            # 1. Get cached transformation matrices
            rot_matrix_l, trans_l = self.get_cached_matrix(msg_l.header.frame_id, target_frame)
            rot_matrix_r, trans_r = self.get_cached_matrix(msg_r.header.frame_id, target_frame)

            # 2. Extract instantly via memory buffer
            points_l = self.extract_xyz(msg_l)
            points_r = self.extract_xyz(msg_r)

            # 3. Apply the transforms (Vectorized dot product)
            points_l_tf = np.dot(points_l, rot_matrix_l.T) + trans_l if points_l.size > 0 else np.empty((0,3))
            points_r_tf = np.dot(points_r, rot_matrix_r.T) + trans_r if points_r.size > 0 else np.empty((0,3))

            # ==========================================
            # OBSTACLE DETECTION & HYSTERESIS LOGIC
            # ==========================================
            max_z_l = -100.0
            high_points_l = 0
            if points_l_tf.size > 0:
                z_l = points_l_tf[:, 2]
                if z_l.size > 0:
                    max_z_l = float(np.max(z_l))
                    high_points_l = np.sum(z_l > self.OBSTACLE_HEIGHT)

            max_z_r = -100.0
            high_points_r = 0
            if points_r_tf.size > 0:
                z_r = points_r_tf[:, 2]
                if z_r.size > 0:
                    max_z_r = float(np.max(z_r))
                    high_points_r = np.sum(z_r > self.OBSTACLE_HEIGHT)

            # --- Left Camera Hysteresis ---
            if high_points_l > self.MIN_OBSTACLE_POINTS:
                self.left_bad_frames = min(self.left_bad_frames + 1, self.PERSISTENCE_LIMIT)
            else:
                self.left_bad_frames = max(self.left_bad_frames - 1, 0)
            
            left_valid = (self.left_bad_frames < self.PERSISTENCE_LIMIT)

            # --- Right Camera Hysteresis ---
            if high_points_r > self.MIN_OBSTACLE_POINTS:
                self.right_bad_frames = min(self.right_bad_frames + 1, self.PERSISTENCE_LIMIT)
            else:
                self.right_bad_frames = max(self.right_bad_frames - 1, 0)
            
            right_valid = (self.right_bad_frames < self.PERSISTENCE_LIMIT)

            # Throttled diagnostics printout
            left_state_str = "OPEN" if left_valid else "KILLED"
            right_state_str = "OPEN" if right_valid else "KILLED"
            
            self.get_logger().info(
                f"Max Z (L: {max_z_l:.2f}m, R: {max_z_r:.2f}m) | State (L: {left_state_str}, R: {right_state_str})",
                throttle_duration_sec=2.0
            )

            if not left_valid:
                self.get_logger().warn("[FLAP-FIX] Left cam persistence locked to KILLED.", throttle_duration_sec=1.0)
            if not right_valid:
                self.get_logger().warn("[FLAP-FIX] Right cam persistence locked to KILLED.", throttle_duration_sec=1.0)

            # 4. Append only the valid point clouds together
            valid_points = []
            if left_valid and points_l_tf.size > 0:
                valid_points.append(points_l_tf)
            if right_valid and points_r_tf.size > 0:
                valid_points.append(points_r_tf)

            if not valid_points:
                # Both cameras killed, skip publishing
                return

            all_points = np.vstack(valid_points).astype(np.float32)

            # 5. Create the fused message
            fused_msg = PointCloud2()
            fused_msg.header.stamp = msg_l.header.stamp
            fused_msg.header.frame_id = target_frame
            fused_msg.height = 1
            fused_msg.width = all_points.shape[0]
            fused_msg.fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
            ]
            fused_msg.is_bigendian = False
            fused_msg.point_step = 12 
            fused_msg.row_step = fused_msg.point_step * fused_msg.width
            fused_msg.is_dense = True
            
            fused_msg.data = all_points.tobytes()

            # 6. Publish
            self.pc_pub.publish(fused_msg)

        except TransformException as ex:
            self.get_logger().warn(f'TF Error: {ex}', throttle_duration_sec=2.0)

def main(args=None):
    rclpy.init(args=args)
    node = CameraFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()