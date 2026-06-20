#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import message_filters

# TF2 imports
from tf2_ros import Buffer, TransformListener, TransformException

import numpy as np

class CameraFusionNode(Node):
    def __init__(self):
        super().__init__('camera_fusion_node')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Cache for transformation matrices (Assumes static sensor mounting)
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

        self.get_logger().info("Lightning Camera Fusion Node with Blockage Protection Started.")

    def extract_xyz(self, cloud_msg):
        """Extracts X, Y, Z instantly using native memory buffers and hardcoded offsets."""
        if len(cloud_msg.data) == 0:
            return np.empty((0, 3), dtype=np.float32)

        # Hardcoded offsets for standard PointCloud2 formats (x=0, y=4, z=8)
        dtype = np.dtype({
            'names': ['x', 'y', 'z'],
            'formats': ['<f4', '<f4', '<f4'],
            'offsets': [0, 4, 8], 
            'itemsize': cloud_msg.point_step
        })

        pts = np.frombuffer(cloud_msg.data, dtype=dtype)
        
        # Filter out NaN values efficiently
        valid_mask = ~np.isnan(pts['x']) & ~np.isnan(pts['y']) & ~np.isnan(pts['z'])
        pts = pts[valid_mask]

        return np.column_stack((pts['x'], pts['y'], pts['z']))

    def get_cached_matrix(self, frame_id, target_frame):
        """Fetches or calculates the transformation matrix to avoid repeated TF math."""
        if frame_id not in self.cached_transforms:
            # Look up the transform only once per sensor
            tf_msg = self.tf_buffer.lookup_transform(
                target_frame, frame_id, rclpy.time.Time()
            )
            
            t = tf_msg.transform.translation
            q = tf_msg.transform.rotation

            translation = np.array([t.x, t.y, t.z])
            x, y, z, w = q.x, q.y, q.z, q.w
            
            # Convert quaternion to rotation matrix once
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
            # 1. Extract raw coordinates instantly via memory mapping
            points_l = self.extract_xyz(msg_l)
            points_r = self.extract_xyz(msg_r)

            # 2. Blockage Check Parameters
            # Points closer than 5cm (0.05m) are classified as severe lens obstruction
            BLOCK_DISTANCE_THRESHOLD = 0.05 
            # Minimum point cluster size to trigger a safety override
            BLOCK_POINT_COUNT_THRESHOLD = 150 

            is_left_blocked = False
            is_right_blocked = False

            # Check Left Sensor for close-up leaf obstruction
            if points_l.size > 0:
                dist_l = np.linalg.norm(points_l, axis=1)
                close_points_l = np.sum(dist_l < BLOCK_DISTANCE_THRESHOLD)
                if close_points_l > BLOCK_POINT_COUNT_THRESHOLD:
                    is_left_blocked = True
                    self.get_logger().warn("⚠️ LEFT SENSOR BLOCKED BY LEAF! Dropping left frame.")

            # Check Right Sensor for close-up leaf obstruction
            if points_r.size > 0:
                dist_r = np.linalg.norm(points_r, axis=1)
                close_points_r = np.sum(dist_r < BLOCK_DISTANCE_THRESHOLD)
                if close_points_r > BLOCK_POINT_COUNT_THRESHOLD:
                    is_right_blocked = True
                    self.get_logger().warn("⚠️ RIGHT SENSOR BLOCKED BY LEAF! Dropping right frame.")

            # 3. Decision Matrix (Enforce single-camera backup modes dynamically)
            if is_left_blocked and is_right_blocked:
                self.get_logger().error("🛑 BOTH SENSORS BLOCKED! Blind tracking active.")
                return # Abort processing for this loop iteration to protect path planners

            elif is_left_blocked:
                # Left is blind -> Process only Right camera data
                if points_r.size == 0: return
                rot_matrix_r, trans_r = self.get_cached_matrix(msg_r.header.frame_id, target_frame)
                all_points = np.dot(points_r, rot_matrix_r.T) + trans_r

            elif is_right_blocked:
                # Right is blind -> Process only Left camera data
                if points_l.size == 0: return
                rot_matrix_l, trans_l = self.get_cached_matrix(msg_l.header.frame_id, target_frame)
                all_points = np.dot(points_l, rot_matrix_l.T) + trans_l

            else:
                # Both clear -> Execute full matrix fusion pipeline
                if points_l.size == 0 or points_r.size == 0:
                    return

                rot_matrix_l, trans_l = self.get_cached_matrix(msg_l.header.frame_id, target_frame)
                rot_matrix_r, trans_r = self.get_cached_matrix(msg_r.header.frame_id, target_frame)

                points_l_tf = np.dot(points_l, rot_matrix_l.T) + trans_l
                points_r_tf = np.dot(points_r, rot_matrix_r.T) + trans_r
                all_points = np.vstack((points_l_tf, points_r_tf))

            # 4. Final Data Alignment & Output Formatting
            all_points = all_points.astype(np.float32)

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

            # 5. Broadcast to Path Extraction Network
            self.pc_pub.publish(fused_msg)

        except TransformException as ex:
            self.get_logger().warn(f'Could not transform point clouds: {ex}')

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