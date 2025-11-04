#!/usr/bin/env python3
"""
Example usage of the Z1 Gym environment with end-effector pose control using joint commands
and non-blocking step execution.

This example demonstrates how to use the EEPoseCtrlJointCmdWrapper with the new
wait argument to control the Z1 robot arm's end-effector pose using jointCtrlCmd.
"""

import sys
import os
import numpy as np
import time
import torch
import torch.nn as nn
import cv2
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import signal
import atexit
from scipy.ndimage import gaussian_filter1d

# Add the envs directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "envs"))
from envs.z1_env_jointctrl_wait_R import EEPoseCtrlJointCmdWrapper

# # custom (hand-made), 3x5 charuco reference marker: 1
# T_B_M = np.array([[0.00000000, 0.00000000, 1.00000000, 0.76401688],
#                 [ 0.00000000, -1.00000000, 0.00000000,  0.0615125 ],
#                 [1.00000000, 0.00000000,  0.00000000,  0.56736933],
#                 [ 0.,          0.,          0.,          1.        ]])

# # custom (hand-made), 3x5 charuco reference marker: 1 (lowest table)
T_B_M = np.array([[0.00000000, 0.00000000, 1.00000000, 0.76401688],
                [ 0.00000000, -1.00000000, 0.00000000,  0.0615125 ],
                [1.00000000, 0.00000000,  0.00000000,  0.73136933],
                [ 0.,          0.,          0.,          1.        ]])

# # # custom (hand-made)
# T_B_M = np.array([[-0.0211232, 0.00961882, -0.99973061, 0.86901688],
#                 [ 0.00934025, -0.99990818, -0.00981788,  0.0965125 ],
#                 [-0.99973325, -0.00954512,  0.02103141,  0.55736933],
#                 [ 0.,          0.,          0.,          1.        ]])

# original (accurate value when we make the robot to reach the [0,0,0.4] in marker coordinate) [0.5507966 , 0.12502528, 0.28048738]
# T_B_M = np.array([[-0.0211232, 0.00961882, -0.99973061, 0.87401688],
#                 [ 0.00934025, -0.99990818, -0.00981788,  0.0965125 ],
#                 [-0.99973325, -0.00954512,  0.02103141,  0.54736933],
#                 [ 0.,          0.,          0.,          1.        ]])

class DummyMLP(nn.Module):
    """Simple dummy MLP model for thread-torch compatibility testing."""
    def __init__(self, input_dim=21, hidden_dim=64, output_dim=8):
        super(DummyMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)


# ========== Camera and ArUco Marker Detection Functions ==========

def init_realsense_camera(resolution="HD"):
    """
    Initialize RealSense D435 camera.
    
    Args:
        resolution: Camera resolution ("VGA", "HD", "FHD")
    
    Returns:
        pipeline: RealSense pipeline object
        align: Frame aligner object
        camera_matrix: Camera intrinsic matrix (3x3)
        dist_coeffs: Distortion coefficients
        success: Whether initialization was successful
    """
    try:
        print("Initializing RealSense D435 camera...")
        pipeline = rs.pipeline()
        config = rs.config()
        
        # Get resolution settings
        resolution_settings = {
            "VGA": (640, 480, 30),
            "HD": (1280, 720, 30),
            "FHD": (1920, 1080, 30)
        }
        
        if resolution not in resolution_settings:
            print(f"Unknown resolution '{resolution}', using HD as default.")
            resolution = "HD"
        
        width, height, fps = resolution_settings[resolution]
        
        # Configure color and depth streams
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        
        # Create aligner
        align = rs.align(rs.stream.color)
        
        # Start streaming
        pipeline.start(config)
        print(f"RealSense D435 camera initialized successfully with {width}x{height} @ {fps}fps.")
        
        # Get camera intrinsic parameters
        profile = pipeline.get_active_profile()
        color_stream = profile.get_stream(rs.stream.color)
        
        # camera_matrix = np.array([[592.5122474,    0.,         329.31307053],
        #                         [  0.,         592.41377421, 248.28416052],
        #                         [  0.,           0.,           1.        ]])
        # dist_coeffs = np.array([ 4.32352189e-02,  4.27595321e-01,  1.99344572e-03, -7.07460350e-04, -1.65811952e+00])
        camera_matrix = np.array([[594.77780809,   0.,         330.89839472],
                                [  0.,         594.59358254, 244.54097395],
                                [  0.,           0.,           1.        ]])

        # dist_coeffs = np.array([7.11160300e-02,  1.96758488e-01, -9.70781397e-05, -1.21502610e-03, -1.18364923e+00])
        dist_coeffs = None
        
        print("\n=== Camera Intrinsic Parameters ===")
        print(f"Width: {width}, Height: {height}")
        print(f"Focal Length (fx, fy): ({camera_matrix[0,0]:.6f}, {camera_matrix[1,1]:.6f})")
        print(f"Principal Point (cx, cy): ({camera_matrix[0,2]:.6f}, {camera_matrix[1,2]:.6f})")
        print(f"Distortion Model: {dist_coeffs}")
        print("=====================================\n")
        
        return pipeline, align, camera_matrix, dist_coeffs, True
        
    except Exception as e:
        print(f"Failed to initialize camera: {e}")
        return None, None, None, None, False


def capture_realsense_image(pipeline, align):
    """
    Capture RGB and depth images from RealSense camera.
    
    Args:
        pipeline: RealSense pipeline object
        align: Frame aligner object
    
    Returns:
        rgb_image: RGB image (BGR format for OpenCV)
        depth_image: Depth image (uint16)
        success: Whether capture was successful
    """
    try:
        # Wait for frames
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            return None, None, False
        
        # Convert to numpy arrays
        rgb_image = np.asanyarray(color_frame.get_data())  # BGR format
        depth_image = np.asanyarray(depth_frame.get_data())  # uint16
        
        return rgb_image, depth_image, True
        
    except Exception as e:
        print(f"Error capturing image: {e}")
        return None, None, False


# ========== ChArUco Board Helper Functions ==========

def ensure_SE3(T: np.ndarray) -> np.ndarray:
    """Ensure T is a valid SE(3) transformation matrix."""
    Tout = T.copy()
    R = T[:3, :3]
    # Ensure rotation matrix is orthogonal
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    Tout[:3, :3] = Rn
    return Tout

def get_board_corners_3d(board):
    """ChArUco 보드의 3D 코너 좌표를 가져옵니다 (버전 독립적)."""
    if hasattr(board, "getChessboardCorners"):
        pts = board.getChessboardCorners()           # OpenCV 일부 빌드
    elif hasattr(board, "chessboardCorners"):
        pts = board.chessboardCorners                # 다른 빌드
    else:
        raise RuntimeError("CharucoBoard에 코너 접근자가 없습니다.")
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
    return pts

def make_detector_params():
    """ChArUco 마커 검출 파라미터 생성."""
    p = cv2.aruco.DetectorParameters()
    
    # APRILTAG 정제 사용 (마커에 최적화, 더 정확함)
    try:
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    except AttributeError:
        # 구 버전 OpenCV는 APRILTAG 미지원
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    
    p.cornerRefinementWinSize = 5
    p.cornerRefinementMaxIterations = 30
    p.cornerRefinementMinAccuracy = 0.01
    
    # # 추가 검출 파라미터
    # p.adaptiveThreshWinSizeMin = 3
    # p.adaptiveThreshWinSizeMax = 23
    # p.adaptiveThreshWinSizeStep = 10
    
    return p

def create_charuco_board(squaresX, squaresY, square_size_m, marker_size_m, aruco_dict_type):
    """
    ChArUco 보드를 생성합니다.
    
    Args:
        squaresX: 가로 사각형 개수
        squaresY: 세로 사각형 개수
        square_size_m: 사각형 크기 (미터)
        marker_size_m: 마커 크기 (미터)
        aruco_dict_type: ArUco 딕셔너리 타입
    
    Returns:
        board: cv2.aruco.CharucoBoard 객체
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    board = cv2.aruco.CharucoBoard(
        (squaresX, squaresY),
        square_size_m,
        marker_size_m,
        aruco_dict
    )
    return board

def detect_charuco_board(gray, board, camera_matrix, dist_coeffs):
    """
    ChArUco 보드를 감지하고 보드 자세를 추정합니다.
    
    Args:
        gray: 그레이스케일 이미지
        board: cv2.aruco.CharucoBoard 객체
        camera_matrix: 카메라 내부 파라미터 (3x3)
        dist_coeffs: 왜곡 계수
    
    Returns:
        success: 보드 감지 성공 여부
        charuco_corners: ChArUco 코너 좌표
        charuco_ids: ChArUco 코너 ID
        marker_corners: 마커 코너 좌표
        marker_ids: 마커 ID
        rvec_board: 보드 회전 벡터
        tvec_board: 보드 평행이동 벡터
    """
    ch_params = cv2.aruco.CharucoParameters()
    det_params = make_detector_params()
    cdet = cv2.aruco.CharucoDetector(board, ch_params, det_params)
    
    # 보드 감지
    out = cdet.detectBoard(gray)
    charuco_corners, charuco_ids = out[0], out[1]
    
    if len(out) >= 4:
        marker_corners, marker_ids = out[2], out[3]
    else:
        marker_corners, marker_ids = None, None
    
    # 보드 자세 추정
    rvec_board = None
    tvec_board = None
    
    if charuco_corners is not None and charuco_ids is not None and len(charuco_ids) >= 6:
        # 2D: charuco_corners -> (N,2), 3D: 보드 좌표계 코너 -> (N,3)
        cc2 = charuco_corners.reshape(-1, 2).astype(np.float32)
        ids1 = charuco_ids.reshape(-1).astype(np.int32)
        
        board_corners_3d = get_board_corners_3d(board)
        valid = (ids1 >= 0) & (ids1 < board_corners_3d.shape[0])
        objp = board_corners_3d[ids1[valid], :].astype(np.float32)  # (N,3)
        imgp = cc2[valid]  # (N,2)
        
        # 최소 6개 코너 필요
        if len(objp) >= 6:
            success, rvec_board, tvec_board = cv2.solvePnP(
                objp, imgp, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not success:
                rvec_board = None
                tvec_board = None
    
    success = (rvec_board is not None and tvec_board is not None)
    return success, charuco_corners, charuco_ids, marker_corners, marker_ids, rvec_board, tvec_board

def compute_marker_poses_from_board(board, rvec_board, tvec_board, marker_ids, marker_corners):
    """
    보드 자세로부터 각 마커의 개별 자세를 계산합니다.
    
    Args:
        board: cv2.aruco.CharucoBoard 객체
        rvec_board: 보드 회전 벡터
        tvec_board: 보드 평행이동 벡터
        marker_ids: 감지된 마커 ID 배열
        marker_corners: 감지된 마커 코너 좌표
    
    Returns:
        T_cam_marker_dict: {marker_id: T_cam_marker (4x4)} 딕셔너리
        T_cam_board: Camera to board transformation (4x4)
    """
    # 보드 회전 행렬 계산
    R_board, _ = cv2.Rodrigues(rvec_board)
    
    # 보드에서 각 마커의 보드 좌표계 위치 정보 가져오기
    try:
        if hasattr(board, 'getObjPoints'):
            board_marker_obj_points = board.getObjPoints()
            board_marker_ids = board.getIds().flatten()
        elif hasattr(board, 'objPoints'):
            board_marker_obj_points = board.objPoints
            board_marker_ids = board.ids.flatten()
        else:
            board_marker_obj_points = None
            board_marker_ids = None
    except:
        board_marker_obj_points = None
        board_marker_ids = None
    
    if board_marker_obj_points is None or board_marker_ids is None:
        return {}, None
    
    # Convert rvec, tvec to T_cam_board
    T_cam_board = np.eye(4)
    T_cam_board[:3, :3] = R_board
    T_cam_board[:3, 3] = tvec_board.flatten()
    
    T_cam_marker_dict = {}
    
    if marker_ids is not None and len(marker_ids) > 0:
        detected_ids = marker_ids.flatten()
        for i, marker_id in enumerate(detected_ids):
            marker_id = int(marker_id)
            
            # 보드에서 해당 마커 ID 찾기
            marker_idx = np.where(board_marker_ids == marker_id)[0]
            if len(marker_idx) == 0:
                continue
            
            marker_idx = marker_idx[0]
            # 마커의 4개 코너 (보드 좌표계)
            marker_corners_board = board_marker_obj_points[marker_idx]  # (4, 3)
            # 마커 중심 (보드 좌표계)
            marker_center_board = np.mean(marker_corners_board, axis=0).reshape(3, 1)  # (3, 1)
            
            # 보드 좌표계를 카메라 좌표계로 변환: tvec_cam = R_board @ tvec_board + tvec_b
            marker_center_cam = R_board @ marker_center_board + tvec_board
            
            # 마커는 보드 평면에 있으므로 회전은 보드와 동일
            rvec_marker = rvec_board.copy()
            tvec_marker = marker_center_cam
            
            # SE(3) 변환 행렬 생성 (camera -> marker)
            R_marker, _ = cv2.Rodrigues(rvec_marker)
            T_cm = np.eye(4)
            T_cm[:3, :3] = R_marker
            T_cm[:3, 3] = tvec_marker.flatten()
            T_cm = ensure_SE3(T_cm)
            
            T_cam_marker_dict[marker_id] = T_cm
    
    return T_cam_marker_dict, T_cam_board

# ========== ArUco Marker Detection Functions (Legacy, for reference) ==========

def detect_aruco_marker(rgb_image, camera_matrix, dist_coeffs, marker_size, 
                        aruco_dict_type=cv2.aruco.DICT_6X6_250):
    """
    Detect ArUco markers and estimate their poses.
    
    Args:
        rgb_image: RGB image (BGR format)
        camera_matrix: Camera intrinsic matrix (3x3)
        dist_coeffs: Distortion coefficients
        marker_size: Marker size in meters
        aruco_dict_type: ArUco dictionary type
    
    Returns:
        marker_poses: Dict of {marker_id: T_cm} where T_cm is SE(3) transformation matrix from camera to marker (4x4)
        corners: Detected marker corners
        ids: Detected marker IDs
        success: Whether detection was successful
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        
        # Create ArUco detector
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        
        # Detect markers
        corners, ids, rejected = detector.detectMarkers(gray)
        
        if ids is None or len(ids) == 0:
            return {}, None, None, False
        
        # Process all detected markers
        marker_poses = {}
        marker_corners_3d = np.array([
            [-marker_size/2,  marker_size/2, 0],  # top-left
            [ marker_size/2,  marker_size/2, 0],  # top-right  
            [ marker_size/2, -marker_size/2, 0],  # bottom-right
            [-marker_size/2, -marker_size/2, 0]   # bottom-left
        ], dtype=np.float32)
        
        # Process each detected marker
        for marker_idx, (corner, marker_id) in enumerate(zip(corners, ids)):
            marker_id = marker_id[0]
            
            # Pose estimation using solvePnP
            try:
                success, rvec, tvec = cv2.solvePnP(
                    marker_corners_3d, corner[0], camera_matrix, dist_coeffs
                )
                
                if success:
                    # Convert to SE(3)
                    R_cm, _ = cv2.Rodrigues(rvec)
                    T_cm = np.eye(4)
                    T_cm[:3, :3] = R_cm
                    T_cm[:3, 3] = tvec.flatten()
                    
                    # Store the pose for this marker
                    marker_poses[marker_id] = T_cm
                else:
                    print(f"Failed to estimate pose for marker {marker_id}")
                    continue
            except Exception as e:
                print(f"Pose estimation failed for marker {marker_id}: {e}")
                continue
        
        if len(marker_poses) == 0:
            return {}, corners, ids, False
        
        return marker_poses, corners, ids, True
        
    except Exception as e:
        print(f"Error detecting ArUco marker: {e}")
        return {}, None, None, False


def create_aruco_board_from_config(marker_config, marker_size, aruco_dict):
    """
    1) Board 정의: marker_config의 relative_poses를 사용해서 cv2.aruco.Board 생성
    
    Args:
        marker_config: Dict containing relative_poses information
        marker_size: Marker size in meters
        aruco_dict: ArUco dictionary object
    
    Returns:
        board: cv2.aruco.Board object
        T_board_marker_dict: Dict of {marker_id: T_board_marker} transformations
    """
    if marker_config is None or 'relative_poses' not in marker_config:
        return None, {}
    
    # Collect all marker IDs
    all_marker_ids = set()
    for rel_pose_info in marker_config['relative_poses']:
        all_marker_ids.add(rel_pose_info['marker1_id'])
        all_marker_ids.add(rel_pose_info['marker2_id'])
    
    # Build T_board_marker for each marker
    # Board coordinate frame = marker 0's frame (reference marker)
    T_board_marker_dict = {}
    T_board_marker_dict[0] = np.eye(4)  # Marker 0 is the board origin
    
    # Build dependency graph from relative poses
    relative_pose_constraints = {}
    for rel_pose_info in marker_config['relative_poses']:
        marker1_id = rel_pose_info['marker1_id']
        marker2_id = rel_pose_info['marker2_id']
        T_m1m2 = np.array(rel_pose_info['relative_pose'])
        relative_pose_constraints[marker2_id] = (marker1_id, T_m1m2)
    
    # Propagate from marker 0 to all other markers
    processed = set([0])
    queue = [0]
    
    while queue:
        current_marker_id = queue.pop(0)
        for marker2_id, (marker1_id, T_m1m2) in relative_pose_constraints.items():
            if marker1_id == current_marker_id and marker2_id not in processed:
                T_board_marker_dict[marker2_id] = T_board_marker_dict[marker1_id] @ T_m1m2
                processed.add(marker2_id)
                queue.append(marker2_id)
    
    # Build objPoints and ids for Board
    # objPoints: list of arrays, each array is [4x3] (4 corners, 3D in board frame)
    obj_points_list = []
    ids_list = []
    
    # Define marker corners in marker's local frame
    marker_corners_local = np.array([
        [-marker_size/2,  marker_size/2, 0],  # top-left
        [ marker_size/2,  marker_size/2, 0],  # top-right  
        [ marker_size/2, -marker_size/2, 0],  # bottom-right
        [-marker_size/2, -marker_size/2, 0]   # bottom-left
    ], dtype=np.float32)
    
    for marker_id in sorted(all_marker_ids):
        if marker_id in T_board_marker_dict:
            # Transform marker corners from marker frame to board frame
            T_bm = T_board_marker_dict[marker_id]
            marker_corners_homo = np.concatenate([marker_corners_local, np.ones((4, 1))], axis=1)  # [4, 4]
            board_corners = (T_bm @ marker_corners_homo.T).T[:, :3]  # [4, 3]
            
            # Ensure float32 type and correct shape for OpenCV (contiguous array)
            board_corners = np.ascontiguousarray(board_corners, dtype=np.float32).reshape(4, 3)
            obj_points_list.append(board_corners)
            ids_list.append(marker_id)
    
    # Create Board object
    # objPoints should be a list of arrays, each array is [4, 3] with dtype=np.float32
    ids_array = np.array(ids_list, dtype=np.int32)
    
    if len(obj_points_list) == 0:
        return None, {}
    
    # Ensure all arrays are properly formatted and contiguous
    obj_points_array = []
    for corners in obj_points_list:
        corners_float32 = np.ascontiguousarray(corners, dtype=np.float32)
        if corners_float32.shape != (4, 3):
            print(f"Warning: Invalid corner shape {corners_float32.shape}, expected (4, 3)")
            continue
        obj_points_array.append(corners_float32)
    
    if len(obj_points_array) == 0:
        return None, {}
    
    board = cv2.aruco.Board(obj_points_array, aruco_dict, ids_array)
    
    return board, T_board_marker_dict


def detect_aruco_board(rgb_image, board, camera_matrix, dist_coeffs, T_board_marker_dict=None, marker_size=None, aruco_dict_type=cv2.aruco.DICT_6X6_250):
    """
    2) 검출 & 상대 SE(3) 제약을 반영한 공동 최적화
    detectMarkers 후 estimatePoseBoard를 사용하여 board pose 추정
    
    Args:
        rgb_image: RGB image (BGR format)
        board: cv2.aruco.Board object
        camera_matrix: Camera intrinsic matrix (3x3)
        dist_coeffs: Distortion coefficients
        aruco_dict_type: ArUco dictionary type
    
    Returns:
        success: Whether board pose estimation was successful
        rvec: Rotation vector (3x1)
        tvec: Translation vector (3x1)
        corners: Detected marker corners
        ids: Detected marker IDs
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        
        # Create ArUco detector
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        
        # Detect markers
        corners, ids, rejected = detector.detectMarkers(gray)
        
        if ids is None or len(ids) == 0:
            return False, None, None, None, None
        
        # (Optional) Refine detected markers using board information
        
        corners, ids, rejectedImgPoints, recoveredIds = detector.refineDetectedMarkers(
            gray, board, corners, ids, rejected, camera_matrix, dist_coeffs
        )
        
        
        # Estimate board pose
        # If estimatePoseBoard is not available, implement it manually using solvePnP
        # Collect all 3D object points and 2D image points from detected markers
        obj_points_all = []
        img_points_all = []
        
        # Get board's objPoints (3D corners in board coordinate system)
        # Try different methods to access board data depending on OpenCV version
        board_obj_points = None
        board_ids = None
        
        try:
            if hasattr(board, 'getObjPoints'):
                board_obj_points = board.getObjPoints()
            elif hasattr(board, 'objPoints'):
                board_obj_points = board.objPoints
                
            if hasattr(board, 'getIds'):
                board_ids = board.getIds().flatten()
            elif hasattr(board, 'ids'):
                board_ids = board.ids.flatten()
        except Exception as e:
            pass
        
        # If cannot access board data, reconstruct from T_board_marker_dict
        if board_obj_points is None or board_ids is None:
            if T_board_marker_dict is None or marker_size is None:
                print("Error: Cannot access board data and T_board_marker_dict not provided")
                return False, None, None, corners, ids
            
            # Reconstruct board_obj_points from T_board_marker_dict
            marker_corners_local = np.array([
                [-marker_size/2,  marker_size/2, 0],  # top-left
                [ marker_size/2,  marker_size/2, 0],  # top-right  
                [ marker_size/2, -marker_size/2, 0],  # bottom-right
                [-marker_size/2, -marker_size/2, 0]   # bottom-left
            ], dtype=np.float32)
            
            board_obj_points = []
            board_ids = []
            for marker_id in sorted(T_board_marker_dict.keys()):
                T_bm = T_board_marker_dict[marker_id]
                marker_corners_homo = np.concatenate([marker_corners_local, np.ones((4, 1))], axis=1)
                board_corners = (T_bm @ marker_corners_homo.T).T[:, :3]
                board_obj_points.append(board_corners.astype(np.float32))
                board_ids.append(marker_id)
            board_ids = np.array(board_ids, dtype=np.int32)
        
        # Create mapping from marker ID to board objPoints index
        id_to_idx = {int(board_ids[i]): i for i in range(len(board_ids))}
        
        # Collect all detected markers' corners
        for i, marker_id in enumerate(ids.flatten()):
            marker_id = int(marker_id)
            if marker_id in id_to_idx:
                # Get 3D corners for this marker in board coordinate system
                marker_3d_corners = np.array(board_obj_points[id_to_idx[marker_id]], dtype=np.float32)  # [4, 3]
                
                # Get 2D corners for this detected marker
                marker_2d_corners = corners[i][0]  # [4, 2]
                
                # Add to collections
                obj_points_all.append(marker_3d_corners)
                img_points_all.append(marker_2d_corners)
        
        if len(obj_points_all) == 0:
            return False, None, None, corners, ids
        
        # Concatenate all points
        obj_points_flat = np.vstack(obj_points_all)  # [N*4, 3]
        img_points_flat = np.vstack(img_points_all)  # [N*4, 2]
        
        # Use solvePnP to estimate board pose (all markers together)
        success, rvec, tvec = cv2.solvePnP(
            obj_points_flat, img_points_flat, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success:
            return True, rvec, tvec, corners, ids
        else:
            return False, None, None, corners, ids
            
    except Exception as e:
        print(f"Error detecting ArUco board: {e}")
        return False, None, None, None, None


def compute_individual_marker_poses_from_board(board, rvec_board, tvec_board, T_board_marker_dict):
    """
    3) 각 마커의 개별 자세가 필요하면
    Board pose로부터 각 마커의 개별 자세를 계산: T_cam_marker = T_cam_board @ T_board_marker
    
    Args:
        board: cv2.aruco.Board object
        rvec_board: Board rotation vector (from estimatePoseBoard)
        tvec_board: Board translation vector (from estimatePoseBoard)
        T_board_marker_dict: Dict of {marker_id: T_board_marker} transformations
    
    Returns:
        T_cam_marker_dict: Dict of {marker_id: T_cam_marker} transformations
        T_cam_board: Camera to board transformation (4x4)
    """
    # Convert rvec, tvec to T_cam_board
    R_cb, _ = cv2.Rodrigues(rvec_board)
    T_cb = np.eye(4)
    T_cb[:3, :3] = R_cb
    T_cb[:3, 3] = tvec_board.flatten()
    
    # Compute T_cam_marker for each marker
    T_cam_marker_dict = {}
    for marker_id, T_bm in T_board_marker_dict.items():
        # T_cam_marker = T_cam_board @ T_board_marker
        T_cam_marker_dict[marker_id] = T_cb @ T_bm
    
    return T_cam_marker_dict, T_cb


def compute_marker_poses_with_config(marker_poses_dict, marker_config=None):
    """
    Compute camera-to-marker transformations using marker_config to handle relative poses.
    
    Args:
        marker_poses_dict: Dict of {marker_id: T_cm} where T_cm is camera-to-marker transformation
        marker_config: Optional dict containing relative_poses information
    
    Returns:
        T_cm_reference: Camera-to-reference-marker transformation (4x4)
        reference_marker_id: ID of the reference marker used
        all_marker_poses: Dict of {marker_id: T_cm} with computed poses for all markers
    """
    if len(marker_poses_dict) == 0:
        return None, None, {}
    
    # If no marker_config or only one marker, use the first detected marker
    if marker_config is None or 'relative_poses' not in marker_config:
        # Use first marker as reference
        reference_marker_id = list(marker_poses_dict.keys())[0]
        T_cm_reference = marker_poses_dict[reference_marker_id]
        return T_cm_reference, reference_marker_id, marker_poses_dict
    
    # Find reference marker (marker1_id=0 in relative_poses, or first marker with id=0)
    reference_marker_id = None
    for rel_pose_info in marker_config['relative_poses']:
        if rel_pose_info['marker1_id'] == 0:
            reference_marker_id = 0
            break
    
    # If no marker with id=0 found, use first detected marker
    if reference_marker_id is None:
        reference_marker_id = list(marker_poses_dict.keys())[0]
        T_cm_reference = marker_poses_dict[reference_marker_id]
        return T_cm_reference, reference_marker_id, marker_poses_dict
    
    # Check if reference marker is detected
    if reference_marker_id not in marker_poses_dict:
        # Fallback: use first detected marker
        reference_marker_id = list(marker_poses_dict.keys())[0]
        T_cm_reference = marker_poses_dict[reference_marker_id]
        return T_cm_reference, reference_marker_id, marker_poses_dict
    
    # Get reference marker pose
    T_cm_reference = marker_poses_dict[reference_marker_id]
    all_marker_poses = {reference_marker_id: T_cm_reference}
    
    # Compute poses for other markers using relative poses
    for rel_pose_info in marker_config['relative_poses']:
        marker1_id = rel_pose_info['marker1_id']
        marker2_id = rel_pose_info['marker2_id']
        T_m1m2 = np.array(rel_pose_info['relative_pose'])
        
        # If marker1 is the reference and we have T_cm for marker1
        if marker1_id == reference_marker_id and marker1_id in all_marker_poses:
            # T_cm2 = T_cm1 @ T_m1m2
            T_cm1 = all_marker_poses[marker1_id]
            T_cm2 = T_cm1 @ T_m1m2
            all_marker_poses[marker2_id] = T_cm2
        # If marker2 is the reference and we have T_cm for marker1
        elif marker2_id == reference_marker_id and marker1_id in marker_poses_dict:
            # T_cm1 = T_cm2 @ T_m2m1 = T_cm2 @ inv(T_m1m2)
            T_cm2 = marker_poses_dict[marker2_id] if marker2_id in marker_poses_dict else all_marker_poses[marker2_id]
            T_m2m1 = np.linalg.inv(T_m1m2)
            T_cm1 = T_cm2 @ T_m2m1
            all_marker_poses[marker1_id] = T_cm1
        # If both markers are detected, use the one that's already computed
        elif marker1_id in all_marker_poses:
            T_cm1 = all_marker_poses[marker1_id]
            T_cm2 = T_cm1 @ T_m1m2
            all_marker_poses[marker2_id] = T_cm2
        elif marker2_id in all_marker_poses:
            T_cm2 = all_marker_poses[marker2_id]
            T_m2m1 = np.linalg.inv(T_m1m2)
            T_cm1 = T_cm2 @ T_m2m1
            all_marker_poses[marker1_id] = T_cm1
        # If neither is computed yet but both are detected, use detected pose
        elif marker1_id in marker_poses_dict:
            all_marker_poses[marker1_id] = marker_poses_dict[marker1_id]
        elif marker2_id in marker_poses_dict:
            all_marker_poses[marker2_id] = marker_poses_dict[marker2_id]
    
    # Add any remaining detected markers that weren't in the config
    for marker_id, T_cm in marker_poses_dict.items():
        if marker_id not in all_marker_poses:
            all_marker_poses[marker_id] = T_cm
    
    return T_cm_reference, reference_marker_id, all_marker_poses


def project_3d_to_image(point_3d, camera_matrix):
    """
    Project a 3D point in camera coordinates to image coordinates.
    
    Args:
        point_3d: 3D point in camera frame (3,)
        camera_matrix: Camera intrinsic matrix (3x3)
    
    Returns:
        point_2d: 2D image coordinates (u, v)
        visible: Whether the point is visible (z > 0)
    """
    if point_3d[2] <= 0:
        return None, False
    
    # Project to image
    point_2d_homo = camera_matrix @ point_3d
    point_2d = point_2d_homo[:2] / point_2d_homo[2]
    
    return point_2d, True


def inference_function(observation, square_vertices, orientation_vertices, total_steps, current_step, dummy_mlp, sequence_length=16, inference_time=0.15):
    """
    Simulate inference function that takes observation and returns action sequence.
    This uses the same square sequence generation as the original example.
    
    Args:
        observation: Current observation from the environment
        square_vertices: List of square vertices defining the path
        orientation_vertices: List of orientations for each vertex
        total_steps: Total number of steps for the entire square path
        current_step: Current step to start the sequence from
        dummy_mlp: Dummy MLP model for thread-torch compatibility testing
        sequence_length: Length of target pose sequence to generate
        inference_time: Time to simulate inference (seconds)
        
    Returns:
        action_sequence: Generated action sequence [sequence_length, 8]
    """
    print(f"Starting inference with observation shape {observation.shape} (simulating {inference_time:.3f}s)...")
    
    # Dummy MLP inference for thread-torch compatibility testing
    with torch.no_grad():
        # Convert observation to tensor and add batch dimension
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
        dummy_output = dummy_mlp(obs_tensor)
        print(f"Dummy MLP inference completed: device: {dummy_output.device}, input shape {obs_tensor.shape}, output shape {dummy_output.shape}")
    
    # Simulate inference time
    time.sleep(inference_time)
    
    # Extract current position and orientation from observation
    current_pos = observation[12:15]  # End-effector position
    current_orient = observation[15:19]  # End-effector orientation (quaternion)
    current_gripper = observation[19]  # Gripper position
    
    # Use the same square sequence generation as the original example
    action_sequence = generate_action_sequence(
        current_pos, current_orient, current_gripper, 
        square_vertices, orientation_vertices, 
        total_steps, current_step, sequence_length
    )
    
    print(f"Inference completed: generated sequence with shape {action_sequence.shape}")
    return action_sequence


def generate_action_sequence(base_position, base_orientation, base_gripper, square_vertices, orientation_vertices, total_steps, current_step, sequence_length=10):
    """
    Generate a target pose sequence that follows the square path.
    This function generates the sequence directly without simulation of inference time.
    
    Args:
        base_position: Base position to use as starting point
        base_orientation: Base orientation to use
        base_gripper: Base gripper position to use
        square_vertices: List of square vertices defining the path
        orientation_vertices: List of orientations for each vertex
        total_steps: Total number of steps for the entire square path
        current_step: Current step to start the sequence from
        sequence_length: Length of target pose sequence to generate
        
    Returns:
        target_sequence: Pre-computed target pose sequence [sequence_length, 8]
    """
    sequence = np.zeros((sequence_length, 8))
    
    # Calculate square path parameters based on total_steps
    total_edges = 4  # 4 edges of the square
    steps_per_edge = total_steps // total_edges
    
    for i in range(sequence_length):
        # Calculate the actual step number in the total path
        actual_step = current_step + i
        
        # Calculate which edge we're on and position along that edge
        edge_index = actual_step // steps_per_edge
        step_in_edge = actual_step % steps_per_edge
        
        # Ensure we don't go beyond the last vertex
        if edge_index >= 4:
            edge_index = 3
            step_in_edge = steps_per_edge - 1
        
        # Calculate current target position along the current edge
        start_vertex = square_vertices[edge_index]
        end_vertex = square_vertices[edge_index + 1]
        
        # Interpolate position along the edge
        t = step_in_edge / steps_per_edge
        target_position = start_vertex + t * (end_vertex - start_vertex)
        
        # Interpolate orientation along the edge
        start_orientation = orientation_vertices[edge_index]
        end_orientation = orientation_vertices[edge_index + 1]
        
        # Use SLERP (Spherical Linear Interpolation) for proper quaternion interpolation
        start_rot = R.from_quat(start_orientation)
        end_rot = R.from_quat(end_orientation)
        
        # Create a rotation that represents the interpolation
        # This uses scipy's built-in SLERP functionality
        new_rot = start_rot * (start_rot.inv() * end_rot) ** t
        
        # Convert back to quaternion
        new_orientation = new_rot.as_quat().squeeze()
        
        # Gripper: alternate between open and close for each edge
        # Edge 0: open (1), Edge 1: close (-1), Edge 2: open (1), Edge 3: close (-1)
        if edge_index % 2 == 0:
            new_gripper = 1.0  # Open
        else:
            new_gripper = -1.0  # Close
        
        # Combine into action
        sequence[i] = np.concatenate([target_position, new_orientation, [new_gripper]])
    
    return sequence

def backproject_pixel(K, u, v, depth, dist_coeffs=None):
    """
    Backproject a 2D pixel coordinate to 3D point in camera coordinates.
    If dist_coeffs is provided, undistorts the pixel coordinate first.
    
    Args:
        K: (3,3) camera intrinsic matrix
        u, v: pixel coordinates
        depth: depth value (z coordinate in camera frame)
        dist_coeffs: (5,) or None, distortion coefficients
    
    Returns:
        point3d: (3,) 3D point in camera coordinates
    """
    if dist_coeffs is not None:
        # Undistort the pixel coordinate first
        pixel = np.array([[u, v]], dtype=np.float32)
        pixel_undist = cv2.undistortPoints(pixel, K, dist_coeffs, P=K)
        u_undist = pixel_undist[0, 0, 0]
        v_undist = pixel_undist[0, 0, 1]
    else:
        u_undist = u
        v_undist = v
    
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    z = depth
    x = (u_undist - cx) * z / fx
    y = (v_undist - cy) * z / fy
    return np.array([x, y, z])

def apply_temporal_smoothing(data, sigma=1.0):
    """
    Apply temporal smoothing to trajectory data using Gaussian filter
    
    Args:
        data: numpy array of shape (T, 3) for translation or (T, 3) for Euler angles
        sigma: standard deviation for Gaussian filter (higher = more smoothing)
    
    Returns:
        smoothed_data: numpy array of same shape as input
    """
    if len(data.shape) == 2:
        # Apply smoothing to each component separately
        smoothed_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            smoothed_data[:, i] = gaussian_filter1d(data[:, i], sigma=sigma)
        return smoothed_data
    else:
        # For 1D data
        return gaussian_filter1d(data, sigma=sigma)

def get_dift_point_base_frame_data_for_debug(camera_matrix=None, dist_coeffs=None, droid=False):
    import zarr
    from im2flow2act.common.utility.zarr import parallel_reading
    # data_buffer_path = "/home/dcho302/slow_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/object_first/test_for_hand_eye_calib_debug"
    # data_buffer_path = "/home/dcho302/slow_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/object_first/test_for_hand_pose_calib_debug"
    # data_buffer_path = "/home/dcho302/slow_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/object_first/multi_marker_test_for_hand_pose_calib_debug"
    # data_buffer_path = "/home/dcho302/slow_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/object_first/multi_marker_test_for_hand_pose_calib_debug_w_wrist_depth_scale"
    # data_buffer_path = "/home/dcho302/slow_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/object_first/charuco_marker_test_for_hand_pose_calib_debug"
    data_buffer_path = "/home/dcho302/slow_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/object_first/charuco_marker_static_bottle"
    
    data_buffer = zarr.open(data_buffer_path, mode="a")
    episode_idx = 0
    dift_point_tracking_sequence = data_buffer[f"episode_{episode_idx}/dift_point_tracking_sequence"][:, :, :3].copy().transpose(1, 0, 2) # [N, T, 4 -> 3] -> [T, N, 3] camera frame
    if droid:
        T_mc_transformation = data_buffer[f"episode_{episode_idx}/T_mc_opt_droid"][:].copy() # [T, 4, 4]
    else:
        T_mc_transformation = data_buffer[f"episode_{episode_idx}/T_mc_opt"][:].copy() # [T, 4, 4]
    dift_points = data_buffer[f"episode_{episode_idx}/dift_points"][:].copy() # [N, 2]
    assert dift_point_tracking_sequence.shape[0] == T_mc_transformation.shape[0]
    
    T, N, _ = dift_point_tracking_sequence.shape
    T_bc_transformation = np.einsum('ij,hjk->hik', T_B_M, T_mc_transformation) # [T, 4, 4]
    

    dift_points_custom_unprojected_list = []
    depth = parallel_reading(group=data_buffer[f"episode_{episode_idx}/camera_0"], array_name="depth")[0].astype(np.float32) / 1000.0
    for dift_x, dift_y in dift_points:
        dift_points_custom_unprojected = backproject_pixel(camera_matrix, dift_x, dift_y, depth[dift_y, dift_x], dist_coeffs=dist_coeffs)
        dift_points_custom_unprojected_list.append(dift_points_custom_unprojected)
    dift_points_custom_unprojected = np.tile(np.stack(dift_points_custom_unprojected_list), (T, 1, 1)) # [T, N, 3]
    dift_points_custom_unprojected_c_homo = np.concatenate([dift_points_custom_unprojected, np.ones((T, N, 1))], axis=2) # [T, N, 4]
    dift_points_custom_unprojected_c = dift_points_custom_unprojected.copy()
    # [T, 4, 4] @ [T, N, 4] -> [T, N, 4]
    dift_points_custom_unprojected_b = np.einsum('hij,hkj->hki', T_bc_transformation, dift_points_custom_unprojected_c_homo)[:, :, :3]


    
    
    # Convert to homogeneous coordinates: [T, N, 3] -> [T, N, 4]
    camera_points_homo = np.concatenate([
        dift_point_tracking_sequence, 
        np.ones((T, N, 1))
    ], axis=2)  # [T, N, 4]
    
    # Reshape for batch matrix multiplication: [T, N, 4] -> [T*N, 4]
    camera_points_homo_flat = camera_points_homo.reshape(-1, 4)  # [T*N, 4]
    
    # Expand transformation matrices: [T, 4, 4] -> [T*N, 4, 4]
    T_bc_expanded = np.repeat(T_bc_transformation, N, axis=0)  # [T*N, 4, 4]
    T_mc_expanded = np.repeat(T_mc_transformation, N, axis=0)  # [T*N, 4, 4]
    
    # Batch matrix multiplication: [T*N, 4, 4] @ [T*N, 4] -> [T*N, 4]
    base_points_homo_flat = np.einsum('ijk,ik->ij', T_bc_expanded, camera_points_homo_flat)  # [T*N, 4]
    marker_points_homo_flat = np.einsum('ijk,ik->ij', T_mc_expanded, camera_points_homo_flat)  # [T*N, 4]
    
    # Convert back to 3D coordinates and reshape: [T*N, 4] -> [T*N, 3] -> [T, N, 3]
    dift_point_base_frame = base_points_homo_flat[:, :3].reshape(T, N, 3)  # [T, N, 3]
    dift_point_marker_frame = marker_points_homo_flat[:, :3].reshape(T, N, 3)  # [T, N, 3]



    # point_in_front_of_the_marker_homo = np.array([0.0, 0.0, 0.4, 1.0])
    point_in_front_of_the_marker_homo = np.array([0.0, 0.0, -0.4, 1.0])
    point_in_front_of_the_marker_base_frame = np.einsum('ij,j->i', T_B_M, point_in_front_of_the_marker_homo)[:3]

    
    return dift_point_base_frame, dift_point_marker_frame, dift_point_tracking_sequence, point_in_front_of_the_marker_base_frame, dift_points_custom_unprojected_c, dift_points_custom_unprojected_b

def get_estimated_hand_pose_base_frame_data_for_debug(droid=False):
    import zarr
    # data_buffer_path = "/home/dcho302/slow_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/object_first/test_for_hand_eye_calib_debug"
    # data_buffer_path = "/home/dcho302/slow_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/object_first/test_for_hand_pose_calib_debug"
    # data_buffer_path = "/home/dcho302/slow_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/object_first/multi_marker_test_for_hand_pose_calib_debug"
    # data_buffer_path = "/home/dcho302/slow_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/object_first/multi_marker_test_for_hand_pose_calib_debug_w_dist_coeff"
    # data_buffer_path = "/home/dcho302/slow_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/object_first/multi_marker_test_for_hand_pose_calib_debug_w_wrist_depth_scale"
    # data_buffer_path = "/home/dcho302/slow_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/object_first/charuco_marker_test_for_hand_pose_calib_debug"
    data_buffer_path = "/home/dcho302/slow_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/object_first/charuco_marker_static_bottle"
    data_buffer = zarr.open(data_buffer_path, mode="a")
    episode_idx = 0
    all_detected_frame_index = data_buffer[f"episode_{episode_idx}/all_detected_frame_index"][()] # scalar
    if droid:
        T_mc_transformation = data_buffer[f"episode_{episode_idx}/T_mc_opt_droid"][:].copy() # [T, 4, 4]
    else:
        T_mc_transformation = data_buffer[f"episode_{episode_idx}/T_mc_opt"][:].copy() # [T, 4, 4]
    T = T_mc_transformation.shape[0]
    

    # original ver
    # proprioception = data_buffer[f"episode_{episode_idx}/proprioception"][:].copy() # [T, 7], camera frame

    # NOTE: for debug
    import pickle
    def load_data_dict(data_path):
        """Load the data dictionary from pickle file"""
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)
        return data_dict

    data_dict = load_data_dict(data_buffer_path + f'/episode_{episode_idx}/data_dict.pkl')
    
    # use first obs
    R_cg_opt = data_dict['R_cg_opt_trajectory'][:] # [T, 3, 3]
    R_cg_opt_depth = data_dict['R_cg_opt_depth_trajectory'][:] # [T, 3, 3]
    t_cg_opt_depth = data_dict['t_cg_opt_depth_trajectory'][:] # [T, 3]
    t_cg_opt = data_dict['t_cg_opt_trajectory'][:] # [T, 3]
    t_cg_closed = data_dict['t_cg_closed_trajectory'][:] # [T, 3]
    euler_cg_opt = R.from_matrix(R_cg_opt).as_euler('xyz') # [T, 3]
    euler_cg_opt_depth = R.from_matrix(R_cg_opt_depth).as_euler('xyz') # [T, 3]
    temp_gripper = np.tile(np.array([0.0]), (euler_cg_opt.shape[0], 1)) # [T, 1]
    # dscho NOTE: depth-based one is much smoother when using accurate T_B_M. accuracy is slightly better.
    # proprioception = np.concatenate([t_cg_opt_depth, euler_cg_opt, temp_gripper], axis=-1) # [T, 7]
    # proprioception = np.concatenate([t_cg_opt_depth, euler_cg_opt_depth, temp_gripper], axis=-1) # [T, 7]
    # proprioception = np.concatenate([t_cg_opt, euler_cg_opt, temp_gripper], axis=-1) # [T, 7]
    # proprioception = np.concatenate([t_cg_closed, euler_cg_opt, temp_gripper], axis=-1) # [T, 7]

    # debugging (depth scaling)
    # proprioception[:, :3] = proprioception[:, :3] * 1.05 # 2.5% scaling (custom calibration)
    # print("@@@@@@@@@@@@@@@@@@@@@@@@@ apply scaling to the proprioception for debugging")
    # time.sleep(2)

    smoothing_sigma = 2.0
    t_cg_opt_depth_smooth = apply_temporal_smoothing(t_cg_opt_depth, smoothing_sigma)
    euler_cg_opt_depth_smooth = apply_temporal_smoothing(euler_cg_opt_depth, smoothing_sigma)
    proprioception = np.concatenate([t_cg_opt_depth_smooth, euler_cg_opt_depth_smooth, temp_gripper], axis=-1) # [T, 7]
    
    
    # assert proprioception.shape[0] == T_mc_transformation.shape[0]
    
    
    T = proprioception.shape[0]
    
    # NOTE: debug for fixed camera
    T_mc_transformation = np.tile(T_mc_transformation[0], (T, 1, 1)) # [T, 4, 4]


    proprioception_se3 = np.tile(np.eye(4), (T, 1, 1)) # [T, 4, 4]
    proprioception_se3[:, :3, :3] = R.from_euler('xyz', proprioception[:, 3:6]).as_matrix() # [T, 4, 4]
    proprioception_se3[:, :3, 3] = proprioception[:, :3] # [T, 4, 4]

    T_bc_transformation = np.einsum('ij,hjk->hik', T_B_M, T_mc_transformation) # [T, 4, 4]
    proprioception_se3_b = np.einsum('hij,hjk->hik', T_bc_transformation, proprioception_se3) # [T, 4, 4]
    
    pos = proprioception_se3_b[:, :3, 3] # [T, 3]
    euler = R.from_matrix(proprioception_se3_b[:, :3, :3]).as_euler('xyz')
    quat = R.from_matrix(proprioception_se3_b[:, :3, :3]).as_quat()
    proprioception_b = np.concatenate([pos, euler, proprioception[:, 6:7]], axis=1) # [T, 7]
    proprioception_b_quat = np.concatenate([pos, quat, proprioception[:, 6:7]], axis=1) # [T, 8]

    # temporary debug
    # proprioception_b = None


    return proprioception_b, proprioception_b_quat, proprioception


# ========== Global cleanup variables and functions ==========
# Global variables to store references for cleanup
_cleanup_env = None
_cleanup_pipeline = None

def cleanup_resources():
    """Cleanup function that will be called on program termination.
    This function is idempotent and can be called multiple times safely."""
    global _cleanup_env, _cleanup_pipeline
    
    try:
        # Check if cleanup has already been done
        if _cleanup_env is None and _cleanup_pipeline is None:
            return  # Already cleaned up
        
        print("\n[Cleanup] Starting resource cleanup...")
        
        # Close environment
        if _cleanup_env is not None:
            try:
                print("[Cleanup] Closing environment...")
                _cleanup_env.close()
                print("[Cleanup] Environment closed.")
                _cleanup_env = None  # Clear reference after cleanup
            except Exception as e:
                print(f"[Cleanup] Error closing environment: {e}")
                _cleanup_env = None  # Clear reference even on error
        
        # Stop camera pipeline
        if _cleanup_pipeline is not None:
            try:
                print("[Cleanup] Stopping RealSense camera pipeline...")
                _cleanup_pipeline.stop()
                print("[Cleanup] RealSense camera pipeline stopped.")
                _cleanup_pipeline = None  # Clear reference after cleanup
            except Exception as e:
                print(f"[Cleanup] Error stopping camera pipeline: {e}")
                _cleanup_pipeline = None  # Clear reference even on error
        
        print("[Cleanup] Resource cleanup completed.")
    except Exception as e:
        print(f"[Cleanup] Error during cleanup: {e}")
        # Try to clear references on error
        try:
            _cleanup_env = None
            _cleanup_pipeline = None
        except:
            pass

def signal_handler(signum, frame):
    """Handle system signals to ensure cleanup."""
    try:
        signal_name = signal.Signals(signum).name
        print(f"\n[Signal Handler] Received signal: {signal_name} ({signum})")
        print("[Signal Handler] Executing cleanup...")
        cleanup_resources()
        
        # For fatal signals, exit cleanly after cleanup
        # For non-fatal signals like SIGINT, we can let Python handle it
        if signum in (signal.SIGABRT, signal.SIGSEGV):
            print("[Signal Handler] Fatal signal received. Exiting after cleanup...")
            sys.exit(1)
        elif signum == signal.SIGTERM:
            print("[Signal Handler] Termination signal received. Exiting after cleanup...")
            sys.exit(0)
        # For SIGINT, allow KeyboardInterrupt to be raised naturally
    except Exception as e:
        print(f"[Signal Handler] Error in signal handler: {e}")
        sys.exit(1)


def main():
    """Main example function."""
    global _cleanup_env, _cleanup_pipeline
    
    # Register cleanup handlers for signals and normal exit
    print("Registering cleanup handlers...")
    atexit.register(cleanup_resources)
    
    # Register signal handlers for various termination signals
    signal.signal(signal.SIGABRT, signal_handler)  # Abort signal (core dump)
    signal.signal(signal.SIGSEGV, signal_handler)  # Segmentation fault
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    signal.signal(signal.SIGINT, signal_handler)   # Keyboard interrupt (Ctrl+C)
    print("Cleanup handlers registered.")
    print("=" * 80)
    
    print("Z1 Gym Environment Example - Non-blocking Step Execution")
    print("=" * 80)
    
    # Create dummy MLP model for thread-torch compatibility testing
    # Note: observation shape is 21, so we need to match that
    dummy_mlp = DummyMLP(input_dim=21, hidden_dim=4096, output_dim=8)
    dummy_mlp.eval()  # Set to evaluation mode
    print(f"Created dummy MLP model: {dummy_mlp}")
    
    # Create the environment with 5Hz control frequency
    control_frequency = 2 # 5
    sequence_length = 16   # Length of future target pose sequences
    step_interval = 1.0 / control_frequency  # Time between steps in seconds
    
    env = EEPoseCtrlJointCmdWrapper(
        has_gripper=True,
        control_frequency=control_frequency,  # 5Hz control frequency
        position_tolerance=0.01,
        orientation_tolerance=0.1,
        joint_speed=0.5,  # Joint speed limit
        sequence_length=sequence_length,  # Length of future sequences
        use_current_joint_pos_when_ik_fails = False,
        
    )
    
    # Store env in global variable for cleanup
    _cleanup_env = env
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print()
    
    try:
        # Reset the environment
        print("Resetting environment...")
        # joint_angle = np.array([1.0, 1.5, -1.0, -0.54, 0.0, 0.0])
        # joint_angle = np.array([-0.8, 2.572, -1.533, -0.609, 1.493, 1.004])
        # joint_angle = np.array([0.0, 1.5, -1.0, -0.54, 0.0, 0.0]) #forward
        joint_angle = np.array([-0.579, 1.581, -0.389, -1.139, 0.484, 1.665]) # lowest table, new demo static bottle
        # joint_angle = None
        obs = env.reset(joint_angle)
        print(f"Initial observation shape: {obs.shape}")
        print(f"Initial joint positions: {obs[:6]}")
        print(f"Initial end-effector position: {obs[12:15]}")
        print(f"Initial end-effector orientation: {obs[15:19]}")
        print(f"Initial gripper position: {obs[19]}")
        print()
    
        
        # Example: Non-blocking step execution with square movement
        print("Example: Non-blocking step execution with square movement")
        print("Using wait=False for non-blocking step execution")
        print("Gripper will alternate between open and close for each edge")
        print("Orientation will change roll: 0°, +90°, 0°, -90°, 0° for each edge")
        print(f"Control frequency: {control_frequency} Hz")
        print(f"Step interval: {step_interval:.3f}s")
        print(f"Joint speed limit: {env.joint_speed} rad/s")
        print(f"Sequence length: {sequence_length}")
        print()
        
        # Get current end-effector position and orientation from observation
        original_position = obs[12:15].copy()  # Store original position
        current_orientation = obs[15:19]  # Current end-effector orientation (quaternion)
        target_gripper = 0 
        
        print(f"Original EE position: {original_position}")
        print(f"Current EE orientation: {current_orientation}")
        print()
        


        # ========== Initialize RealSense Camera ==========
        print("=" * 80)
        print("Initializing RealSense Camera for ChArUco Marker Detection")
        print("=" * 80)
        
        pipeline, align, camera_matrix, dist_coeffs, cam_success = init_realsense_camera(resolution="VGA")
        
        # Store pipeline in global variable for cleanup
        _cleanup_pipeline = pipeline
        
        droid = True
        if not cam_success:
            print("Warning: Camera initialization failed. Continuing without camera...")
            pipeline = None
            _cleanup_pipeline = None  # Also clear global
            align = None
            camera_matrix = None
            dist_coeffs = None

        # dscho debug to specify the position and orientation
        # original_position = np.array([0.41145274, -0.00121779, 0.40713578])
        # current_orientation_wxyz = np.array([0.9998209, -0.0011671, -0.01868073, -0.00280654])  # [w,x,y,z]
        current_orientation_wxyz = np.array([1.0, 0.0, 0.0, 0.0])  # [w,x,y,z]
        current_orientation = np.array([current_orientation_wxyz[1], current_orientation_wxyz[2], 
                                       current_orientation_wxyz[3], current_orientation_wxyz[0]])  # [x,y,z,w]

        # Define square vertices in YZ plane (0.1m x 0.1m square)
        square_size = 0.1  # 0.1m
        square_vertices = [
            original_position + np.array([0.0, 0.0, 0.0]),           # Start point
            original_position + np.array([0.0, square_size, 0.0]),   # Y+0.1
            original_position + np.array([0.0, square_size, square_size]),  # Y+0.1, Z+0.1
            original_position + np.array([0.0, 0.0, square_size]),   # Z+0.1
            original_position + np.array([0.0, 0.0, 0.0])            # Back to start
        ]
        
        # square_vertices = [
        #     original_position + np.array([0.0, 0.0, 0.0]),           # Start point
        #     original_position + np.array([0.0, 0.0, -square_size]),   # Z-0.1
        #     original_position + np.array([0.0, square_size, -square_size]),  # Y+0.1, Z-0.1
        #     original_position + np.array([0.0, square_size, 0.0]),   # Y+0.1
        #     original_position + np.array([0.0, 0.0, 0.0])            # Back to start
        # ]

        # D435 depth-based dift point (leftside bottle cap): array([0.592886  , 0.24988669, 0.1399751 ]), (on the socket): array([ 0.58816114, -0.0290327 , -0.00381193]), 
        # unidepth-based dift point (leftside bottle cap): array([0.62878956, 0.22977131, 0.13334631]), (on the socket): array([0.52487209, 0.02690133, 0.01839266]), 
        # NOTE
        dift_point_base_frame, dift_point_marker_frame, dift_point_camera_frame, point_in_front_of_the_marker_base_frame, dift_points_custom_unprojected_c, dift_points_custom_unprojected_b = get_dift_point_base_frame_data_for_debug(camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, droid=droid) # [T, N, 3]
        # DEBUG_POINT = dift_point_base_frame[0,1] # + np.array([0.0, 0.0, 0.1])
        # DEBUG_POINT = point_in_front_of_the_marker_base_frame

        proprioception_base_frame, proprioception_quat_base_frame, proprioception_camera_frame = get_estimated_hand_pose_base_frame_data_for_debug(droid=droid) # [T, 7]
        # DEBUG_POINT = proprioception_base_frame[0, :3] # + np.array([0.0, 0.0, 0.1])
        
        
        
        # DEBUG_ACTION = np.concatenate([DEBUG_POINT, current_orientation,  np.array([0.0])])
        # print('@@@@@@@@@@@@@@@@@@@@@@@@@ Currently using DEBUG_ACTION :', DEBUG_ACTION)

        
        # DEBUG_ROLL_ANGLES = [0, 0, 0, 0, 0]
        # Define orientation vertices with roll changes: 0°, +90°, 0°, -90°, 0°
        roll_angles = [0, np.pi/2, 0, -np.pi/2, 0]  # 0°, +90°, 0°, -90°, 0°
        
        # roll_angles = DEBUG_ROLL_ANGLES
        orientation_vertices = []
        
        for roll_angle in roll_angles:
            # Convert base orientation (quaternion) to rotation matrix, apply roll, convert back
            base_rot = R.from_quat(current_orientation)
            roll_rot = R.from_euler('x', roll_angle)
            new_rot = base_rot * roll_rot
            orientation_vertices.append(new_rot.as_quat())
        
        print("Square movement plan:")
        for i, vertex in enumerate(square_vertices):
            gripper_state = "Open" if i % 2 == 0 else "Close"
            print(f"  Vertex {i}: {vertex} (Gripper: {gripper_state})")
        print()

        
        total_steps = 190 # 150 #  10 # proprioception_base_frame.shape[0] # 100
        inference_time = 0.15  # Inference time in seconds
        
        print(f"Running non-blocking control with overlapped inference for {total_steps} steps")
        print(f"Inference time: {inference_time}s")
        print(f"Step interval: {step_interval:.3f}s")
        print("Note: Get observation → Execute step immediately → Run inference while robot moves")
        print()
        
        
        # ========== ChArUco Marker Detection (Before Loop) ==========
        # ChArUco board configuration (values from get_charuco_optimize.yaml as reference)
        squaresX = 3                      # 가로 사각형 개수
        squaresY = 5                      # 세로 사각형 개수
        square_size_m = 0.052             # 사각형 크기 (미터)
        marker_size_m = 0.0416            # 마커 크기 (미터)
        aruco_dict_type = cv2.aruco.DICT_4X4_250  # ArUco 딕셔너리 타입
        reference_marker_id = 1           # 기준 마커 ID
        
        print(f"ChArUco board configuration:")
        print(f"  squaresX: {squaresX}, squaresY: {squaresY}")
        print(f"  square_size_m: {square_size_m}, marker_size_m: {marker_size_m}")
        print(f"  aruco_dict_type: DICT_4X4_250, reference_marker_id: {reference_marker_id}")
        
        # 1) ChArUco Board 생성
        board = create_charuco_board(squaresX, squaresY, square_size_m, marker_size_m, aruco_dict_type)
        if board is not None:
            print(f"Created ChArUco Board: {squaresX}x{squaresY} squares")
        else:
            print("Warning: Failed to create ChArUco Board")
        
        T_cm_reference = None  # Camera to reference marker transformation
        T_mc = None  # Reference marker to camera transformation
        
    
        print("\n" + "=" * 80)
        print("Detecting ChArUco Markers (Before Loop)")
        print("=" * 80)
        

        while True:
            # Capture initial RGB image for marker detection
            rgb_init, depth_init, capture_success = capture_realsense_image(pipeline, align)
            
        
            print("Successfully captured initial RGB and depth images for marker detection")
            
            # 2) ChArUco 보드 검출 및 자세 추정
        
            # Convert to grayscale for ChArUco detection
            gray_init = cv2.cvtColor(rgb_init, cv2.COLOR_BGR2GRAY)
            
            # Detect ChArUco board
            board_success, charuco_corners, charuco_ids, marker_corners, marker_ids, rvec_board, tvec_board = detect_charuco_board(
                gray_init, board, camera_matrix, dist_coeffs
            )
            if board_success:
                break
        
        if board_success:
            print(f"Successfully detected ChArUco Board!")
            
            # 3) 보드 자세로부터 각 마커의 개별 자세 계산
            T_cam_marker_dict, T_cam_board = compute_marker_poses_from_board(
                board, rvec_board, tvec_board, marker_ids, marker_corners
            )
            
            # Use reference marker as specified in config
            if reference_marker_id in T_cam_marker_dict:
                T_cm_reference = T_cam_marker_dict[reference_marker_id]
            else:
                # Reference marker not detected, use board pose or first detected marker
                if len(T_cam_marker_dict) > 0:
                    # Fallback to first detected marker
                    reference_marker_id = list(T_cam_marker_dict.keys())[0]
                    T_cm_reference = T_cam_marker_dict[reference_marker_id]
                    print(f"Warning: Reference marker not detected, using first detected marker {reference_marker_id}")
                else:
                    # No markers detected, use board pose
                    print(f"Warning: No markers detected, using board pose as reference")
                    T_cm_reference = T_cam_board
            
            print(f"Using marker {reference_marker_id} as reference")
            print(f"T_cm_reference (Camera to Marker {reference_marker_id}):\n{T_cm_reference}")
            print(f"Detected markers: {len(T_cam_marker_dict)} markers")
            
            # Visualize detected markers
            vis_image = rgb_init.copy()
            if marker_corners is not None and marker_ids is not None:
                cv2.aruco.drawDetectedMarkers(vis_image, marker_corners, marker_ids)
            
            # Draw board pose axes
            cv2.drawFrameAxes(vis_image, camera_matrix, dist_coeffs, rvec_board, tvec_board, square_size_m * 2.0)
            
            # Draw individual marker axes (optional, using computed poses)
            for marker_id, T_cm in T_cam_marker_dict.items():
                rvec, _ = cv2.Rodrigues(T_cm[:3, :3])
                tvec = T_cm[:3, 3]
                color = (0, 255, 0) if marker_id == reference_marker_id else (255, 0, 0)
                cv2.drawFrameAxes(vis_image, camera_matrix, dist_coeffs, rvec, tvec, marker_size_m * 0.5)
                # Draw marker ID text
                cv2.putText(vis_image, f"ID:{marker_id}", tuple((tvec[:2] * 100).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Save visualization
            cv2.imwrite("aruco_detection_init.png", vis_image)
            print("Saved ChArUco detection visualization to: aruco_detection_init.png")
            
            # Calculate T_mc (reference marker to camera) = inv(T_cm_reference)
            T_mc = np.linalg.inv(T_cm_reference)
            print(f"T_mc (Marker {reference_marker_id} to Camera):\n{T_mc}")
        else:
            raise NotImplementedError("Board-based detection is required. Individual marker visualization fallback is not implemented.")
        
        # Track errors for average calculation
        position_errors = []
        orientation_errors = []
        
        # Store images and errors for batch saving
        image_data_list = []  # List of dicts: {'step': int, 'image': np.ndarray, 'pos_error': float, 'orient_error': float}
        
        # Store T_mc_current transformations for analysis
        T_mc_current_list = []  # List of T_mc_current (4x4) matrices
        
        # Store action and actual robot pose for comparison plot
        action_positions = []  # List of action positions [x, y, z]
        action_orientations = []  # List of action quaternions [qx, qy, qz, qw]
        actual_positions = []  # List of actual robot positions [x, y, z]
        actual_orientations = []  # List of actual robot quaternions [qx, qy, qz, qw]
        
        # Sequence tracking variables
        current_action_sequence = None
        current_action_index = 0
        latest_inference_result = None
        
        # Generate initial action sequence from o_0 (before for loop)
        print("Generating initial action sequence from o_0...")
        current_action_sequence = inference_function(
            obs, square_vertices, orientation_vertices, total_steps, 0, dummy_mlp, sequence_length, inference_time
        ) # o_0 -> a_0, a_1, a_2, ...
        current_action_index = 0
        
        # Main execution loop for all steps
        # T_mc_current will be updated after each env.step() completion
        T_mc_current = T_mc  # Initialize with pre-computed T_mc from before loop
        
        # Store initial T_mc for step=0
        if T_mc_current is not None:
            T_mc_current_list.append(T_mc_current.copy())
        
        for step in range(total_steps):
            print(f"\nStep {step}: Processing")
            print("-" * 40)
            # ========== Define camera_data (temporary 3D point in camera frame) ==========
            # This is a placeholder - user should provide actual camera_data
            camera_data = proprioception_camera_frame[step*2, :3]
            
            print(f"\nUsing temporary camera_data (camera frame): {camera_data}")
            print("Note: Replace this with actual camera_data from your detection/estimation")
            
            # ========== Transform camera_data: Camera → Marker → Base (before step) ==========
            # We use T_mc_current which was computed before this iteration
            # For step=0: uses T_mc computed before loop
            # For step>0: uses T_mc_current updated after previous step's env.step() completion
            camera_data_base_frame = None
            
            if cam_success and pipeline is not None and T_mc_current is not None:
                # Transform camera_data using current T_mc (no image capture here)
                camera_data_homo = np.concatenate([camera_data, [1.0]])
                camera_data_marker_homo = T_mc_current @ camera_data_homo
                camera_data_marker = camera_data_marker_homo[:3]
                camera_data_marker_homo = np.concatenate([camera_data_marker, [1.0]])
                camera_data_base_homo = T_B_M @ camera_data_marker_homo
                camera_data_base_frame = camera_data_base_homo[:3]
            else:
                print(f"Step {step}: Camera not available, using fallback")
                if 'proprioception_base_frame' in locals() and step < len(proprioception_base_frame):
                    camera_data_base_frame = proprioception_base_frame[step, :3]
                else:
                    camera_data_base_frame = obs[12:15]
            
            # Select action for current step (use current sequence)
            if current_action_sequence is not None and current_action_index < len(current_action_sequence):
                # Use action from current sequence
                action = current_action_sequence[current_action_index] # a_t
                current_action_index += 1
                print(f"Step {step}: Using action a_{step} from sequence (index {current_action_index-1})")
            else:
                # Fallback: maintain current pose
                current_pos = obs[12:15]
                current_orient = obs[15:19]
                action = np.concatenate([current_pos, current_orient, [target_gripper]])
                print(f"Step {step}: No action available, maintaining current pose")
            
            # Print gripper command
            gripper_cmd = action[7] if len(action) > 7 else 0.0
            print(f"Step {step}: Gripper Command = {gripper_cmd:.3f} ({'Open' if gripper_cmd > 0 else 'Close' if gripper_cmd < 0 else 'Neutral'})")
            
            # Execute step with non-blocking execution
            print(f"Step {step}: Executing action a_{step} with non-blocking...")
            
            start = time.time()
            
            # Use camera_data_base_frame (transformed from camera_data) for DEBUG_ACTION
            if camera_data_base_frame is not None:
                DEBUG_POINT = camera_data_base_frame.copy()
            else:
                # Fallback to original method
                if 'proprioception_base_frame' in locals() and step < len(proprioception_base_frame):
                    DEBUG_POINT = proprioception_base_frame[step, :3]
                else:
                    DEBUG_POINT = obs[12:15]
            

            # DEBUG_POINT = point_in_front_of_the_marker_base_frame
            DEBUG_POINT = dift_point_base_frame[0,0] # + np.array([0.0, 0.0, 0.1])
            # DEBUG_POINT = proprioception_base_frame[0, :3] # + np.array([0.0, 0.0, 0.1])
            # DEBUG_POINT = dift_points_custom_unprojected_b[0,1] + np.array([0.0, 0.0, 0.1]) # result is almost same as g.t
            
            

            # DEBUG_ACTION = np.concatenate([DEBUG_POINT, current_orientation, np.array([0.0])])

            # DEBUG_ACTION = np.concatenate([original_position+np.array([0.0, 0.0, 0.0]), proprioception_quat_base_frame[step*2, 3:7].copy(), np.array([0.0])])
            DEBUG_ACTION = np.concatenate([proprioception_quat_base_frame[step*2, :7].copy(), np.array([0.0])])
            # DEBUG_ACTION = np.concatenate([proprioception_quat_base_frame[step, :3].copy(), current_orientation, np.array([0.0])])

            print(f'@@@@@@@@@@@@@@@@@@@@@@@@@ Step {step}: Using DEBUG_ACTION with camera_data (base frame): {DEBUG_ACTION[:3]}')
            assert DEBUG_ACTION[-1] == 0.0, "assume camera is attached, so the gripper should not be moved"
            env.step(DEBUG_ACTION, wait=False) # a_t
            
            # env.step(action, wait=False) # a_t
            

            print(f"Step {step}: Started non-blocking execution in {time.time() - start:.6f}s")
            
            # Run inference in main process while robot is moving (step > 0 and not last step)
            if step > 0 and step < total_steps - 1:  # Don't run inference for step 0 or last step
                print(f"Step {step}: Running inference with o_{step} while robot moves...")
                inference_start_time = time.time()
                latest_inference_result = inference_function(
                    obs, square_vertices, orientation_vertices, total_steps, step, dummy_mlp, sequence_length, inference_time
                ) # o_t+1 -> a_t+1, a_t+2, a_t+3, ...
                inference_time_actual = time.time() - inference_start_time
                print(f"Step {step}: Inference completed in {inference_time_actual:.3f}s")
                
                # Update action sequence with latest inference result for next step
                if latest_inference_result is not None:
                    current_action_sequence = latest_inference_result
                    current_action_index = 1  # Reset index for new sequence, NOTE: we assume that inference is completed before env.step is finished
                    print(f"Step {step}: Updated action sequence for next steps")
            
            
            print(f"Step {step}: Waiting for o_{step+1}...")
            start = time.time()
            while not env.is_step_complete():
                time.sleep(0.001)  # Small sleep to avoid busy waiting
            print(f"Step {step}: Waited for {time.time() - start:.6f}s for o_{step+1}")

            # Get the result from background execution
            start = time.time()
            result = env.get_step_result()
            if result:
                obs, reward, done, info = result # o_t+1, r_t, etc
                print(f"Step {step}: Received o_{step+1}, time taken: {time.time() - start:.6f}s")
                
                # Store action and actual robot pose for comparison plot
                # Action format: [x, y, z, qx, qy, qz, qw, gripper]
                action_positions.append(DEBUG_ACTION[:3].copy())
                action_orientations.append(DEBUG_ACTION[3:7].copy())
                # action_positions.append(action[:3].copy())
                # action_orientations.append(action[3:7].copy())
                
                # Actual pose format: [x, y, z, qx, qy, qz, qw] from current_ee_pose or obs
                actual_ee_pose = info['current_ee_pose']
                actual_positions.append(actual_ee_pose[:3].copy())
                actual_orientations.append(actual_ee_pose[3:7].copy())
            else:
                print(f"Step {step}: Warning - No result from background step")
                # Use previous observation if no result
                continue  # Skip this step if no result
            
            # Print gripper state
            gripper_state = obs[19] if len(obs) > 19 else 0.0
            print(f"Step {step}: Gripper State = {gripper_state:.3f}")
            
            # ========== Capture RGB and Depth Images and Update T_mc (AFTER env.step is complete) ==========
            # This updated T_mc_current will be used in the NEXT iteration for camera_data transformation
            if cam_success and pipeline is not None:
                rgb_step, depth_step, capture_success = capture_realsense_image(pipeline, align)
                
                # ========== Detect ChArUco Board and Update T_mc_current ==========
                corners_step = None
                ids_step = None
                reference_marker_id_step = reference_marker_id  # Use same reference marker as initial
                
                if capture_success and board is not None:
                    charuco_start = time.time()
                    # ChArUco 보드 검출 및 자세 추정
                    gray_step = cv2.cvtColor(rgb_step, cv2.COLOR_BGR2GRAY)
                    
                    board_success_step, charuco_corners_step, charuco_ids_step, marker_corners_step, marker_ids_step, rvec_board_step, tvec_board_step = detect_charuco_board(
                        gray_step, board, camera_matrix, dist_coeffs
                    )
                    
                    corners_step = marker_corners_step
                    ids_step = marker_ids_step
                    
                    if board_success_step:
                        # 보드 자세로부터 각 마커의 개별 자세 계산
                        T_cam_marker_dict_step, T_cam_board_step = compute_marker_poses_from_board(
                            board, rvec_board_step, tvec_board_step, marker_ids_step, marker_corners_step
                        )
                        
                        # Use reference marker as specified in config
                        if reference_marker_id_step in T_cam_marker_dict_step:
                            T_cm_reference_step = T_cam_marker_dict_step[reference_marker_id_step]
                        else:
                            # Reference marker not detected, use board pose or first detected marker
                            if len(T_cam_marker_dict_step) > 0:
                                # Fallback to first detected marker
                                reference_marker_id_step = list(T_cam_marker_dict_step.keys())[0]
                                T_cm_reference_step = T_cam_marker_dict_step[reference_marker_id_step]
                            else:
                                # No markers detected, use board pose
                                T_cm_reference_step = T_cam_board_step
                        
                        # Update T_mc_current for use in next iteration
                        T_mc_current = np.linalg.inv(T_cm_reference_step)
                        
                        # Store T_mc_current for analysis
                        T_mc_current_list.append(T_mc_current.copy())
                        
                        if step % 10 == 0:  # Print every 10 steps to avoid too much output
                            print(f"Step {step}: Detected ChArUco Board with {len(T_cam_marker_dict_step)} markers")
                            print(f"Step {step}: Using marker {reference_marker_id_step} as reference")
                            print(f"Step {step}: Updated T_mc_current for next iteration")
                    else:
                        print(f"Step {step}: No ChArUco board detected, keeping previous T_mc_current")
                    
                    print(f"Step {step}: ChArUco board detection time: {time.time() - charuco_start:.6f}s")
                    
                elif capture_success:
                    raise NotImplementedError("ChArUco board detection is required. Board-based detection fallback is not implemented.")
            
                # ========== Project camera_data to RGB Image (AFTER step) ==========
                point_2d, visible = project_3d_to_image(camera_data, camera_matrix)
                
                # Draw projected point on RGB image
                vis_rgb = rgb_step.copy()
                
                # Draw detected markers if available
                if capture_success and corners_step is not None:
                    cv2.aruco.drawDetectedMarkers(vis_rgb, corners_step, ids_step)
                    
                    # Draw board pose if available (from board detection)
                    if board is not None and 'rvec_board_step' in locals() and rvec_board_step is not None:
                        # Draw board axes (larger, represents entire board)
                        cv2.drawFrameAxes(vis_rgb, camera_matrix, dist_coeffs, rvec_board_step, tvec_board_step, square_size_m * 2.0)
                        
                        # Draw individual marker axes if computed
                        if 'T_cam_marker_dict_step' in locals():
                            for marker_id, T_cm_step in T_cam_marker_dict_step.items():
                                rvec_step, _ = cv2.Rodrigues(T_cm_step[:3, :3])
                                tvec_step = T_cm_step[:3, 3]
                                color = (0, 255, 0) if marker_id == reference_marker_id_step else (255, 0, 0)
                                cv2.drawFrameAxes(vis_rgb, camera_matrix, dist_coeffs, rvec_step, tvec_step, marker_size_m * 0.5)
                    else:
                        # Fallback: draw individual markers (not implemented)
                        # raise NotImplementedError("Board-based detection is required. Individual marker visualization fallback is not implemented.")
                        # Note: This code is kept for future use but not implemented
                        pass
                
                u, v = int(point_2d[0]), int(point_2d[1])
                if 0 <= u < vis_rgb.shape[1] and 0 <= v < vis_rgb.shape[0]:
                    cv2.circle(vis_rgb, (u, v), 10, (0, 0, 255), -1)  # Red circle
                    cv2.circle(vis_rgb, (u, v), 15, (255, 255, 255), 2)  # White outline
                    cv2.putText(vis_rgb, f"camera_data", (u + 20, v),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Store image for batch saving (errors will be added later)
                image_data_list.append({'step': step, 'image': vis_rgb.copy(), 'pos_error': None, 'orient_error': None})
            
            # Collect errors for average calculation
            position_errors.append(info['position_error'])
            orientation_errors.append(info['orientation_error'])
            
            # Update error info in the corresponding image data entry (find by step number)
            for img_data in image_data_list:
                if img_data['step'] == step:
                    img_data['pos_error'] = info['position_error']
                    img_data['orient_error'] = info['orientation_error']
                    break
            
            if done:
                print(f"Episode finished at step {step}!")
                break
        
        
        
        # Print final status
        final_pos = obs[12:15]
        final_error = info['position_error']
        print(f"\nNon-blocking square movement completed!")
        print(f"Final position: [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}]")
        print(f"Final position error: {final_error:.4f}")
        print(f"Final FSM state: {info['fsm_state']}")
        print(f"Final joint directions: {info['joint_directions']}")
        print(f"Final speeds - Joint: {info['actual_joint_speed']:.3f}, Gripper: {info['gripper_speed']:.3f}")
        print(f"DT ratio (actual_control_time/arm_dt): {info['dt_ratio']}")
        print(f"Final sequence index: {env.sequence_index}")
        print(f"Final sequence length: {len(env.current_sequence) if env.current_sequence is not None else 0}")
        
        # Calculate and print average errors
        if position_errors:
            avg_position_error = np.mean(position_errors)
            avg_orientation_error = np.mean(orientation_errors)
            print(f"\nAverage Errors:")
            print(f"  Average position error: {avg_position_error:.4f} m")
            print(f"  Average orientation error: {avg_orientation_error:.4f} rad")
            print(f"  Total steps executed: {len(position_errors)}")
            print(f" last step error: {position_errors[-1]:.4f} m, {orientation_errors[-1]:.4f} rad")
        
        # Save all images with error text to camera_proj_result folder
        print("\nSaving all images to camera_proj_result folder...")
        output_dir = "camera_proj_result"
        os.makedirs(output_dir, exist_ok=True)
        
        for img_data in image_data_list:
            step_num = img_data['step']
            img = img_data['image'].copy()
            pos_err = img_data['pos_error']
            orient_err = img_data['orient_error']
            
            # Add error text to image
            if pos_err is not None and orient_err is not None:
                # Position error text
                pos_text = f"Pos Error: {pos_err:.4f} m"
                cv2.putText(img, pos_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Orientation error text
                orient_text = f"Orient Error: {orient_err:.4f} rad"
                cv2.putText(img, orient_text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Save image
            filename = os.path.join(output_dir, f"camera_data_projection_step_{step_num:04d}.png")
            cv2.imwrite(filename, img)
        
        print(f"Saved {len(image_data_list)} images to {output_dir}/")
        
        # ========== Analyze T_mc_current variations ==========
        if len(T_mc_current_list) > 0 and T_mc is not None:
            print("\nAnalyzing T_mc_current variations...")
            
            # Convert list to numpy array for easier processing
            T_mc_current_array = np.array(T_mc_current_list)  # [N, 4, 4]
            
            # Calculate relative transformations: T_rel = T_mc_current @ inv(T_mc)
            # This gives the transformation from initial marker frame to current marker frame
            T_mc_inv = np.linalg.inv(T_mc)
            T_relative_list = []
            
            for T_mc_curr in T_mc_current_list:
                T_rel = T_mc_curr @ T_mc_inv  # Relative transformation
                T_relative_list.append(T_rel)
            
            T_relative_array = np.array(T_relative_list)  # [N, 4, 4]
            
            # Extract position and orientation changes
            # Position: translation part of T_rel
            position_changes = T_relative_array[:, :3, 3]  # [N, 3] (x, y, z changes)
            
            # Orientation: euler angles from rotation matrix
            euler_changes_list = []
            for T_rel in T_relative_list:
                R_rel = T_rel[:3, :3]
                euler_rel = R.from_matrix(R_rel).as_euler('xyz')  # roll, pitch, yaw
                euler_changes_list.append(euler_rel)
            euler_changes = np.array(euler_changes_list)  # [N, 3] (roll, pitch, yaw changes)
            
            # Create figure with 2 subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Subplot 1: Position changes (x, y, z)
            steps = np.arange(len(position_changes))
            
            # Normalize position changes to [0, 1] for RGB color mapping
            pos_normalized = position_changes.copy()
            if len(position_changes) > 0:
                pos_min = position_changes.min(axis=0)
                pos_max = position_changes.max(axis=0)
                pos_range = pos_max - pos_min
                pos_range[pos_range == 0] = 1  # Avoid division by zero
                pos_normalized = (position_changes - pos_min) / pos_range
            
            # Create RGB colors for each point (x->R, y->G, z->B)
            colors_pos = pos_normalized  # [N, 3] where each row is [R, G, B]
            
            # Plot position changes with RGB colors
            for i in range(len(position_changes)):
                ax1.plot(steps[i], position_changes[i, 0], 'o', color=colors_pos[i], markersize=8, alpha=0.7, label='x' if i == 0 else '')
                ax1.plot(steps[i], position_changes[i, 1], 's', color=colors_pos[i], markersize=8, alpha=0.7, label='y' if i == 0 else '')
                ax1.plot(steps[i], position_changes[i, 2], '^', color=colors_pos[i], markersize=8, alpha=0.7, label='z' if i == 0 else '')
            
            # Also plot lines for better visualization
            ax1.plot(steps, position_changes[:, 0], 'r-', alpha=0.3, linewidth=1, label='x (line)')
            ax1.plot(steps, position_changes[:, 1], 'g-', alpha=0.3, linewidth=1, label='y (line)')
            ax1.plot(steps, position_changes[:, 2], 'b-', alpha=0.3, linewidth=1, label='z (line)')
            
            ax1.set_xlabel('Step', fontsize=12)
            ax1.set_ylabel('Position Change (m)', fontsize=12)
            ax1.set_title('Position Changes Relative to Initial T_mc\n(Color: RGB = XYZ normalized)', fontsize=14)
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)
            
            # Subplot 2: Orientation changes (roll, pitch, yaw)
            # Normalize euler changes to [0, 1] for RGB color mapping
            euler_normalized = euler_changes.copy()
            if len(euler_changes) > 0:
                euler_min = euler_changes.min(axis=0)
                euler_max = euler_changes.max(axis=0)
                euler_range = euler_max - euler_min
                euler_range[euler_range == 0] = 1  # Avoid division by zero
                euler_normalized = (euler_changes - euler_min) / euler_range
            
            # Create RGB colors for each point (roll->R, pitch->G, yaw->B)
            colors_euler = euler_normalized  # [N, 3] where each row is [R, G, B]
            
            # Plot orientation changes with RGB colors
            for i in range(len(euler_changes)):
                ax2.plot(steps[i], euler_changes[i, 0], 'o', color=colors_euler[i], markersize=8, alpha=0.7, label='roll' if i == 0 else '')
                ax2.plot(steps[i], euler_changes[i, 1], 's', color=colors_euler[i], markersize=8, alpha=0.7, label='pitch' if i == 0 else '')
                ax2.plot(steps[i], euler_changes[i, 2], '^', color=colors_euler[i], markersize=8, alpha=0.7, label='yaw' if i == 0 else '')
            
            # Also plot lines for better visualization
            ax2.plot(steps, euler_changes[:, 0], 'r-', alpha=0.3, linewidth=1, label='roll (line)')
            ax2.plot(steps, euler_changes[:, 1], 'g-', alpha=0.3, linewidth=1, label='pitch (line)')
            ax2.plot(steps, euler_changes[:, 2], 'b-', alpha=0.3, linewidth=1, label='yaw (line)')
            
            ax2.set_xlabel('Step', fontsize=12)
            ax2.set_ylabel('Orientation Change (rad)', fontsize=12)
            ax2.set_title('Orientation Changes Relative to Initial T_mc (Euler XYZ)\n(Color: RGB = Roll/Pitch/Yaw normalized)', fontsize=14)
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure to output_dir
            fig_path = os.path.join(output_dir, "T_mc_variation_analysis.png")
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            print(f"Saved T_mc variation analysis plot to {fig_path}")
            
            # Print statistics
            print(f"\nT_mc Variation Statistics:")
            print(f"  Number of T_mc_current collected: {len(T_mc_current_list)}")
            print(f"  Position changes (m):")
            print(f"    X: min={position_changes[:, 0].min():.6f}, max={position_changes[:, 0].max():.6f}, std={position_changes[:, 0].std():.6f}")
            print(f"    Y: min={position_changes[:, 1].min():.6f}, max={position_changes[:, 1].max():.6f}, std={position_changes[:, 1].std():.6f}")
            print(f"    Z: min={position_changes[:, 2].min():.6f}, max={position_changes[:, 2].max():.6f}, std={position_changes[:, 2].std():.6f}")
            print(f"  Orientation changes (rad):")
            print(f"    Roll:  min={euler_changes[:, 0].min():.6f}, max={euler_changes[:, 0].max():.6f}, std={euler_changes[:, 0].std():.6f}")
            print(f"    Pitch: min={euler_changes[:, 1].min():.6f}, max={euler_changes[:, 1].max():.6f}, std={euler_changes[:, 1].std():.6f}")
            print(f"    Yaw:   min={euler_changes[:, 2].min():.6f}, max={euler_changes[:, 2].max():.6f}, std={euler_changes[:, 2].std():.6f}")
            
            plt.close()
        
        # ========== Plot Action vs Actual Robot Pose Comparison ==========
        if len(action_positions) > 0 and len(actual_positions) > 0:
            print("\nCreating action vs actual robot pose comparison plot...")
            
            # Convert lists to numpy arrays
            action_positions_array = np.array(action_positions)  # [N, 3]
            action_orientations_array = np.array(action_orientations)  # [N, 4]
            actual_positions_array = np.array(actual_positions)  # [N, 3]
            actual_orientations_array = np.array(actual_orientations)  # [N, 4]
            
            # Convert quaternions to euler angles
            action_euler_list = []
            actual_euler_list = []
            for i in range(len(action_orientations_array)):
                # Action quaternion format: [qx, qy, qz, qw]
                action_quat = action_orientations_array[i]
                action_rot = R.from_quat(action_quat)  # scipy uses [x, y, z, w] format
                action_euler = action_rot.as_euler('xyz')
                action_euler_list.append(action_euler)
                
                # Actual quaternion format: [qx, qy, qz, qw]
                actual_quat = actual_orientations_array[i]
                actual_rot = R.from_quat(actual_quat)
                actual_euler = actual_rot.as_euler('xyz')
                actual_euler_list.append(actual_euler)
            
            action_euler_array = np.array(action_euler_list)  # [N, 3] (roll, pitch, yaw)
            actual_euler_array = np.array(actual_euler_list)  # [N, 3] (roll, pitch, yaw)
            
            # Create figure with 2 subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Subplot 1: Position comparison
            steps = np.arange(len(action_positions_array))
            
            # Normalize positions to [0, 1] for RGB color mapping
            pos_min = np.minimum(action_positions_array.min(axis=0), actual_positions_array.min(axis=0))
            pos_max = np.maximum(action_positions_array.max(axis=0), actual_positions_array.max(axis=0))
            pos_range = pos_max - pos_min
            pos_range[pos_range == 0] = 1  # Avoid division by zero
            
            # Normalize action positions
            action_pos_normalized = (action_positions_array - pos_min) / pos_range
            # Normalize actual positions
            actual_pos_normalized = (actual_positions_array - pos_min) / pos_range
            
            # Plot position comparison with RGB colors
            # Action: solid lines
            ax1.plot(steps, action_positions_array[:, 0], 'r-', linewidth=2, label='Action X', alpha=0.8)
            ax1.plot(steps, action_positions_array[:, 1], 'g-', linewidth=2, label='Action Y', alpha=0.8)
            ax1.plot(steps, action_positions_array[:, 2], 'b-', linewidth=2, label='Action Z', alpha=0.8)
            
            # Actual: dashed lines
            ax1.plot(steps, actual_positions_array[:, 0], 'r--', linewidth=2, label='Actual X', alpha=0.8)
            ax1.plot(steps, actual_positions_array[:, 1], 'g--', linewidth=2, label='Actual Y', alpha=0.8)
            ax1.plot(steps, actual_positions_array[:, 2], 'b--', linewidth=2, label='Actual Z', alpha=0.8)
            
            ax1.set_xlabel('Step', fontsize=12)
            ax1.set_ylabel('Position (m)', fontsize=12)
            ax1.set_title('Action vs Actual Robot Position Comparison\n(Solid: Action, Dashed: Actual)', fontsize=14)
            ax1.legend(loc='best', ncol=2)
            ax1.grid(True, alpha=0.3)
            
            # Subplot 2: Orientation comparison (Euler angles)
            # Normalize euler angles to [0, 1] for RGB color mapping
            euler_min = np.minimum(action_euler_array.min(axis=0), actual_euler_array.min(axis=0))
            euler_max = np.maximum(action_euler_array.max(axis=0), actual_euler_array.max(axis=0))
            euler_range = euler_max - euler_min
            euler_range[euler_range == 0] = 1  # Avoid division by zero
            
            # Normalize action euler angles
            action_euler_normalized = (action_euler_array - euler_min) / euler_range
            # Normalize actual euler angles
            actual_euler_normalized = (actual_euler_array - euler_min) / euler_range
            
            # Plot orientation comparison with RGB colors
            # Action: solid lines
            ax2.plot(steps, action_euler_array[:, 0], 'r-', linewidth=2, label='Action Roll', alpha=0.8)
            ax2.plot(steps, action_euler_array[:, 1], 'g-', linewidth=2, label='Action Pitch', alpha=0.8)
            ax2.plot(steps, action_euler_array[:, 2], 'b-', linewidth=2, label='Action Yaw', alpha=0.8)
            
            # Actual: dashed lines
            ax2.plot(steps, actual_euler_array[:, 0], 'r--', linewidth=2, label='Actual Roll', alpha=0.8)
            ax2.plot(steps, actual_euler_array[:, 1], 'g--', linewidth=2, label='Actual Pitch', alpha=0.8)
            ax2.plot(steps, actual_euler_array[:, 2], 'b--', linewidth=2, label='Actual Yaw', alpha=0.8)
            
            ax2.set_xlabel('Step', fontsize=12)
            ax2.set_ylabel('Orientation (rad)', fontsize=12)
            ax2.set_title('Action vs Actual Robot Orientation Comparison (Euler XYZ)\n(Solid: Action, Dashed: Actual)', fontsize=14)
            ax2.legend(loc='best', ncol=2)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure to output_dir
            fig_path = os.path.join(output_dir, "action_vs_actual_pose_comparison.png")
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            print(f"Saved action vs actual pose comparison plot to {fig_path}")
            
            plt.close()
        
        print()
        print("\nNon-blocking square movement with sequence handling completed successfully!")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up resources (this will also be called by signal handlers and atexit)
        cleanup_resources()
        
        # Clear global variables to prevent double cleanup
        _cleanup_env = None
        _cleanup_pipeline = None


if __name__ == "__main__":    
    # Run main example
    main()
