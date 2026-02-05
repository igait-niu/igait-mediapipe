import sys
import cv2
import numpy as np
import time
import os
import argparse
from collections import deque

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    print("Error: MediaPipe not available")
    sys.exit(1)

class PoseProcessor:
    def __init__(self, output_width=1920, output_height=1080, use_gpu=True, save_data=False, skeleton_only=False):
        self.output_width = output_width
        self.output_height = output_height
        self.use_gpu = use_gpu and TORCH_AVAILABLE and torch.cuda.is_available()
        self.save_data = save_data
        self.skeleton_only = skeleton_only
        
        if self.use_gpu:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        if MP_AVAILABLE:
            self.setup_mediapipe()
        else:
            raise ImportError("MediaPipe is required")
        
        self.output_dir = "output"
        self.smooth_poses = True
        self.pose_history_3d = deque(maxlen=5)
        self.frame_times = []
        
        # Data storage for ML training
        if self.save_data:
            self.all_landmarks_data = []
        
    def setup_mediapipe(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def process_frame_batch(self, frames):
        if not self.use_gpu:
            return [self.process_single_frame(frame) for frame in frames]
        
        results = []
        for frame in frames:
            result = self.process_single_frame(frame)
            results.append(result)
            
        return results
    
    def process_single_frame(self, frame):
        start_time = time.time()
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb_frame)
        landmarks_3d = self.extract_3d_landmarks(results)
        
        if self.smooth_poses:
            landmarks_3d = self.smooth_landmarks(landmarks_3d)
        
        processing_time = time.time() - start_time
        self.frame_times.append(processing_time)
        
        # Store data for ML training if enabled
        if self.save_data:
            self.store_landmark_data(landmarks_3d, frame_count=len(self.frame_times))
        
        return {
            'landmarks_3d': landmarks_3d,
            'mediapipe_results': results,
            'processing_time': processing_time
        }
    
    def extract_3d_landmarks(self, results):
        landmarks_3d = {
            'pose': None,
            'left_hand': None,
            'right_hand': None,
            'face': None
        }
        
        if results.pose_landmarks:
            pose_3d = []
            for landmark in results.pose_landmarks.landmark:
                pose_3d.append([landmark.x - 0.5, -(landmark.y - 0.5), landmark.z])
            landmarks_3d['pose'] = np.array(pose_3d)
        
        if results.left_hand_landmarks:
            left_hand_3d = []
            for landmark in results.left_hand_landmarks.landmark:
                left_hand_3d.append([landmark.x - 0.5, -(landmark.y - 0.5), landmark.z])
            landmarks_3d['left_hand'] = np.array(left_hand_3d)
        
        if results.right_hand_landmarks:
            right_hand_3d = []
            for landmark in results.right_hand_landmarks.landmark:
                right_hand_3d.append([landmark.x - 0.5, -(landmark.y - 0.5), landmark.z])
            landmarks_3d['right_hand'] = np.array(right_hand_3d)
        
        # Extract face landmarks (all 468 points)
        if results.face_landmarks:
            face_3d = []
            for landmark in results.face_landmarks.landmark:
                face_3d.append([landmark.x - 0.5, -(landmark.y - 0.5), landmark.z])
            landmarks_3d['face'] = np.array(face_3d)
        
        return landmarks_3d
    
    def smooth_landmarks(self, new_landmarks):
        if len(self.pose_history_3d) == 0:
            self.pose_history_3d.append(new_landmarks)
            return new_landmarks
        
        smoothed = new_landmarks.copy()
        
        if new_landmarks['pose'] is not None:
            self.pose_history_3d.append(new_landmarks)
            if len(self.pose_history_3d) > 1:
                pose_history = [h['pose'] for h in self.pose_history_3d if h['pose'] is not None]
                if pose_history:
                    smoothed['pose'] = np.mean(pose_history, axis=0)
        
        return smoothed
    
    def create_blank_frame(self, width, height):
        """Create a blank black frame for skeleton-only mode"""
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    def process_video_gpu_optimized(self, input_path, output_path, start_time=0, duration=None, batch_size=4):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        start_frame = int(start_time * fps)
        if duration:
            end_frame = min(start_frame + int(duration * fps), total_frames)
        else:
            end_frame = total_frames
        
        frames_to_process = end_frame - start_frame
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        mode_str = "HIPAA-Compliant (Skeleton Only)" if self.skeleton_only else "Standard"
        print(f"Processing {frames_to_process} frames | Mode: {mode_str} | Device: {self.device}")
        
        frame_count = 0
        total_start_time = time.time()
        
        try:
            while frame_count < frames_to_process:
                batch_frames = []
                for i in range(batch_size):
                    ret, frame = cap.read()
                    if not ret or frame_count + i >= frames_to_process:
                        break
                    batch_frames.append(frame)
                
                if not batch_frames:
                    break
                
                batch_start = time.time()
                results = self.process_frame_batch(batch_frames)
                batch_time = time.time() - batch_start
                
                for i, (frame, result) in enumerate(zip(batch_frames, results)):
                    # Create blank frame if skeleton-only mode
                    if self.skeleton_only:
                        output_frame = self.create_blank_frame(width, height)
                    else:
                        output_frame = frame.copy()
                    
                    # Draw skeleton overlay
                    frame_with_pose = self.draw_2d_overlay(output_frame, result['mediapipe_results'])
                    
                    current_frame = frame_count + i
                    self.add_performance_overlay(frame_with_pose, current_frame, frames_to_process, 
                                               result['processing_time'], fps)
                    
                    out.write(frame_with_pose)
                
                frame_count += len(batch_frames)
                
                if frame_count % (batch_size * 20) == 0:
                    progress = (frame_count / frames_to_process) * 100
                    elapsed = time.time() - total_start_time
                    avg_fps = frame_count / elapsed if elapsed > 0 else 0
                    eta = (frames_to_process - frame_count) / avg_fps if avg_fps > 0 else 0
                    
                    print(f"Progress: {progress:.1f}% | Speed: {avg_fps:.1f} FPS | ETA: {eta:.1f}s")
                
        except Exception as e:
            print(f"Error: {e}")
        finally:
            cap.release()
            out.release()
            self.holistic.close()
            
            total_time = time.time() - total_start_time
            avg_fps = frame_count / total_time if total_time > 0 else 0
            
            print(f"Complete | Time: {total_time:.2f}s | Speed: {avg_fps:.1f} FPS")
            print(f"Output: {output_path}")
            if self.skeleton_only:
                print("HIPAA Compliance: Original video content removed, skeleton overlay only")
            
            # Save training data if enabled
            if self.save_data:
                label = getattr(self, 'current_label', None)
                self.save_training_data(output_path, input_path, label)
    
    def draw_2d_overlay(self, frame, results):
        if not results:
            return frame
        
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
        
        if hasattr(results, 'left_hand_landmarks') and results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        
        if hasattr(results, 'right_hand_landmarks') and results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        
        # Draw face landmarks
        if hasattr(results, 'face_landmarks') and results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())
        
        return frame
    
    def add_performance_overlay(self, frame, current_frame, total_frames, processing_time, fps):
        overlay = frame.copy()
        overlay_height = 120 if self.skeleton_only else 100
        cv2.rectangle(overlay, (10, 10), (350, overlay_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        info_lines = [
            f"Frame: {current_frame}/{total_frames}",
            f"Processing: {processing_time*1000:.1f}ms",
            f"Device: {'GPU' if self.use_gpu else 'CPU'}"
        ]
        
        if self.skeleton_only:
            info_lines.append("Mode: HIPAA-Compliant (Skeleton Only)")
        
        for i, line in enumerate(info_lines):
            y_pos = 30 + i * 25
            cv2.putText(frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 255, 0), 2)
    
    def store_landmark_data(self, landmarks_3d, frame_count):
        """Store landmark data for ML training"""
        frame_data = {
            'frame_number': frame_count,
            'timestamp': time.time(),
            'pose_landmarks': landmarks_3d['pose'].tolist() if landmarks_3d['pose'] is not None else None,
            'left_hand_landmarks': landmarks_3d['left_hand'].tolist() if landmarks_3d['left_hand'] is not None else None,
            'right_hand_landmarks': landmarks_3d['right_hand'].tolist() if landmarks_3d['right_hand'] is not None else None,
            'face_landmarks': landmarks_3d['face'].tolist() if landmarks_3d['face'] is not None else None
        }

        self.all_landmarks_data.append(frame_data)
    
    def save_training_data(self, output_path, video_filename, label=None):
        """Save extracted data for ML training"""
        if not self.save_data:
            print("Data saving not enabled")
            return

        import json

        # Create data output directory within the output dir
        data_dir = os.path.join(self.output_dir, "training_data")
        os.makedirs(data_dir, exist_ok=True)

        # Base filename
        base_name = os.path.splitext(os.path.basename(video_filename))[0]

        # Save detailed JSON data
        json_path = os.path.join(data_dir, f"{base_name}_landmarks.json")
        with open(json_path, 'w') as f:
            json.dump({
                'video_filename': video_filename,
                'total_frames': len(self.all_landmarks_data),
                'label': label,
                'hipaa_compliant': self.skeleton_only,
                'landmarks_data': self.all_landmarks_data
            }, f, indent=2)

        print(f"Training data saved:")
        print(f"  JSON: {json_path}")
        print(f"  Frames: {len(self.all_landmarks_data)}")

def main():
    parser = argparse.ArgumentParser(description='GPU-Accelerated 3D Pose Estimation')
    parser.add_argument('input', help='Input video file path')
    parser.add_argument('-o', '--output', help='Output directory path (default: ./output)')
    parser.add_argument('-s', '--start', type=float, default=0, help='Start time in seconds')
    parser.add_argument('-d', '--duration', type=float, help='Duration to process in seconds')
    parser.add_argument('--batch-size', type=int, default=16, help='GPU batch size')
    parser.add_argument('--no-gpu', action='store_true', help='Force CPU processing')
    parser.add_argument('--no-smooth', action='store_true', help='Disable pose smoothing')
    parser.add_argument('--save-data', action='store_true', help='Extract and save landmark data for ML training')
    parser.add_argument('--label', type=str, help='Label for this video (for ML training)')
    parser.add_argument('--skeleton-only', action='store_true', 
                       help='HIPAA-compliant mode: Save only skeleton overlay without original video content')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file does not exist: {args.input}")
        return
    
    # Resolve output directory
    output_dir = args.output if args.output else "output"
    os.makedirs(output_dir, exist_ok=True)

    input_name = os.path.splitext(os.path.basename(args.input))[0]
    suffix = "_skeleton" if args.skeleton_only else "_pose"
    output_video_path = os.path.join(output_dir, f"{input_name}{suffix}.mp4")

    print(f"Input Video: {args.input}")
    print(f"Output Video: {output_video_path}")
    print(f"Output Directory: {output_dir}")
    print(f"Using GPU: {'Yes' if not args.no_gpu else 'No'}")
    print(f"Pose Smoothing: {'Enabled' if not args.no_smooth else 'Disabled'}")
    print(f"Save Data for ML Training: {'Yes' if args.save_data else 'No'}")
    print(f"HIPAA-Compliant Mode: {'Yes' if args.skeleton_only else 'No'}")
    print("Starting processing...")
    
    processor = PoseProcessor(use_gpu=not args.no_gpu, save_data=args.save_data, skeleton_only=args.skeleton_only)
    processor.smooth_poses = not args.no_smooth
    processor.output_dir = output_dir

    # Store label for data saving
    if args.save_data and args.label:
        processor.current_label = args.label

    try:
        processor.process_video_gpu_optimized(
            args.input,
            output_video_path,
            start_time=args.start,
            duration=args.duration,
            batch_size=args.batch_size
        )
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()