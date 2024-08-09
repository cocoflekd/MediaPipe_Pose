import cv2
import mediapipe as mp
import numpy as np
import math
import os

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class PoseKeyPoint:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_z(self):
        return self.z

def dot_product(A, B):
    return sum(a * b for a, b in zip(A, B))

def cosine_similarity(pose_world_landmarks1, pose_world_landmarks2):
    A = [pose_world_landmarks1.get_x(), pose_world_landmarks1.get_y(), pose_world_landmarks1.get_z()]
    B = [pose_world_landmarks2.get_x(), pose_world_landmarks2.get_y(), pose_world_landmarks2.get_z()]

    return dot_product(A, B) / (math.sqrt(dot_product(A, A)) * math.sqrt(dot_product(B, B)))

def normalize_landmarks(landmarks):
    xs = [lm.get_x() for lm in landmarks]
    ys = [lm.get_y() for lm in landmarks]
    zs = [lm.get_z() for lm in landmarks]
    
    center_x = sum(xs) / len(xs)
    center_y = sum(ys) / len(ys)
    center_z = sum(zs) / len(zs)
    
    max_dist = max([math.sqrt((lm.get_x() - center_x) ** 2 + (lm.get_y() - center_y) ** 2 + (lm.get_z() - center_z) ** 2) for lm in landmarks])
    
    normalized_landmarks = [PoseKeyPoint((lm.get_x() - center_x) / max_dist, (lm.get_y() - center_y) / max_dist, (lm.get_z() - center_z) / max_dist) for lm in landmarks]
    
    return normalized_landmarks

def calculate_angle(a, b, c):
    a = np.array([a.get_x(), a.get_y()]) 
    b = np.array([b.get_x(), b.get_y()]) 
    c = np.array([c.get_x(), c.get_y()]) 
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360.0 - angle
        
    return angle

def calculate_pose_similarity(landmarks1, landmarks2):
    joints = [
        (11, 13, 15),  # Left Arm
        (12, 14, 16),  # Right Arm
        (23, 25, 27),  # Left Leg
        (24, 26, 28)   # Right Leg
    ]
    
    angles1 = [calculate_angle(landmarks1[j[0]], landmarks1[j[1]], landmarks1[j[2]]) for j in joints]
    angles2 = [calculate_angle(landmarks2[j[0]], landmarks2[j[1]], landmarks2[j[2]]) for j in joints]
    
    angle_similarities = [cosine_similarity(PoseKeyPoint(a, 0, 0), PoseKeyPoint(b, 0, 0)) for a, b in zip(angles1, angles2)]
    return sum(angle_similarities) / len(angle_similarities)

def process_frame(img, reference_landmarks):
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        results = pose.process(img)
        annotated_image = img.copy()
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
            )
            
            current_landmarks = [
                PoseKeyPoint(lm.x, lm.y, lm.z)
                for lm in results.pose_landmarks.landmark
            ]
            
            if reference_landmarks:
                normalized_landmarks1 = normalize_landmarks(current_landmarks)
                normalized_landmarks2 = normalize_landmarks(reference_landmarks)
                
                similarity = calculate_pose_similarity(normalized_landmarks1, normalized_landmarks2)
                cv2.putText(annotated_image, f'Similarity: {similarity:.2f}', (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                return annotated_image, similarity
        
        return annotated_image, None

cap = cv2.VideoCapture(0)
image_paths = ["C:\\Users\\chaeri\\Desktop\\spuat1.png", "C:\\Users\\chaeri\\Desktop\\spuat2.png"]
reference_landmarks_list = []

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    for path in image_paths:
        if not os.path.exists(path):
            print(f"Warning: Image path {path} does not exist.")
            continue
        image = cv2.imread(path)
        if image is None:
            print(f"Warning: Failed to load image from {path}.")
            continue
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            reference_landmarks = [
                PoseKeyPoint(lm.x, lm.y, lm.z)
                for lm in results.pose_landmarks.landmark
            ]
            reference_landmarks_list.append(reference_landmarks)

current_stage = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # print(len(reference_landmarks_list))
    if current_stage < len(reference_landmarks_list):
        processed_frame, similarity = process_frame(frame, reference_landmarks_list[current_stage])
        print(similarity)
        if similarity is not None and similarity > 0.8:
            current_stage += 1
            if current_stage >= len(reference_landmarks_list):
                cv2.putText(processed_frame, f'{current_stage} cherry', (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        processed_frame, _ = process_frame(frame, None)
        cv2.putText(processed_frame, f'{current_stage} num', (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('Pose Detection', processed_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
