import os
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def extract_reference_landmarks_from_folder(folder_path):
    reference_landmarks = []
    video_filenames = []
    
    for filename in os.listdir(folder_path):
        video_path = os.path.join(folder_path, filename)
        cap = cv2.VideoCapture(video_path)
        
        print(f"Processing video: {filename}")
        video_landmarks = []
        
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)

                if results.pose_landmarks:
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
                    video_landmarks.append(landmarks)
        
        cap.release()

        if video_landmarks:
            video_mean_landmarks = np.mean(video_landmarks, axis=0)
            reference_landmarks.append(video_mean_landmarks)
            video_filenames.append(filename)
            print(f"Average landmarks for {filename}: {video_mean_landmarks}")
    
    return reference_landmarks, video_filenames

def process_frame(img, reference_landmarks, video_filenames):
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if not results.pose_landmarks:
            return img, None, None
        
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
        
        max_similarity = -1
        best_match_idx = -1
        for i, ref_landmarks in enumerate(reference_landmarks):
            similarity = calculate_cosine_similarity(ref_landmarks, landmarks)
            if similarity > max_similarity:
                max_similarity = similarity
                best_match_idx = i
        
        annotated_image = img.copy()
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2))
        
        return annotated_image, max_similarity, video_filenames[best_match_idx] if best_match_idx != -1 else None

def calculate_cosine_similarity(ref_landmarks, curr_landmarks):
    # 모든 랜드마크를 첫 번째 랜드마크(코)를 기준으로 정규화
    ref_landmarks -= ref_landmarks[0]
    curr_landmarks -= curr_landmarks[0]
    
    ref_array = ref_landmarks.flatten()
    curr_array = curr_landmarks.flatten()
    
    dot_product = np.dot(ref_array, curr_array)
    norm_ref = np.linalg.norm(ref_array)
    norm_curr = np.linalg.norm(curr_array)
    
    cosine_similarity = dot_product / (norm_ref * norm_curr)
    
    return cosine_similarity

# 기준 동영상이 저장된 폴더 경로 설정
standard_video_folder_path = r"C:\Users\chaeri\Desktop\lv1_video"

# 기준 동영상에서 랜드마크 평균 추출 및 파일명 저장
reference_landmarks_avg, video_filenames = extract_reference_landmarks_from_folder(standard_video_folder_path)

# 실시간 웹캠 비디오 캡처
cap = cv2.VideoCapture(0)

SIMILARITY_THRESHOLD = 0.8  # 유사도 임계값 조정

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if reference_landmarks_avg:
        processed_frame, similarity, best_match_video = process_frame(frame, reference_landmarks_avg, video_filenames)
        if similarity is not None:
            cv2.putText(processed_frame, f'Similarity: {similarity:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if best_match_video:
                cv2.putText(processed_frame, f'Watching: {best_match_video}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            if similarity >= SIMILARITY_THRESHOLD:
                cv2.putText(processed_frame, 'Pose Matched!', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        processed_frame = frame.copy()
        cv2.putText(processed_frame, 'No reference landmarks', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow('Pose Detection', processed_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
