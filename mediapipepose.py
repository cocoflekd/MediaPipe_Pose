import os
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#포즈 전환 감지
def detect_pose_transitions(video_path, threshold=0.1):
    cap = cv2.VideoCapture(video_path)
    transition_frames = []
    previous_landmarks = None
    
    with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
                
                #다중 동작 감지
                #이전 동작과 현재 동작 간의 관절 좌표 간 차이 계산
                if previous_landmarks is not None:
                    diff = np.linalg.norm(landmarks - previous_landmarks, axis=1).mean()
                    
                    #차이가 임계값보다 크면 다른 동작으로 인식
                    if diff > threshold:
                        transition_frames.append(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
                
                previous_landmarks = landmarks
            else:
                print("관절 좌표 감지 실패")

        if not transition_frames:
            print("동작 전환 감지 실패")
        
    cap.release()
    return transition_frames

#전환된 부분(다른 동작으로 인식된 부분)의 관절 좌표 추출
def extract_reference_landmarks_from_video_with_transitions(video_path, transition_frames):
    cap = cv2.VideoCapture(video_path)
    #참조 좌표 저장 리스트
    reference_landmarks = []
    #구간별 좌표 저장 리스트
    segment_landmarks = []
    current_segment = 0
    frame_idx = 0
    
    with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
                segment_landmarks.append(landmarks)
            
            if frame_idx in transition_frames or not ret:
                #구간별 좌표값 평균 계산 -> 참조 좌표로 저장
                if segment_landmarks:
                    segment_mean_landmarks = np.mean(segment_landmarks, axis=0)
                    reference_landmarks.append(segment_mean_landmarks)
                    segment_landmarks = []
                    print(f"Segment {current_segment} average landmarks calculated.")
                current_segment += 1
            
            frame_idx += 1

        if not reference_landmarks:
            print("No reference landmarks could be extracted.")
        
    cap.release()
    
    return reference_landmarks

#코사인 유사도 계산 함수
def calculate_cosine_similarity(ref_landmarks, curr_landmarks):
    # 기준점(코)를 기준으로 정규화
    ref_landmarks -= ref_landmarks[0]
    curr_landmarks -= curr_landmarks[0]
    
    # 랜드마크에 대한 기본 가중치 설정
    weights = np.ones((33, 3))
    important_landmarks = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26]
    weights[important_landmarks] = [8.0, 8.0, 8.0]  

    ref_array = (ref_landmarks * weights).flatten()
    curr_array = (curr_landmarks * weights).flatten()
    
    dot_product = np.dot(ref_array, curr_array)
    norm_ref = np.linalg.norm(ref_array)
    norm_curr = np.linalg.norm(curr_array)
    
    # 코사인 유사도 계산
    cosine_similarity = dot_product / (norm_ref * norm_curr)
    
    return cosine_similarity

def process_frame(img, reference_landmarks):
    with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3) as pose:
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if not results.pose_landmarks:
            return img, None, None
        
        # 현재 프레임의 관절 좌표
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
        
        #최대 유사도 설정 및 치적 일치 구간 인덱스 설정
        max_similarity = -1
        best_match_idx = -1

        # 유사도 계산
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
        
         # 처리된 이미지, 유사도, 최적 일치 구간 반환
        return annotated_image, round(max_similarity, 3), best_match_idx 


#영상 경로
video_folder_path = r"C:\Users\chaeri\Desktop\lv1_video"
video_files = [f for f in os.listdir(video_folder_path) if f.endswith(('.mp4', '.avi', '.mov'))]

#참조 좌표 저장 리스트
reference_landmarks_avg = []

for video_file in video_files:
    video_path = os.path.join(video_folder_path, video_file)
    
    print(f"Processing video: {video_file}")
    
    # 동영상에서 동작 전환 감지
    transition_frames = detect_pose_transitions(video_path)
    # 전환된 구간의 랜드마크 추출 및 평균 계산
    reference_landmarks_avg = extract_reference_landmarks_from_video_with_transitions(video_path, transition_frames)

    if reference_landmarks_avg:
        print(f"Extracted {len(reference_landmarks_avg)} reference landmarks sets from {video_file}.")
    else:
        print(f"No reference landmarks extracted from {video_file}.")

cap = cv2.VideoCapture(0)

#유사도 임계값 설정
SIMILARITY_THRESHOLD = 0.85

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if reference_landmarks_avg:
        processed_frame, similarity, best_match_idx = process_frame(frame, reference_landmarks_avg)
        if similarity is not None:
            cv2.putText(processed_frame, f'Similarity: {similarity:.3f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 최적 일치 구간 출력
            if best_match_idx is not None:
                cv2.putText(processed_frame, f'Matched Segment: {best_match_idx+1}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # 유사도가 임계값 이상인 경우
            if similarity >= SIMILARITY_THRESHOLD:
                cv2.putText(processed_frame, 'Perfect!', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        processed_frame = frame.copy()
        cv2.putText(processed_frame, 'No reference landmarks', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow('Pose Detection', processed_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()