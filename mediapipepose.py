import cv2
import mediapipe as mp
import numpy as np
from sklearn.preprocessing import Normalizer
from collections import deque
import time
import matplotlib.pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# 큐와 신체 부위 인덱스 정의
queue = deque()
queue_la = deque()
queue_ra = deque()
queue_ll = deque()
queue_rl = deque()
queue_bo = deque()
total_body_queue = [queue_la, queue_ra, queue_ll, queue_rl, queue_bo]

face_num = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
left_arm = {11, 13, 15, 17, 19, 21}
right_arm = {12, 14, 16, 18, 20, 22}
left_leg = {23, 25, 27, 29, 31}
right_leg = {24, 26, 28, 30, 32}
body = {11, 12, 23, 24}

# 임계값과 비디오 설정
prev_time = 0
FPS_S = [13.5, 15, 15, 15, 15, 15]
sim_threshold = 0.15
video_path = ["C:/Users/chaeri/OneDrive/바탕 화면/lv1_video/Lv.2Bast.mp4"]
video_width = 1248
video_height = 702

# 유사도 결과를 저장할 리스트
similarity_history = []

# 코사인 유사도 계산 함수
def findCosineSimilarity_1(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    cosine_similarity = np.sqrt(2 * (1 - (a / (np.sqrt(b) * np.sqrt(c)))))
    similarity_percentage = (1 - cosine_similarity) * 100 
    return similarity_percentage

# Mediapipe 관절 좌표를 배열로 변환
def get_position(lmlist):
    features = [0] * 44
    k = -2
    for j in range(0, 22):
        k = k + 2
        if k >= 44:
            break
        features[k] = lmlist[j][0]
        features[k + 1] = lmlist[j][1]
    return [features]

# 신체 중심 좌표 조정 함수
def adapt_center(center_body):
    center_x = sum([c[0] for c in center_body]) / 4
    center_y = sum([c[1] for c in center_body]) / 4
    return center_x, center_y

# 신체 부위별 위치 계산 함수
def findPosition(img, landmarks):
    lmlist = []
    center_body = []
    h, w, c = img.shape
    for id, lm in enumerate(landmarks):
        if id not in body:
            continue
        cx, cy = int(lm.x * w), int(lm.y * h)
        center_body.append([cx, cy])
    center_x, center_y = adapt_center(center_body)
    for id, lm in enumerate(landmarks):
        if id in face_num:
            continue
        cx, cy = int(lm.x * w), int(lm.y * h)
        cx -= center_x
        cy -= center_y
        lmlist.append([cx, cy])
    return lmlist

# 신체 부위별 유사도 계산
def cosine_each_body(num, web_body, cam_body):
    trans_body = Normalizer().fit([web_body])
    web_body = trans_body.transform([web_body])
    cam_body = trans_body.transform([cam_body])
    total_body_queue[num].append(web_body)
    if len(total_body_queue[num]) > 10:
        total_body_queue[num].popleft()

    min_2 = min(findCosineSimilarity_1(total_body_queue[num][j][0], cam_body[0]) for j in range(len(total_body_queue[num])))
    return round(min_2, 5)

# 신체 부위별 리스트 생성 함수
def cut_body(keyp_list):
    full_body = []
    for i, indices in enumerate([left_arm, right_arm, left_leg, right_leg, body]):
        part = [keyp_list[0][2 * (i - 11) + j] for i in indices for j in (0, 1)]
        full_body.append(part)
    return full_body

# 비디오와 웹캠 캡처 설정
cap = cv2.VideoCapture(video_path[0])
webcap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_height)

with mp_pose.Pose(min_detection_confidence=0.5, model_complexity=0, min_tracking_confidence=0.5) as pose, \
     mp_pose.Pose(min_detection_confidence=0.5, model_complexity=0, min_tracking_confidence=0.5) as pose2:
    
    while cap.isOpened() and webcap.isOpened():
        ret, image = cap.read()
        camsuccess, camimage = webcap.read()
        camimage = cv2.flip(camimage, 1)
        
        if not ret or not camsuccess:
            break

        current_time = time.time() - prev_time
        if current_time > 1. / FPS_S[0]:
            prev_time = time.time()
            results = pose.process(image)
            camresults = pose2.process(camimage)
            if not results.pose_landmarks or not camresults.pose_landmarks:
                continue

            landmarks = results.pose_landmarks.landmark
            lmlist = findPosition(image, landmarks)

            camlandmarks = camresults.pose_landmarks.landmark
            camlmlist = findPosition(camimage, camlandmarks)

            keyp_list = get_position(lmlist)
            full_body = cut_body(keyp_list)

            cam_keyp_list = get_position(camlmlist)
            cam_full_body = cut_body(cam_keyp_list)

            # 신체 부위별 유사도 계산
            body_sim_result = [cosine_each_body(i, full_body[i], cam_full_body[i]) for i in range(5)]
            similarity_percentage = np.mean(body_sim_result)
            similarity_history.append(similarity_percentage)

            cv2.putText(camimage, f'Similarity: {similarity_percentage:.2f}%', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            color = (0, 255, 0) if similarity_percentage > (100 - sim_threshold * 100) else (0, 0, 255)
            cv2.rectangle(camimage, (0, 0), (camimage.shape[1], camimage.shape[0]), color, 10)

        cv2.imshow('Video', image)
        cv2.imshow('Webcam', camimage)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
webcap.release()
cv2.destroyAllWindows()

# 전체 평균 유사도 계산
average_similarity = np.mean(similarity_history)
print(f"전체 평균 유사도: {average_similarity:.2f}%")

# 유사도 변화 그래프 출력
plt.plot(similarity_history, label="Similarity %")
plt.xlabel("Frame")
plt.ylabel("Similarity (%)")
plt.title("Similarity Over Time")
plt.legend()
plt.show()
