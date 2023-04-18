import cv2 # opencv套件
import mediapipe as mp # mediapipe套件
import math
import numpy as np
mp_pose=mp.solutions.pose

mp_drawing=mp.solutions.drawing_utils #繪圖方法
mp_drawing_styles=mp.solutions.drawing_styles
Pose_LmsStyle=mp_drawing.DrawingSpec(color=(155,50,0),thickness=10)
Pose_ConStyle=mp_drawing.DrawingSpec(color=(255,255,0),thickness=5)

cap=cv2.VideoCapture("Jumping Jacks.mp4") # 導入影片

def calculateAngle(landmark1, landmark2, landmark3):
    # 獲取所需座標
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3
    # 計算三點之間的夾角
    angle = math.floor(
        math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    )
    if angle < 0: # 負角度 可理解為在第四象限-60度 等於順時針60度 等於負時針300度 
        angle += 360 
    if angle > 180: # 大角換小角  
        angle = 360 - angle
    return angle

# 啟用姿勢偵測
with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,  # 模型複雜度
    smooth_landmarks=True,  # 平滑關鍵點 減少抖動
    enable_segmentation=False,  # 去背用 需另外設置參數
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as pose:  # 相同於 pose = mp_pose.Pose()
    
    while True:
        ret,img=cap.read() #讀取影片 ret是布林函數,意思是是否可以讀取frame img是抓出來的frame
        if not ret:
            print("cannot receive frame") # 如果不行 印出不行 並break
            break
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 將 BGR 轉換成 RGB
        results = pose.process(img2)  # 取得姿勢偵測結果
        # 根據姿勢偵測結果，標記身體節點和骨架
        if results.pose_landmarks:
            # print(results.pose_landmarks)
            # break
            mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                Pose_LmsStyle,
                Pose_ConStyle,
            )
            
            lms = results.pose_landmarks.landmark
            left_elbow = [
                lms[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                lms[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
            ]
            # print (left_elbow) # 查看left_elbow是什麼狀態
            # break

            height, width, _ = img.shape # 接收img的資訊作為引數 長、寬、通道數
            landmarks = [] # 設一個空的list
            for landmark in results.pose_landmarks.landmark:
               
                landmarks.append( # 將關鍵點加進list內.
                    (
                        int(landmark.x * width), # 比例乘上長寬
                        int(landmark.y * height),
                        (landmark.z * 0),
                    )
                )
            l_elbow = calculateAngle( #計算left elbow的角度
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],
            )
            r_elbow = calculateAngle(
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],
            )

            color = (150, 0, 60)
            
            
            cv2.putText(
                img,
                str(l_elbow),
                (
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value][0]-20,#-20
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value][1]-10,#-10
                ),
                # (np.multiply(left_elbow, [img.shape[1], img.shape[0]]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,color,2,cv2.LINE_AA,)

            cv2.putText(
                img,
                str(r_elbow),
                (
                    (landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value][0])-25,#-25
                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value][1]-10,#-10
                ),

                cv2.FONT_HERSHEY_SIMPLEX,
                1,color,2,cv2.LINE_AA,)
            
        cv2.imshow("pose",img) # 顯示影片 視窗名稱為pose
        if cv2.waitKey(10)==ord("q"): # 按Q break
            break
    cap.release()
    cv2.destroyAllWindows()