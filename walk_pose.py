import cv2
import mediapipe as mp
import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    # parser.add_argument('--upper_body_only', action='store_true')  # 0.8.3 or less
    parser.add_argument("--model_complexity",help='model_complexity(0,1(default),2)',type=int,default=1)
    parser.add_argument("--min_detection_confidence",help='min_detection_confidence',type=float,default=0.5)
    parser.add_argument("--min_tracking_confidence",help='min_tracking_confidence',type=float,default=0.5)
    parser.add_argument('--enable_segmentation', action='store_true')
    parser.add_argument("--segmentation_score_th",help='segmentation_score_threshold',type=float,default=0.5)

    parser.add_argument('--use_brect', action='store_true')
    parser.add_argument('--plot_world_landmark', action='store_true')

    args = parser.parse_args()
    return args


def setup_pose_landmarker():
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    # upper_body_only = args.upper_body_only
    model_complexity = args.model_complexity
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence
    enable_segmentation = args.enable_segmentation
    segmentation_score_th = args.segmentation_score_th

    use_brect = args.use_brect
    plot_world_landmark = args.plot_world_landmark

    # カメラ準備 #
    cap = cv2.VideoCapture(cap_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    # モデルロード ##
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        # upper_body_only=upper_body_only,
        model_complexity=model_complexity,
        enable_segmentation=enable_segmentation,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return pose

def detect_landmarks(frame, pose):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    landmarks = []
    if results.pose_landmarks is not None:
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((landmark.x, landmark.y))
    return landmarks


#https://github.com/7tsuno/walkInHome/blob/main/app/hooks/usePose.ts
def main():
    cap = cv2.VideoCapture(0)  # デフォルトのカメラを使用する場合は0を指定
    width, height = 640, 480
    # cap.set(3, width)
    # cap.set(4, height)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)

    pose = setup_pose_landmarker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame,1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks is not None:
            # PoseDetectorのインスタンスを作成
            pose_detector = PoseDetector()

            # サンプルのlandmarksでポーズの更新を行う
            pose_detector.update_pose(results.pose_landmarks)

            # ウォークの状態を確認
            pose_detector.check_walk()

            # ポーズの情報を取得
            pose_info = pose_detector.get_pose_info()
            print(pose_info)
            # for landmark in results.pose_landmarks.landmark:
            #     landmarks.append((landmark.x, landmark.y))


        cv2.imshow("ポーズ検出", frame)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


from typing import List, Optional, Tuple

class PoseDetector:
    def __init__(self):
        self.before_walk_pose_type = "none"
        self.right_foot_up_threshold = None
        self.left_foot_up_threshold = None
        self.walk_count = 0
        self.both_hands_up = False
        self.right_hands_up = False
        self.left_hands_up = False
        self.walk = False
        # 
        self.right_footup = False
        self.left_footup = False
        self.right_footdown = False
        self.left_footdown = False

    def update_pose(self, landmarks):
        if not landmarks:
            return

        right_hands_up = landmarks.landmark[16].y < landmarks.landmark[12].y
        left_hands_up = landmarks.landmark[15].y < landmarks.landmark[11].y
        both_hands_up = right_hands_up and left_hands_up

        self.right_hands_up = right_hands_up
        self.left_hands_up = left_hands_up
        self.both_hands_up = both_hands_up
        #print("xx")
        #
        # right_footup = landmarks.landmark[30].y < self.right_foot_up_threshold
        # left_footup = landmarks.landmark[29].y < self.left_foot_up_threshold
        # self.right_footup = right_footup
        # self.left_footup = left_footup
        #print("xx")

        # if both_hands_up and not self.right_foot_up_threshold and not self.left_foot_up_threshold:
        #     self.right_foot_up_threshold = landmarks.landmark[30].y - (landmarks.landmark[30].y - landmarks.landmark[26].y) / 8
        #     self.left_foot_up_threshold = landmarks.landmark[29].y - (landmarks.landmark[29].y - landmarks.landmark[25].y) / 8
        
        if not self.right_foot_up_threshold and not self.left_foot_up_threshold:
            self.right_foot_up_threshold = landmarks.landmark[30].y - (landmarks.landmark[30].y - landmarks.landmark[26].y) / 8
            self.left_foot_up_threshold = landmarks.landmark[29].y - (landmarks.landmark[29].y - landmarks.landmark[25].y) / 8

        if not self.right_foot_up_threshold or not self.left_foot_up_threshold:
            return
        
        if self.right_foot_up_threshold is not None and self.left_foot_up_threshold is not None:
            right_footup = landmarks.landmark[30].y < self.left_foot_up_threshold #左右逆にすることで正しく検出できるように?
            left_footup = landmarks.landmark[29].y < self.right_foot_up_threshold
            self.right_footup = right_footup
            self.left_footup = left_footup

            right_footdown = landmarks.landmark[30].y > self.left_foot_up_threshold
            left_footdown =  landmarks.landmark[29].y > self.right_foot_up_threshold
            self.right_footdown = right_footdown
            self.left_footdown = left_footdown

            if (self.before_walk_pose_type == "none" or self.before_walk_pose_type == "leftDown") and landmarks.landmark[30].y <= self.left_foot_up_threshold:
                self.before_walk_pose_type = "rightUp"

            if self.before_walk_pose_type == "rightUp" and landmarks.landmark[30].y > self.left_foot_up_threshold:
                self.before_walk_pose_type = "rightDown"
                self.walk_count += 1

            if (self.before_walk_pose_type == "none" or self.before_walk_pose_type == "rightDown") and landmarks.landmark[29].y <= self.right_foot_up_threshold:
                self.before_walk_pose_type = "leftUp"

            if self.before_walk_pose_type == "leftUp" and landmarks.landmark[29].y > self.right_foot_up_threshold:
                self.before_walk_pose_type = "leftDown"
                self.walk_count += 1

        # if (self.before_walk_pose_type == "none" or self.before_walk_pose_type == "leftDown") and landmarks.landmark[30].y < self.right_foot_up_threshold:
        #     self.before_walk_pose_type = "rightUp"

        # if self.before_walk_pose_type == "rightUp" and landmarks.landmark[30].y > self.right_foot_up_threshold:
        #     self.before_walk_pose_type = "rightDown"
        #     self.walk_count += 1

        # if (self.before_walk_pose_type == "none" or self.before_walk_pose_type == "rightDown") and landmarks.landmark[29].y < self.left_foot_up_threshold:
        #     self.before_walk_pose_type = "leftUp"

        # if self.before_walk_pose_type == "leftUp" and landmarks.landmark[29].y > self.left_foot_up_threshold:
        #     self.before_walk_pose_type = "leftDown"
        #     self.walk_count += 1

    def check_walk(self):
        if self.walk_count != 0 and self.walk_count % 1 == 0:
            self.walk = True
        else:
            self.walk = False

    def get_pose_info(self):
        return {
            # "walk_count": self.walk_count,
            # "walk": self.walk,
            # "both_hands_up": self.both_hands_up,
            "before_walk_pose_type":self.before_walk_pose_type,
            "right_footup": self.right_footup,
            "left_footup": self.left_footup,
            "left_footdown":self.left_footdown,
            "right_footdown":self.right_footdown,

        }


if __name__ == "__main__":
    main()
