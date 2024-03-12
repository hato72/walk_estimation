import copy
import argparse

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0) #カメラ
    parser.add_argument("--width", help='cap width', type=int, default=900) #画面横幅
    parser.add_argument("--height", help='cap height', type=int, default=400) #画面縦幅

    parser.add_argument("--max_num_faces", type=int, default=1)
    #parser.add_argument("--model_selection", type=int, default=0)
    parser.add_argument('--refine_landmarks', action='store_true') #landmarkを細緻化するかどうか　指定されるとTrueに
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    parser.add_argument('--use_brect', action='store_true') #外接矩形 (brect) を使用するかどうか

    args = parser.parse_args()
    return args


def main():
    ### 引数解析
    args = get_args()

    device = args.device
    width = args.width
    height = args.height
    max_num_faces = args.max_num_faces
    refine_landmarks = args.refine_landmarks
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence
    use_brect = args.use_brect

    #カメラ
    capture = cv.VideoCapture(device)
    capture.set(cv.CAP_PROP_FRAME_WIDTH,width)
    capture.set(cv.CAP_PROP_FRAME_HEIGHT,height)

    #モデル
    mp_facemesh = mp.solutions.face_mesh
    facemesh = mp_facemesh.FaceMesh(
            max_num_faces = max_num_faces,
            refine_landmarks = refine_landmarks,
            min_detection_confidence = min_detection_confidence,
            min_tracking_confidence = min_tracking_confidence
    )

    ear_threshold_close = 1.4
    ear_threshold_open = 1.2
    eye_open = True
    #blink_count = 0
    #print(blink_count)

    while True:
        ret,image = capture.read()
        if not ret:
            break

        image = cv.flip(image,1)
        debug_image = copy.deepcopy(image)

        #検出
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = facemesh.process(image)

        #描画
        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                #外接矩形の計算
                brect = calc_bounding_rect(debug_image, face_landmarks)
                # left_eye_ratio = calculate_eye_ratio(face_landmarks, [33, 246, 161, 160, 159, 158, 157, 173])
                # right_eye_ratio = calculate_eye_ratio(face_landmarks, [263, 466, 388, 387, 386, 385, 384, 398])
                
                # if(left_eye_ratio+right_eye_ratio) < 0.50:
                #     cv.putText(image,"Close eye",(10,430),cv.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1,cv.LINE_AA)
                # elif (left_eye_ratio+right_eye_ratio) < 1:
                #     cv.putText(image,"middle eye",(10,430),cv.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1,cv.LINE_AA)
                # elif (left_eye_ratio+right_eye_ratio) >= 1:
                #     cv.putText(image,"open eye",(10,430),cv.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1,cv.LINE_AA)
                
                # # 目が閉じていると判断
                # if left_eye_ratio < ear_threshold_close or right_eye_ratio < ear_threshold_close:
                #     eye_open = False
                # # 目が開いていると判断
                # elif left_eye_ratio > ear_threshold_open or right_eye_ratio > ear_threshold_open:
                #     if not eye_open:
                #         blink_count += 1  # 瞬きの回数を増やす
                #         print(blink_count)
                #     eye_open = True

                #目の外接円の計算
                left_eye,right_eye = None,None
                if refine_landmarks:
                    left_eye,right_eye = calc_iris_min_enc_losingCircle(
                        debug_image,
                        face_landmarks
                    )


                #描画
                print_image = draw_landmarks(
                        debug_image,
                        face_landmarks,
                        refine_landmarks,
                        left_eye,
                        right_eye
                )
                printing_image = draw_bounding_rect(use_brect,print_image,brect)

                left_eye_ratio = calculate_eye_ratio(face_landmarks, [33, 246, 161, 160, 159, 158, 157, 173])
                right_eye_ratio = calculate_eye_ratio(face_landmarks, [263, 466, 388, 387, 386, 385, 384, 398])
                        
                if right_eye_ratio < 1.5 and left_eye_ratio < 1.5:
                    #blink_count += 1
                    #cv.putText(printing_image,"middle eye",(10,430),cv.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1,cv.LINE_AA)
                    print("middle eye right")
                # if (right_eye_ratio) >= 1.6:
                #     #cv.putText(printing_image,"open eye",(10,430),cv.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1,cv.LINE_AA)
                #     print("open eye right")

                # if(left_eye_ratio) < 0.50:
                #     cv.putText(printing_image,"Close eye",(10,430),cv.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1,cv.LINE_AA)
                #     print("close eye left")
                # elif (left_eye_ratio) < 1:
                #     cv.putText(printing_image,"middle eye",(10,430),cv.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1,cv.LINE_AA)
                #     print("middle eye left ")
                # elif (left_eye_ratio) > 1.5:
                #     cv.putText(printing_image,"open eye",(10,430),cv.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1,cv.LINE_AA)
                #     print("open eye left")

        cv.putText(printing_image)

        key = cv.waitKey(1)
        if key == 27:
            break

        cv.imshow("mediapipe facemesh demo",printing_image)
    capture.release()
    cv.destroyAllWindows()


def calculate_eye_ratio(face_landmarks, eye_landmarks):
    # 眼のアスペクト比を計算する関数
    eye_points = np.array([[face_landmarks.landmark[i].x, face_landmarks.landmark[i].y] for i in eye_landmarks])
    # EAR計算
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    eye_ratio = (A + B) / (2.0 * C)
    return eye_ratio


#外接矩形の計算
def calc_bounding_rect(image,landmarks):
    image_height,image_width = image.shape[0],image.shape[1]
    landmark_array = np.empty((0,2),int)

    for _,landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0) #np.appendによって既存のlandmark_arrayは変更されず、新しいlandmark_arrayが作られる
    
    x,y,width,height = cv.boundingRect(landmark_array)
    return [x,y,x+width,y+height]

#目の座標の計算と外接円
def calc_iris_min_enc_losingCircle(image,landmarks):
    image_height,image_width = image.shape[0],image.shape[1]
    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append((landmark_x,landmark_y))

    left_eye_points = [
        landmark_point[468],
        landmark_point[469],
        landmark_point[470],
        landmark_point[471],
        landmark_point[472],
    ]
    right_eye_points = [
        landmark_point[473],
        landmark_point[474],
        landmark_point[475],
        landmark_point[476],
        landmark_point[477],
    ]

    left_eye = calc_min_losingCircle(left_eye_points)
    right_eye = calc_min_losingCircle(right_eye_points)
    return left_eye,right_eye

#目の座標から外接円の中心座標と半径を計算
def calc_min_losingCircle(landmark_list): 
    center,radius = cv.minEnclosingCircle(np.array(landmark_list)) #引数の座標をすべて含む最小の円の中心座標と半径
    center = (int(center[0]),int(center[1]))
    radius = int(radius)
    return center,radius


def draw_landmarks(image,face_landmarks,refine_landmarks,left_eye,right_eye):
    image_height,image_width = image.shape[0],image.shape[1]

    landmark_point = []

    for _,landmark in enumerate(face_landmarks.landmark):
        if landmark.visibility < 0 or landmark.presence < 0:
            continue

        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append((landmark_x,landmark_y))

        cv.circle(image,(landmark_x,landmark_y),1,(0,255,0),1)
    
    if len(landmark_point) > 0:
        #https://github.com/tensorflow/tfjs-models/blob/master/face-landmarks-detection/mesh_map.jpg

        #左眉毛
        cv.line(image, landmark_point[55], landmark_point[65], (0, 255, 0), 2) #目頭側
        cv.line(image, landmark_point[65], landmark_point[52], (0, 255, 0), 2)
        cv.line(image, landmark_point[52], landmark_point[53], (0, 255, 0), 2)
        cv.line(image, landmark_point[53], landmark_point[46], (0, 255, 0), 2) #目じり側

        # 右眉毛
        cv.line(image, landmark_point[285], landmark_point[295], (0, 255, 0),2) #目頭側
        cv.line(image, landmark_point[295], landmark_point[282], (0, 255, 0),2)
        cv.line(image, landmark_point[282], landmark_point[283], (0, 255, 0),2)
        cv.line(image, landmark_point[283], landmark_point[276], (0, 255, 0),2) #目じり側

        #左目　目頭から上側にいき、目尻まで
        cv.line(image, landmark_point[133], landmark_point[173], (0, 255, 0),2) 
        cv.line(image, landmark_point[173], landmark_point[157], (0, 255, 0),2)
        cv.line(image, landmark_point[157], landmark_point[158], (0, 255, 0),2)
        cv.line(image, landmark_point[158], landmark_point[159], (0, 255, 0),2)
        cv.line(image, landmark_point[159], landmark_point[160], (0, 255, 0),2)
        cv.line(image, landmark_point[160], landmark_point[161], (0, 255, 0),2)
        cv.line(image, landmark_point[161], landmark_point[246], (0, 255, 0),2)

        #目尻から下側へいき目頭まで
        cv.line(image, landmark_point[246], landmark_point[163], (0, 255, 0),2)
        cv.line(image, landmark_point[163], landmark_point[144], (0, 255, 0),2)
        cv.line(image, landmark_point[144], landmark_point[145], (0, 255, 0),2)
        cv.line(image, landmark_point[145], landmark_point[153], (0, 255, 0),2)
        cv.line(image, landmark_point[153], landmark_point[154], (0, 255, 0),2)
        cv.line(image, landmark_point[154], landmark_point[155], (0, 255, 0),2)
        cv.line(image, landmark_point[155], landmark_point[133], (0, 255, 0),2)

        #右目
        cv.line(image, landmark_point[362], landmark_point[398], (0, 255, 0),2)
        cv.line(image, landmark_point[398], landmark_point[384], (0, 255, 0),2)
        cv.line(image, landmark_point[384], landmark_point[385], (0, 255, 0),2)
        cv.line(image, landmark_point[385], landmark_point[386], (0, 255, 0),2)
        cv.line(image, landmark_point[386], landmark_point[387], (0, 255, 0),2)
        cv.line(image, landmark_point[387], landmark_point[388], (0, 255, 0),2)
        cv.line(image, landmark_point[388], landmark_point[466], (0, 255, 0),2)

        cv.line(image, landmark_point[466], landmark_point[390], (0, 255, 0),2)
        cv.line(image, landmark_point[390], landmark_point[373], (0, 255, 0),2)
        cv.line(image, landmark_point[373], landmark_point[374], (0, 255, 0),2)
        cv.line(image, landmark_point[374], landmark_point[380], (0, 255, 0),2)
        cv.line(image, landmark_point[380], landmark_point[381], (0, 255, 0),2)
        cv.line(image, landmark_point[381], landmark_point[382], (0, 255, 0),2)
        cv.line(image, landmark_point[382], landmark_point[362], (0, 255, 0),2)

        # 口 向かって左側の唇の内側から上側へいき、下側からまわって同じところへ
        cv.line(image, landmark_point[308], landmark_point[415], (0, 255, 0),2)
        cv.line(image, landmark_point[415], landmark_point[310], (0, 255, 0),2)
        cv.line(image, landmark_point[310], landmark_point[311], (0, 255, 0),2)
        cv.line(image, landmark_point[311], landmark_point[312], (0, 255, 0),2)
        cv.line(image, landmark_point[312], landmark_point[13], (0, 255, 0), 2)
        cv.line(image, landmark_point[13], landmark_point[82], (0, 255, 0), 2)
        cv.line(image, landmark_point[82], landmark_point[81], (0, 255, 0), 2)
        cv.line(image, landmark_point[81], landmark_point[80], (0, 255, 0), 2)
        cv.line(image, landmark_point[80], landmark_point[191], (0, 255, 0), 2)
        cv.line(image, landmark_point[191], landmark_point[78], (0, 255, 0), 2)

        cv.line(image, landmark_point[78], landmark_point[95], (0, 255, 0), 2)
        cv.line(image, landmark_point[95], landmark_point[88], (0, 255, 0), 2)
        cv.line(image, landmark_point[88], landmark_point[178], (0, 255, 0), 2)
        cv.line(image, landmark_point[178], landmark_point[87], (0, 255, 0), 2)
        cv.line(image, landmark_point[87], landmark_point[14], (0, 255, 0), 2)
        cv.line(image, landmark_point[14], landmark_point[317], (0, 255, 0), 2)
        cv.line(image, landmark_point[317], landmark_point[402], (0, 255, 0),2)
        cv.line(image, landmark_point[402], landmark_point[318], (0, 255, 0),2)
        cv.line(image, landmark_point[318], landmark_point[324], (0, 255, 0),2)
        cv.line(image, landmark_point[324], landmark_point[308], (0, 255, 0),2)

        if refine_landmarks:
            cv.circle(image, left_eye[0], left_eye[1], (0, 255, 0), 2)
            cv.circle(image, right_eye[0], right_eye[1], (0, 255, 0), 2)

            # 左目：中心
            cv.circle(image, landmark_point[468], 2, (0, 0, 255), -1)
            # 左目：目頭側
            cv.circle(image, landmark_point[469], 2, (0, 0, 255), -1)
            # 左目：上側
            cv.circle(image, landmark_point[470], 2, (0, 0, 255), -1)
            # 左目：目尻側
            cv.circle(image, landmark_point[471], 2, (0, 0, 255), -1)
            # 左目：下側
            cv.circle(image, landmark_point[472], 2, (0, 0, 255), -1)
            # 右目：中心
            cv.circle(image, landmark_point[473], 2, (0, 0, 255), -1)
            # 右目：目尻側
            cv.circle(image, landmark_point[474], 2, (0, 0, 255), -1)
            # 右目：上側
            cv.circle(image, landmark_point[475], 2, (0, 0, 255), -1)
            # 右目：目頭側
            cv.circle(image, landmark_point[476], 2, (0, 0, 255), -1)
            # 右目：下側
            cv.circle(image, landmark_point[477], 2, (0, 0, 255), -1)
    
    return image

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # 外接矩形
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 255, 0), 2)

    return image


if __name__ == '__main__':
    main()