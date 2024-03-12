from datetime import datetime

import cv2
from cv2 import destroyAllWindows
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme(style="darkgrid")
# import pyttsx3

# engine = pyttsx3.init()
# voices = engine.getProperty("voices")

# engine.setProperty("voice", voices[73].id)
# engine.setProperty("rate", 120)


UPPER_THRESHOLD = 45
LOWER_THRESHOLD = 30


# def speak(str):
#     engine.say(str)
#     engine.runAndWait()

#参考：https://github.com/kayeungadrian-tam/personal-trainer/tree/v2

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def rescale_frame(frame, percent=50):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

#https://www.pexels.com/ja-jp/search/videos/push%20up/
#cap = cv2.VideoCapture("data/push_up.mp4")
cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (640, 480)


def count_push_up(
    angle_knee: float, stage: str, counter: int, upper_limit: float, lower_limit: float
):
    if angle_knee > upper_limit:
        stage = "up"
    if angle_knee <= lower_limit and stage == "up":
        stage = "down"
        #speak(counter)
        counter += 1
    return stage, counter


def get_angles_from_pose(landmarks: list, mp_pose: mp.solutions):

    shoulder = [
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
    ]

    hip = [
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
    ]

    knee = [
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
    ]

    ankle = [
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
    ]

    wrist = [
        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
    ]

    body_angles = {
        "shoulder": shoulder,
        "hip": hip,
        "knee": knee,
        "ankle": ankle,
        "wrist": wrist,
    }

    return body_angles


def draw_text(
    image: np.ndarray, stage: str, angle_knee: float, counter: int, time_elapsed: float
):

    cv2.putText(
        image,
        f"Stage : {stage} ",
        (30, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        "Count : " + str(counter),
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        "Angle : " + str(angle_knee),
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        str(time_elapsed)[2:10],
        (int(image.shape[1] * 0.8), 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )


def plot_angle(angles: list[list]):

    for angle in angles:
        sns.lineplot(angle)
    plt.show()
    exit()


def main():

    tmp = 0
    array = [0, 0, 0]

    angle_hip = 0
    angle_knee = 0
    counter = 1
    stage = None
    ave, ave2 = [], []

    with mp_pose.Pose(
        static_image_mode=False, min_detection_confidence=0.5, model_complexity=2
    ) as pose:

        start_time = datetime.now()
        while cap.isOpened():
            tmp += 1
            ret, frame = cap.read()
            if not ret:
                break

            # diff = 0
            diff = datetime.now() - start_time

            if frame is not None:
                frame_ = rescale_frame(frame, percent=75)

                # Recolor image to RGB
                image = cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                try:
                    landmarks = results.pose_landmarks.landmark
                    mp_drawing.draw_landmarks(
                        frame_,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    )

                    body_angles = get_angles_from_pose(landmarks, mp_pose)

                    angle_hip = calculate_angle(
                        body_angles.get("shoulder"),
                        body_angles.get("hip"),
                        body_angles.get("knee"),
                    )
                    angle_hip = round(angle_hip, 2)

                    angle_knee = calculate_angle(
                        body_angles.get("shoulder"),
                        body_angles.get("knee"),
                        body_angles.get("wrist"),
                    )
                    angle_knee = round(angle_knee, 2)

                    # tmp += angle_knee

                    array[tmp % 3] = int(angle_knee)

                    difference_array = np.diff(array)
                    # if sum(difference_array) == 0:
                    # print(sum(abs(difference_array)))

                    if diff.seconds <= 5:
                        cv2.putText(
                            frame_,
                            str(5 - diff.seconds),
                            (int(frame_.shape[1] * 0.5), int(frame_.shape[0] * 0.5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            10,
                            (255, 122, 122),
                            20,
                            cv2.LINE_AA,
                        )

                        ave.append(angle_knee)
                        ave2.append(sum(abs(difference_array)))
                        ave_angle = np.mean(ave)
                        print(ave_angle)
                    else:
                        # plot_angle([ave, ave2])
                        stage, counter = count_push_up(
                            angle_knee,
                            stage,
                            counter,
                            ave_angle * 0.75,
                            ave_angle * 0.5,
                        )
                except:
                    pass

                draw_text(frame_, stage, angle_knee, counter, diff)

                cv2.imshow("Mediapipe Feed", frame_)
                # if cv2.waitKey(25) & 0xFF == ord("q"):
                #     break
                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    break

            else:
                break
        cap.release()

    destroyAllWindows()


if __name__ == "__main__":
    main()