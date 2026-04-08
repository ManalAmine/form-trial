import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils  # all drawing utilities
mp_pose = mp.solutions.pose  # all pose model(face detection,iris...we decided pose)



def calculateangle(a, b, c):
    #easier to transform to numpy array
    a = np.array(a)  # first point (11)
    b = np.array(b)  # mid point(13)
    c = np.array(c)  # end point(15)
    #calulate radians between the three points
    radians=np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])  #c1-b1(y of c - y of b, x of c - x of b(c0-b0)) - (y of a - y of b, x of a - x of b)
    angle = np.abs(radians*180.0/np.pi)  # convert to degree 
    #between 0 and 180
    if angle > 180.0:
        angle = 360 - angle
    return angle



# video feed
cap = cv2.VideoCapture(0)  # setting video(maybe web cam...depends on the device)

counter=0
stage=None #down or up

# simple thresholds (easy to tune)
BOTTOM_ANGLE = 155
TOP_ANGLE = 45
DRIFT_THRESHOLD = 0.06
SWING_THRESHOLD = 0.05
FAST_REP_FRAMES = 18  # ~0.6s at 30 FPS
CONFIRM_FRAMES = 3
WARNING_HOLD_FRAMES = 20
LIVE_WARNING_HOLD_FRAMES = 6
REP_TIMEOUT_FRAMES = 150  # safety reset if rep is interrupted
LIVE_CHECK_START_ANGLE = 147  # start live warnings only after actual curl starts

# rep state tracking (per rep, not per frame warnings)
phase = "waiting_down"
down_confirm = 0
top_confirm = 0
rep_frames = 0
rep_min_angle = 180
rep_max_angle = 0
left_elbow_start_x = 0.0
right_elbow_start_x = 0.0
torso_offset_start = 0.0
max_left_elbow_drift = 0.0
max_right_elbow_drift = 0.0
max_body_swing = 0.0
live_warning_text = ""
live_warning_hold = 0
final_warning_text = ""
final_warning_hold = 0
last_rep_label = "-"
rep_motion_started = False

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:  # access pose estimation with how much confidence we want(accuracy)
    while cap.isOpened():  # while cam is open
        ret, frame = cap.read()  # read the feed of my web cam and status in frame
        if not ret or frame is None:
            continue

        frame = cv2.flip(frame, 1)  # mirror camera horizontally

        # detecting stuff and render
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # recolor image to RGB because mediapipe uses RGB
        image.flags.writeable = False

        # make detection
        results = pose.process(image)  # process the image and store it in results

        

        image.flags.writeable = True  # to improve performance
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # converting to BGR for opencv
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark 
            #grabbing the landmarks we only grab x to begin with then we did y
            #shoulder map to a elbow to b and wrist to c
            left_shoulder =[landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow =[landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist =[landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            left_angle = calculateangle(left_shoulder, left_elbow, left_wrist)

            right_shoulder =[landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow =[landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist =[landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_angle = calculateangle(right_shoulder, right_elbow, right_wrist)

            #visulize
            cv2.putText(image, str(int(left_angle)), 
                tuple(np.multiply(left_elbow, [640, 480]).astype(int)),  # to put the angle text at the elbow
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)  # font and color of the text
            cv2.putText(image, str(int(right_angle)), 
                tuple(np.multiply(right_elbow, [640, 480]).astype(int)),  # to put the angle text at the elbow
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)  # font and color of the text
            
            # Rep logic using average arm angle to keep it simple/stable
            avg_angle = (left_angle + right_angle) / 2.0
            shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2.0
            hip_center_x = (left_hip[0] + right_hip[0]) / 2.0
            torso_offset = shoulder_center_x - hip_center_x

            if phase == "waiting_down":
                if avg_angle >= BOTTOM_ANGLE:
                    down_confirm += 1
                else:
                    down_confirm = 0

                if down_confirm >= CONFIRM_FRAMES:
                    phase = "going_up"
                    stage = "down"
                    rep_frames = 0
                    rep_min_angle = avg_angle
                    rep_max_angle = avg_angle
                    left_elbow_start_x = left_elbow[0]
                    right_elbow_start_x = right_elbow[0]
                    torso_offset_start = torso_offset
                    max_left_elbow_drift = 0.0
                    max_right_elbow_drift = 0.0
                    max_body_swing = 0.0
                    rep_motion_started = False
                    down_confirm = 0

            elif phase == "going_up":
                rep_frames += 1
                rep_min_angle = min(rep_min_angle, avg_angle)
                rep_max_angle = max(rep_max_angle, avg_angle)
                left_drift = abs(left_elbow[0] - left_elbow_start_x)
                right_drift = abs(right_elbow[0] - right_elbow_start_x)
                swing_from_start = abs(torso_offset - torso_offset_start)
                max_left_elbow_drift = max(max_left_elbow_drift, left_drift)
                max_right_elbow_drift = max(max_right_elbow_drift, right_drift)
                max_body_swing = max(max_body_swing, swing_from_start)

                if avg_angle < LIVE_CHECK_START_ANGLE:
                    rep_motion_started = True

                # live coaching (short hold to prevent flicker)
                if rep_motion_started and (left_drift > DRIFT_THRESHOLD or right_drift > DRIFT_THRESHOLD):
                    live_warning_text = "KEEP ELBOWS STILL"
                    live_warning_hold = LIVE_WARNING_HOLD_FRAMES
                elif rep_motion_started and swing_from_start > SWING_THRESHOLD:
                    live_warning_text = "LESS BODY SWING"
                    live_warning_hold = LIVE_WARNING_HOLD_FRAMES

                if avg_angle <= TOP_ANGLE:
                    top_confirm += 1
                else:
                    top_confirm = 0

                if top_confirm >= CONFIRM_FRAMES:
                    phase = "going_down"
                    stage = "up"
                    top_confirm = 0

            elif phase == "going_down":
                rep_frames += 1
                rep_min_angle = min(rep_min_angle, avg_angle)
                rep_max_angle = max(rep_max_angle, avg_angle)
                left_drift = abs(left_elbow[0] - left_elbow_start_x)
                right_drift = abs(right_elbow[0] - right_elbow_start_x)
                swing_from_start = abs(torso_offset - torso_offset_start)
                max_left_elbow_drift = max(max_left_elbow_drift, left_drift)
                max_right_elbow_drift = max(max_right_elbow_drift, right_drift)
                max_body_swing = max(max_body_swing, swing_from_start)

                if rep_motion_started and (left_drift > DRIFT_THRESHOLD or right_drift > DRIFT_THRESHOLD):
                    live_warning_text = "KEEP ELBOWS STILL"
                    live_warning_hold = LIVE_WARNING_HOLD_FRAMES
                elif rep_motion_started and swing_from_start > SWING_THRESHOLD:
                    live_warning_text = "LESS BODY SWING"
                    live_warning_hold = LIVE_WARNING_HOLD_FRAMES

                if avg_angle >= BOTTOM_ANGLE:
                    down_confirm += 1
                else:
                    down_confirm = 0

                if rep_frames > REP_TIMEOUT_FRAMES:
                    phase = "waiting_down"
                    stage = None
                    down_confirm = 0
                    top_confirm = 0
                    rep_frames = 0
                    rep_min_angle = 180
                    rep_max_angle = 0
                    max_left_elbow_drift = 0.0
                    max_right_elbow_drift = 0.0
                    max_body_swing = 0.0
                    rep_motion_started = False
                    live_warning_text = ""
                    live_warning_hold = 0
                    final_warning_text = "RESET FORM"
                    final_warning_hold = WARNING_HOLD_FRAMES

                if down_confirm >= CONFIRM_FRAMES:
                    counter += 1
                    too_fast = rep_frames < FAST_REP_FRAMES
                    partial = rep_min_angle > TOP_ANGLE
                    left_elbow_bad = max_left_elbow_drift > DRIFT_THRESHOLD
                    right_elbow_bad = max_right_elbow_drift > DRIFT_THRESHOLD
                    body_swing_bad = max_body_swing > SWING_THRESHOLD

                    # single priority warning (less chaotic)
                    if too_fast:
                        last_rep_label = "SLOW DOWN"
                    elif partial:
                        last_rep_label = "FULL RANGE"
                    elif left_elbow_bad or right_elbow_bad:
                        last_rep_label = "KEEP ELBOWS STILL"
                    elif body_swing_bad:
                        last_rep_label = "LESS BODY SWING"
                    else:
                        last_rep_label = "GOOD"

                    if last_rep_label != "GOOD":
                        final_warning_text = last_rep_label
                        final_warning_hold = WARNING_HOLD_FRAMES

                    phase = "waiting_down"
                    stage = "down"
                    down_confirm = 0
                    rep_motion_started = False
                    live_warning_text = ""
                    live_warning_hold = 0

        #render curl counter
        cv2.rectangle(image, (0,0), (200,65), (255,145,238), -1)  # rectangle for counter
        #rep data display
        cv2.putText(image, 'REPS', (15,12),#passing image and title and position of the text
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)#font and color of the text
        cv2.putText(image, str(counter), (10,60),#passing the counter and position of the text
            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(image, f'LAST: {last_rep_label}', (210,30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        if final_warning_hold > 0:
            cv2.putText(image, final_warning_text, (210,70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2, cv2.LINE_AA)
            final_warning_hold -= 1
        elif live_warning_hold > 0 and phase != "waiting_down" and rep_motion_started:
            cv2.putText(image, live_warning_text, (210,70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2, cv2.LINE_AA)
            live_warning_hold -= 1
    

        
            # print(landmarks)  # print the landmarks to see what we have
        # draw detection to the image (landmarks and connections)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),  # color at diff dots
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),  # color of connections(lines)
            )

        cv2.imshow("MediaPipe Pose", image)  # visulize it
        if cv2.waitKey(10) & 0xFF == ord("q"):  # breaking out of our feed (if we press q)
            break

cap.release()
cv2.destroyAllWindows()