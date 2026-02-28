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
        try:
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
            right_angle = calculateangle(right_shoulder, right_elbow, right_wrist)

            #visulize
            cv2.putText(image, str(int(left_angle)), 
                tuple(np.multiply(left_elbow, [640, 480]).astype(int)),  # to put the angle text at the elbow
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)  # font and color of the text
            cv2.putText(image, str(int(right_angle)), 
                tuple(np.multiply(right_elbow, [640, 480]).astype(int)),  # to put the angle text at the elbow
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)  # font and color of the text
            
            # Curl counter logic (count only when both arms curl together)
            if left_angle > 160 and right_angle > 160:
                stage = "down"
            if left_angle < 30 and right_angle < 30 and stage == 'down':
                stage = "up"
                counter += 1
        except:
            pass

        #render curl counter
        cv2.rectangle(image, (0,0), (200,65), (255,145,238), -1)  # rectangle for counter
        #rep data display
        cv2.putText(image, 'REPS', (15,12),#passing image and title and position of the text
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)#font and color of the text
        cv2.putText(image, str(counter), (10,60),#passing the counter and position of the text
            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
    

        
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