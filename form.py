import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils  # all drawing utilities
mp_pose = mp.solutions.pose  # all pose model(face detection,iris...we decided pose)

# video feed
cap = cv2.VideoCapture(0)  # setting video(maybe web cam...depends on the device)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:  # access pose estimation with how much confidence we want(accuracy)
    while cap.isOpened():  # while cam is open
        ret, frame = cap.read()  # read the feed of my web cam and status in frame
        if not ret or frame is None:
            continue

        # detecting stuff and render
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # recolor image to RGB because mediapipe uses RGB
        image.flags.writeable = False

        # make detection
        results = pose.process(image)  # process the image and store it in results

        image.flags.writeable = True  # to improve performance
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # converting to BGR for opencv
       #draw detection to the image(lansdmarks and connections)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),#color at diff dots
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))  #color of connections(lines)
            
        cv2.imshow("MediaPipe Pose", image)  # visulize it
        if cv2.waitKey(10) & 0xFF == ord("q"):  # breaking out of our feed (if we press q)
            break

cap.release()
cv2.destroyAllWindows()