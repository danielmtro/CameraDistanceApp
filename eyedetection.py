import numpy as np
import cv2
from util import add_text_to_image


eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                    "haarcascade_eye.xml")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                    "haarcascade_frontalface_default.xml")



def detect_features(frame, width: int, height: int):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 2, 5)
    num_faces = len(faces)

    org = (int(0.75 * width), int(0.8*height))

    gray = add_text_to_image(gray, f'Faces: {num_faces}', org)
    if not num_faces:
        gray = add_text_to_image(gray, f'Eyes: 0', (org[0], org[1] + 32))

    for (x,y,w,h) in faces:
        grayface = gray[y:y+h, x:x+w]
    
        eyes = eye_cascade.detectMultiScale(grayface, 
                    1.5, minNeighbors=4)
        
        num_eyes = len(eyes)
        # gray = cv2.putText(gray, f'Eyes: {num_eyes}', org, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        gray = add_text_to_image(gray, f'Eyes: {num_eyes}', (org[0], org[1] + 32))
        
        for (xp, yp, wp, hp) in eyes:
            grayface = cv2.rectangle(grayface, (xp, yp), (xp+wp, yp+hp),
                    color=(255, 0, 0), thickness=3)
    
        gray = cv2.rectangle(gray, (x, y), (x+w, y+h),
                        color=(255, 0, 0), thickness=1)
        
    return gray


def main():

    stream = cv2.VideoCapture(0)

    if not stream.isOpened():
        print("No stream :(")
        exit()

    fps = stream.get(cv2.CAP_PROP_FPS)
    width = int(stream.get(3))
    height = int(stream.get(4))

    output = cv2.VideoWriter("assets/6_facial_detection.mp4",
             cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
             fps=fps, frameSize=(width, height))
    

    while(True):
        ret, frame = stream.read()
        if not ret:
            print("No more stream :(")
            break
        
        frame = detect_features(frame, width, height)
        output.write(frame)
        cv2.imshow("Webcam!", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    stream.release()
    cv2.destroyAllWindows() 

main()
