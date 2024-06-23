import numpy as np
import cv2
from util import add_text_to_image, calcdistance
import math


eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                    "haarcascade_eye.xml")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                    "haarcascade_frontalface_default.xml")



def detect_features(frame,
                    width: int,
                    height: int, 
                    to_display=None):
    """Takes in a frame and its size.
    Returns the frame in grayscale with bounding boxes around the face and eyes.
    Returns the distance between the eyes in pixels.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    num_faces = len(faces)

    # add the number of faces and number of eyes to the output screen
    org = (int(0.5 * width), int(0.6*height))
    gray = add_text_to_image(gray, f'Faces: {num_faces}', (org[0], org[1] - 32))
    if not num_faces:
        gray = add_text_to_image(gray, f'Eyes: 0', (org[0], org[1]))

    if to_display:
        depth = calcDepth(to_display)
        gray = add_text_to_image(gray,
                                 f'Last Eye Distance\n(Pixels): {to_display:.2f} = {depth:.2f} mm',
                                 (org[0], org[1] + 32),
                                 font_scale=0.5)

        
    
    if len(faces) > 1:  # delete later once generalised for multiple faces
        return gray, None


    eye_midpoints = []
    eye_sizes = []
    eye_distances = []

    # loop through all the possible faces for completenes -- however at the moment only consider one face

    if len(faces) == 0: # if we find no faces then we should search the whole frame for whether there are eyes present or not
        faces = [(0, 0, width, height)]

    for (x,y,w,h) in faces:

        # isolate just the face in the image
        grayface = gray[y:y+h, x:x+w]
    

        # find the eyes in the frame
        eyes = eye_cascade.detectMultiScale(grayface)

        # add the number of eyes to the frame information
        num_eyes = len(eyes)
        gray = add_text_to_image(gray, f'Eyes: {num_eyes}', (org[0], org[1]))

        
        for (xp, yp, wp, hp) in eyes:
            grayface = cv2.rectangle(grayface, (xp, yp), (xp+wp, yp+hp),
                    color=(255, 0, 0), thickness=3)

            eye_midpoints.append((xp + wp/2, yp + hp/2)) # add in the center location of the eye
            eye_sizes.append(wp*hp)

        # create a bounding box around the eye 
        gray = cv2.rectangle(gray, (x, y), (x+w, y+h),
                        color=(255, 0, 0), thickness=1)
        
        
    distance = 0
    if len(eye_midpoints) == 2:
        
        # test to make sure the two eyes found are pretty much the same size
        SIZE_THRESHOLD = 0.05
        eye_ratios = eye_sizes[0]/eye_sizes[1]
        if eye_ratios < 1 + SIZE_THRESHOLD and eye_ratios > 1 - SIZE_THRESHOLD:
            distance = calcdistance(eye_midpoints)
    
    return gray, distance


def correctIVCam(image, rot=90):
    # given a landscape frame convert it to portrait
    height, width = image.shape[:2]
    centerX, centerY = (width // 2, height // 2)

    M = cv2.getRotationMatrix2D((centerX, centerY), rot, 1.0)
    rotated = cv2.warpAffine(image, M, (width, height))
    return rotated


def calcDepth(pwidth,
              foc = 32,
              alpha = 50):
    
    a2 = math.radians(50)/2
    h2 = pwidth/2
    D = foc * (h2 * math.tan(a2))/(h2/(2*math.tan(a2)) - foc)
    return D


def main():

    source = 1
    stream = cv2.VideoCapture(source)

    if not stream.isOpened():
        print("No stream :(")
        exit()

    fps = stream.get(cv2.CAP_PROP_FPS)
    width = int(stream.get(3))
    height = int(stream.get(4))

    output = cv2.VideoWriter("assets/6_facial_detection.mp4",
             cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
             fps=fps, frameSize=(width, height))
    

    distance = None
    prev_distance = distance
    while(True):
        ret, frame = stream.read()

        # rotate if using ivcam
        if source == 0:
            frame = correctIVCam(frame)

        width, height, dim = frame.shape
        if not ret:
            print("No more stream :(")
            break

        # detect the features
        frame, distance = detect_features(frame, width, height, prev_distance)

        if distance:
            prev_distance = distance
        
        output.write(frame)
        cv2.imshow("Webcam!", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    stream.release()
    cv2.destroyAllWindows() 

main()
