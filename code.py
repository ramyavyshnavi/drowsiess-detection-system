import numpy as np
import cv2
import time
from scipy.spatial import distance as dist
import playsound
from threading import Thread
import numpy as np
import face_recognition


# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive frames
# the eye must be below the threshold to set off the alarm
MIN_AER = 0.30  #Minimun eye aspect ratio
EYE_AR_CONSEC_FRAMES = 10   #After passing this number of frames we will detect.

# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0 
ALARM_ON = False #default alarm is false

def eye_aspect_ratio(eye):
 # compute the euclidean distances between the two sets of
 # vertical eye landmarks (x, y)-coordinates
 A = dist.euclidean(eye[1], eye[5])
 B = dist.euclidean(eye[2], eye[4])

 # compute the euclidean distance between the horizontal
 # eye landmark (x, y)-coordinates
 C = dist.euclidean(eye[0], eye[3])

 # compute the eye aspect ratio
 ear = (A + B) / (2.0 * C)

 # return the eye aspect ratio
 return ear

def sound_alarm(alarm_file):
 # play an alarm sound
 playsound.playsound(alarm_file)
 
def main():
    global COUNTER
    video_capture = cv2.VideoCapture(0)
    while True:       
        ret, frame = video_capture.read(0)   #reading video capture

        # get it into the correct format
        #small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # get the correct face landmarks
        
        face_landmarks_list = face_recognition.face_landmarks(frame) #face recognition module to get face landmarks like eyes,nose,etc..

            # get eyes
        for face_landmark in face_landmarks_list:
                        leftEye = face_landmark['left_eye']
                        rightEye = face_landmark['right_eye']
                        #eye aspect ratio for left and right eyes
                        leftEAR = eye_aspect_ratio(leftEye)
                        rightEAR = eye_aspect_ratio(rightEye)
                        # average the eye aspect ratio together for both eyes
                        ear = (leftEAR + rightEAR) / 2
                        #========================converting left and right eye values in numpy arrays
                        lpts = np.array(leftEye)
                        rpts = np.array(rightEye)
                        #==================showing line from left of left eye and right of right eye
                        cv2.polylines(frame, [lpts],True ,(255,255,0), 1)
                        cv2.polylines(frame, [rpts],True ,(255,255,0), 1)
                        
                        # check to see if the eye aspect ratio is below the blink
                        # threshold, and if so, increment the blink frame counter
                        if ear < MIN_AER:
                                COUNTER+= 1

                                # if the eyes were closed for a sufficient number of times
                                # then sound the alarm
                                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                                        # if the alarm is not on, turn it on
                                        if not ALARM_ON:
                                                ALARM_ON = True
                                                t = Thread(target=sound_alarm,
                                                                args=('C:/Users/91807/Desktop/alarm.wav',))
                                                t.deamon = True  #The Daemon Thread does not block the main thread from exiting and continues to run in the background
                                                t.start()

                                        # draw an alarm on the frame
                                        cv2.putText(frame, "ALERT! You are feeling asleep!", (10, 30),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        # otherwise, the eye aspect ratio is not below the blink
                        # threshold, so reset the counter and alarm
                        else:
                                COUNTER = 0
                                ALARM_ON = False

                        # draw the computed eye aspect ratio on the frame to help
                        # with debugging and setting the correct eye aspect ratio
                        # thresholds and frame counters
                        cv2.putText(frame, "EAR: {:.2f}".format(ear), (500, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
                        # show the frame
                        cv2.imshow("Sleep detection program.", frame)

        # if the `q` key was pressed, break from the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    # do a bit of cleanup
    video_capture.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
        
