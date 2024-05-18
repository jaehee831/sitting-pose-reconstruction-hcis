import cv2
import datetime
import time

videoIdx = 4
cap = cv2.VideoCapture(videoIdx)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

print("원본 동영상 너비(가로) : {}, 높이(세로) : %d, %d"%(width, height))

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)

delay = round(1000/fps)

print("변환된 동영상 너비(가로) : 높이(세로) : %d, %d"%(width, height))

now = datetime.datetime.now()
videoFileName = 'D:/Github/ActionNet/data/tests/webcam/' + str(now).replace(':', '').replace('.','') +'_Cam_3.avi'

print(str(now).replace(':', '').replace('.', ''))
codec = "DIVX"
fourcc = cv2.VideoWriter_fourcc(*codec)
out = cv2.VideoWriter(videoFileName, fourcc, fps, (int(width), int(height)))

while True:
    retval, frame = cap.read()

    if not retval:
        break

    unixTime = str(time.time())
    currentTime = str(datetime.datetime.now())
    startTime = str(now)
    cv2.putText(frame, startTime, (5, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
    cv2.putText(frame, currentTime, (400, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
    cv2.putText(frame, unixTime, (800, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

    out.write(frame)
    cv2.imshow("Scene camera 3", frame)

    cv2.waitKey(delay)



out.release()
cap.release()