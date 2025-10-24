import cv2
import numpy as np
from collections import deque

# def nothing(x):
#     pass
# cv2.namedWindow("Color")
# cv2.resizeWindow("Color", 600, 300)
# cv2.createTrackbar('ls', 'Color', 0, 255, nothing)
# cv2.createTrackbar('ls', 'Color', 0, 255, nothing)
# cv2.createTrackbar('ls', 'Color', 0, 255, nothing)
# cv2.createTrackbar('hs', 'Color', 255, 255, nothing)
# cv2.createTrackbar('hs', 'Color', 255, 255, nothing)
# cv2.createTrackbar('hs', 'Color', 255, 255, nothing)

#cap = cv2.VideoCapture(r'C:\Users\8617728\Desktop\mixdt-little-cats-lying-on-an-amchair-32471-hd-ready.mp4', cv2.CAP_ANY)
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(0)

points_buffer = deque(maxlen=20)

while True:
    ret, frame = cap.read()
    if not(ret):
        break
    hsv = frame #cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l1 = 21 #cv2.getTrackbarPos('ls', 'Color')
    l2 = 25 #cv2.getTrackbarPos('ls', 'Color')
    l3 = 60 #cv2.getTrackbarPos('ls', 'Color')
    h1 = 53 #cv2.getTrackbarPos('hs', 'Color')
    h2 = 51 #cv2.getTrackbarPos('hs', 'Color')
    h3 = 255 #cv2.getTrackbarPos('hs', 'Color')

    l_range = np.array([l1, l2, l3])
    h_range = np.array([h1, h2, h3])

    mask = cv2.inRange(hsv, l_range, h_range)

    s_lr1 = np.array([0, 100, 100])
    s_ur1 = np.array([10, 255, 255])
    s_lr2 = np.array([100, 100, 100])
    s_ur2 = np.array([100, 255, 255])
    s_lr3 = np.array([10, 0, 0])
    s_ur3 = np.array([255, 100, 100])

    s_mask1 = cv2.inRange(hsv, s_lr1, s_ur1)
    s_mask2 = cv2.inRange(hsv, s_lr2, s_ur2)
    s_mask3 = cv2.inRange(hsv, s_lr3, s_ur3)

    threshold = s_mask1 + s_mask2 + s_mask3
    threshold = mask

    kernel = np.ones((5,5), np.uint8)
    opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    center = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cx = x + w // 2
            cy = y + h // 2
            center = (cx, cy)
            cross_size = 20
            cv2.line(frame, (cx - cross_size, cy), (cx + cross_size, cy), (255, 0, 0), 2)
            cv2.line(frame, (cx, cy - cross_size), (cx, cy + cross_size), (255, 0, 0), 2)

    if center is not None:
        points_buffer.append(center)
        
    for i in range(1, len(points_buffer)):
        if points_buffer[i-1] is None or points_buffer[i] is None:
            continue
        cv2.line(frame, points_buffer[i-1], points_buffer[i], (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
