import cv2
import numpy as np
#cap = cv2.VideoCapture(r'C:\Users\Arseniy.Zhuk\Documents\CFS_price_copy_validation.mp4', cv2.CAP_ANY)

# while True:
#     ret, frame = cap.read()
#     if not(ret):
#         break
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == 27:
#         break

def print_cam():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('webcam_output.avi', fourcc, 20.0, (640, 480))

    while True:
        ret, frame = cap.read()
        if not(ret):
            break
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        center_pixel = frame[center_y, center_x]
        b, g, r = center_pixel
        bgr_color = (int(b), int(g), int(r))

        # colors = {
        #     'red': [0, 0, 255],
        #     'green': [0, 255, 0],
        #     'blue': [255, 0, 0]
        # }
        # distances = {color: np.linalg.norm(center_pixel - np.array(bgr)) for color, bgr in colors.items()}
        # closest_color = min(distances, key=distances.get)
        # color_bgr = colors[closest_color]
        
        cv2.line(frame, (center_x - 50, center_y), (center_x + 50, center_y), bgr_color, 25)
        cv2.line(frame, (center_x, center_y - 50), (center_x, center_y + 50), bgr_color, 25)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


print_cam()
