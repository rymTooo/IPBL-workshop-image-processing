import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2

dev = 0

def main():
    cap = cv2.VideoCapture(dev)
    ht  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wt  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print("*** CAMERA STATUS ***")
    print(ht, "x", wt, "@", fps)
    print("PRESS \'q\' TO CLOSE...")

    while cap.isOpened():
        ret, frame = cap.read()

        if cv2.waitKey(int(1000/fps))==ord('q') or ret==False:
            break

        if ret:
            cv2.imshow("video", frame)
    
    cv2.destroyAllWindows()
    cap.release()

if __name__=='__main__':
    main()