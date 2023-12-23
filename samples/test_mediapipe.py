import cv2
import numpy as np
import mediapipe as mp
from typing import List

dev = 0

def init(detection=0.2, tracking=0.2):
    global hands, pose, fmesh, face, segment

    hands = mp.solutions.hands.Hands(
        min_detection_confidence=detection, min_tracking_confidence=tracking)
    pose = mp.solutions.pose.Pose(
        static_image_mode=False, model_complexity=1, enable_segmentation=False , min_detection_confidence=detection, min_tracking_confidence=tracking)
    fmesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=detection)
    face = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=detection)
    segment = mp.solutions.selfie_segmentation.SelfieSegmentation(
        model_selection=0)

def getFace(frame, getkeys=True):
    global face

    ht, wt, _ = frame.shape
    results = face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    point_list = []
    face_box = []
    face_keypoints = []
    if results.detections is not None:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            face_box = [int(bbox.xmin*wt), int(bbox.ymin*ht), int(bbox.width*wt), int(bbox.height*ht)]
            if getkeys:
                for i, landmark in enumerate(results.detections[0].location_data.relative_keypoints):
                    x = max(1, min(int(landmark.x * wt), wt-1))
                    y = max(1, min(int(landmark.y * ht), ht-1))
                    face_keypoints.append([int(x), int(y)])
            point_list.append([face_box, face_keypoints])
    return point_list

def getFaceMesh(frame, getkeys=True):
    global fmesh

    ht, wt, _ = frame.shape
    results = fmesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    point_list = []
    face_keypoints = []
    if results.multi_face_landmarks is not None:
        for i, one_face_landmarks in enumerate(results.multi_face_landmarks):
            for i, landmark in enumerate(one_face_landmarks.landmark):
                x = max(1, min(int(landmark.x * wt), wt-1))
                y = max(1, min(int(landmark.y * ht), ht-1))
                z = landmark.z * wt
                face_keypoints.append([int(x), int(y), int(z)])
            point_list.append(face_keypoints)
    return point_list

def getHand(frame):
    global hands

    ht, wt, _ = frame.shape
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    hpoint_list = []
    lhand_points = []
    rhand_points = []
    if results.multi_hand_landmarks is not None:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if results.multi_handedness[i].classification[0].label == "Left":
                for i, landmark in enumerate(hand_landmarks.landmark):
                    x = max(1, min(int(landmark.x * wt), wt-1))
                    y = max(1, min(int(landmark.y * ht), ht-1))
                    lhand_points.append([int(x), int(y), landmark.z])
            elif results.multi_handedness[i].classification[0].label == "Right":
                for i, landmark in enumerate(hand_landmarks.landmark):
                    x = max(1, min(int(landmark.x * wt), wt-1))
                    y = max(1, min(int(landmark.y * ht), ht-1))
                    rhand_points.append([int(x), int(y), landmark.z])
        hpoint_list.append([lhand_points, rhand_points])
    return hpoint_list

def getPose(frame): # Mediapipe can detect only one person.
    global pose

    ht, wt, _ = frame.shape
    results = pose.process(frame)
    pose_points = []
    pose_list = []
    if results.pose_landmarks is not None:
        for i, point in enumerate(results.pose_landmarks.landmark):
            x = max(1, min(int(point.x * wt), wt-1))
            y = max(1, min(int(point.y * ht), ht-1))
            z = int(point.z * wt)
            pose_points.append([x, y, z, point.visibility])
        pose_list.append(pose_points)
    return pose_list

def getSegmentImage(frame, bgimage=[], dep=0.1):
    global segment

    res = []
    sresults = segment.process(frame)

    if sresults.segmentation_mask is not None:
        condition = np.stack(
            (sresults.segmentation_mask, )*3, axis=-1) > dep
        if len(bgimage)==0:
            bg = np.ones(frame.shape, dtype=np.uint8)*255
        else:
            ht, wt, _ = frame.shape
            bg = cv2.resize(bgimage, (wt, ht))

        res = np.where(condition, frame, bg)

    return res


def main():
    global dev

    init()

    cap = cv2.VideoCapture(dev)

    back = cv2.imread("../image/swan.jpg")
    key=-1
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        outframe = frame.copy()

        tmp = cv2.waitKey(1)
        if tmp!=-1:
            print(tmp)
            key = tmp
        if key==ord('q') or ret == False:
            break

        if key==ord('h') or key==ord('a'):
            ## Hands #############################################################
            hands = getHand(frame)
            if len(hands)==1:
                left = hands[0][0]
                right = hands[0][1]

                if len(left)>0:
                    for point in left:
                        cv2.circle(outframe, (point[0], point[1]), 5, [255,0,0], -1)
                        
                if len(right)>0:
                    for point in right:
                        cv2.circle(outframe, (point[0], point[1]), 5, [0,0,255], -1)
        if key==ord('f') or key==ord('a'):
            ## Fase #############################################################
            face = getFace(frame)
            if len(face)==1:
                box, keypoints = face[0]
                cv2.rectangle(outframe, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), [255, 255, 0], 1)
                for point in keypoints:
                    cv2.circle(outframe, (point[0], point[1]), 5, [255,255,0], -1)
        if key==ord('m') or key==ord('a'):
            ## FaseMesh #############################################################
            fmesh = getFaceMesh(frame)
            if len(fmesh)==1:
                meshKeys = fmesh[0]
                for point in meshKeys:
                    cv2.circle(outframe, (point[0], point[1]), 2, [255,0,255], -1)
        if key==ord('p') or key==ord('a'):
            ## Pose #############################################################
            pose = getPose(frame)
            if len(pose)==1:
                poseKeys = pose[0]
                for point in poseKeys:
                    cv2.circle(outframe, (point[0], point[1]), 5, [0,255, 0], -1)
        if key==ord('s') or key==ord('a'):
            ## Segments #############################################################
            segs = getSegmentImage(frame, bgimage=back)
            if len(segs)>0:
                cv2.imshow("seg", segs)            
        


        cv2.imshow("video", outframe)

    cv2.destroyAllWindows()
    cap.release()

if __name__=='__main__':
    main()
