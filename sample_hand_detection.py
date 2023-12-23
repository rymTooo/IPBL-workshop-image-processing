import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import math
import numpy as np
from MediapipeHandLandmark import MediapipeHandLandmark as HandLmk

hand_data = list(range(0,21))
hand_data_empty = True

#distance function
def distance(a,b):
    d = math.sqrt(math.pow(a[0]-b[0],2)+math.pow(a[1]-b[1],2)+math.pow(a[2]-b[2],2))
    return d

# draw line connecting landmarks in hand
def drawhand_mech(i,thickenss,color):
    for j in [0,1,2,3,5,6,7,9,10,11,13,14,15,17,18,19]:
                cv2.line(annotated_frame, (Hand.get_landmark( id_hand = i, id_landmark = j )[:2]),(Hand.get_landmark( id_hand = i, id_landmark = j+1 )[:2]),(color), thickness = thickenss)
    cv2.line(annotated_frame, (Hand.get_landmark( id_hand = i, id_landmark = 0 )[:2]),(Hand.get_landmark( id_hand = i, id_landmark = 5 )[:2]),(color), thickness = thickenss)
    cv2.line(annotated_frame, (Hand.get_landmark( id_hand = i, id_landmark = 0 )[:2]),(Hand.get_landmark( id_hand = i, id_landmark = 17 )[:2]),(color), thickness = thickenss)
    cv2.line(annotated_frame, (Hand.get_landmark( id_hand = i, id_landmark = 5 )[:2]),(Hand.get_landmark( id_hand = i, id_landmark = 9 )[:2]),(color), thickness = thickenss)
    cv2.line(annotated_frame, (Hand.get_landmark( id_hand = i, id_landmark = 13 )[:2]),(Hand.get_landmark( id_hand = i, id_landmark = 9 )[:2]),(color), thickness = thickenss)
    cv2.line(annotated_frame, (Hand.get_landmark( id_hand = i, id_landmark = 13 )[:2]),(Hand.get_landmark( id_hand = i, id_landmark = 17 )[:2]),(color), thickness = thickenss)

# draw hand mech from given list of coordinate
def drawhand_mech_coordinate(crd,color,thickness):
    for j in [0,1,2,3]:
        cv2.line(annotated_frame,crd[j][:2],crd[j+1][:2],(11,35,89),thickness = thickness)
    cv2.line(annotated_frame,crd[0][:2],crd[5][:2],(color),thickness = thickness)
    cv2.line(annotated_frame,crd[0][:2],crd[17][:2],(color),thickness = thickness)
    cv2.line(annotated_frame,crd[5][:2],crd[9][:2],(color),thickness = thickness)
    cv2.line(annotated_frame,crd[9][:2],crd[13][:2],(color),thickness = thickness)
    cv2.line(annotated_frame,crd[17][:2],crd[13][:2],(color),thickness = thickness)
    for j in [5,6,7]:
        cv2.line(annotated_frame,crd[j][:2],crd[j+1][:2],(18,58,146),thickness = thickness)
    for j in [9,10,11]:
        cv2.line(annotated_frame,crd[j][:2],crd[j+1][:2],(27,86,219),thickness = thickness)
    for j in [13,14,15]:
        cv2.line(annotated_frame,crd[j][:2],crd[j+1][:2],(88,133,234),thickness = thickness)
    for j in [17,18,19]:
        cv2.line(annotated_frame,crd[j][:2],crd[j+1][:2],(235,234,252),thickness = thickness)

    for j in [4,8,12,16,20]:#draw circle at finger tip
        cv2.circle(annotated_frame,crd[j][:2],5,(1,1,15),thickness = -1)

#save hand data to hand_data
def save_hand_data(i):
    for j in range(0,21):
        hand_data[j] = (list(Hand.get_landmark(id_hand=i, id_landmark= j)))
    # print(hand_data)
    
"""not use anymore. Yuichi got a better one"""
# def get_hand_data(i):

#     tf10 = Hand.get_landmark(id_hand= i, id_landmark= 10)
#     tf11 = Hand.get_landmark(id_hand= i, id_landmark= 11)
#     tf15 = Hand.get_landmark(id_hand= i, id_landmark= 15)
#     tf14 = Hand.get_landmark(id_hand= i, id_landmark= 14)
#     ref = (distance(tf10,tf11) + distance(tf15,tf14))*0.5
#     #this is reference = average of constant distance


#     f8 = Hand.get_landmark(id_hand= i, id_landmark= 8)
#     f12 = Hand.get_landmark(id_hand= i, id_landmark= 12)
#     f16 = Hand.get_landmark(id_hand= i, id_landmark= 16)
#     d1 = distance(f8,f16)/ref #distance of d1 divide by ref
#     d2 = distance(f16,f12)/ref # distance of d2 divide by ref


#     if(d1 > 1.0 and d2 > 0.7): # the ratio come from testing 
#         print("This is good posture ++")
#     else:
#         print("This is bad posture --")
#     print("")

def draw_chopstick(i,length,frame): # draw virtual chopstick based on hand data, receive 3 args ( id_hand, length of chopstick, frame to be displayed on)
    
    #referecne distance for drawing chopstick
    ref_d = np.linalg.norm(Hand.get_landmark(id_hand=i,id_landmark=14)-Hand.get_landmark(id_hand=i,id_landmark=13))


    #Upper Chopstick code
    f11p = np.array(Hand.get_landmark(id_hand= i, id_landmark= 11))
    f10p = np.array(Hand.get_landmark(id_hand= i, id_landmark= 10))
    vfp2 = f11p - f10p
    v_U_chopstick = vfp2
    U_chopstick_mid = (np.array(Hand.get_landmark(id_hand= i, id_landmark= 8)) + np.array(Hand.get_landmark(id_hand= i, id_landmark= 12)))*0.5
    U_chopstick_p2 = np.array(U_chopstick_mid + length*v_U_chopstick,np.int32)
    U_chopstick_p1 = np.array(U_chopstick_mid - length*0.8*v_U_chopstick,np.int32)
    cv2.line(frame,U_chopstick_p1[:2],list(U_chopstick_p2)[:2],[160,231,248],thickness = 5)#draw upper chopstick


    #Lower Chopstick code
    p1_L = np.array((Hand.get_landmark(id_hand=i,id_landmark=19)*2 + Hand.get_landmark(id_hand=i,id_landmark=20))/3)
    p2_L = np.array((Hand.get_landmark(id_hand=i,id_landmark=15)*2 + Hand.get_landmark(id_hand=i,id_landmark=16))/3)
    v_p2 = p2_L - p1_L
    v_p2_u = v_p2/np.linalg.norm(v_p2)
    L_chopstick_mid = np.array(p1_L + v_p2_u*ref_d*0.35,np.int32)
    # cv2.circle(frame,L_chopstick_mid[:2],5,(1,1,15),thickness = -1)

    rat = 4.5 # for fine tune lower chopstick location
    p4_L = np.array((Hand.get_landmark(id_hand=i,id_landmark=5)*(rat-1) + Hand.get_landmark(id_hand=i,id_landmark=0))/rat)
    v_L2 = p4_L - np.array(Hand.get_landmark(id_hand=i,id_landmark=17))
    v_L2_u = v_L2/np.linalg.norm(v_L2)
    L_chopstick_end = np.array(p4_L + v_L2_u*ref_d*0.3,np.int32)
    # cv2.circle(frame,L_chopstick_end[:2],5,(1,1,15),thickness = -1)
    v_L_chopstick = L_chopstick_mid - L_chopstick_end
    cv2.line(frame,np.array(L_chopstick_end - 0.5*v_L_chopstick,np.int32)[:2],np.array(L_chopstick_end + 2.5*v_L_chopstick,np.int32)[:2],[160,231,248],thickness = 5)#draw lower chopstick


cap = cv2.VideoCapture(0)
Hand = HandLmk(num_hands = 1)


while cap.isOpened():
    ret, frame = cap.read()
    flipped_frame = cv2.flip(frame, 1)#fliped frame
    Hand.detect(flipped_frame)#dectec hand in flipped frame

    annotated_frame = Hand.visualize(flipped_frame)
    if not hand_data_empty and Hand.num_detected_hands != 0:
        # drawhand_mech_coordinate(hand_data,[81,214,241],15)#drawing function
        draw_chopstick(i,5,flipped_frame) # draw virtual chopstick

    for i in range(Hand.num_detected_hands): #if detected multiple hands choose 1
        if Hand.get_handedness( id_hand = i ) == 'Right': #if the hand is right hand
            drawhand_mech(i,2,[255,255,255])
            if key == ord('c'):
                # save_hand_data(i)
                hand_data_empty = not hand_data_empty
                
    cv2.imshow('annotated frame', annotated_frame)
    cv2.imshow('flipped_frame', flipped_frame)
    key = cv2.waitKey(1)&0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
Hand.release()