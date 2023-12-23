import cv2

def main():
    img1 = cv2.imread("C:\oit\py23_ipbl\SourceCode\image\DSC09994.JPG")
    img2 = cv2.imread("C:\oit\py23_ipbl\SourceCode\image\standard\Parrots.bmp")
    # img2 = cv2.imread("../image/standard/Girl.bmp")
    # print("H x W x Color Channel", img.shape)
    # print(img.size)
    # print(img, "img")
    rows, columns, channels = img2.shape
    roi = img1[:rows,:columns]
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret,mask = cv2.threshold(img2gray,50,255,cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    img2_logo = cv2.bitwise_and(img2,img2, mask = mask)
    result = cv2.add(img2_logo,img1_bg)
    img1[50:rows+50,50:columns+50] = result
    img3 = cv2.GaussianBlur(img1,(3,3),2)
    # print(img.shape)
    cv2.imshow("test",img3)
    cv2.waitKey(0)
    # cv2.imshow("image", img)
    # cv2.imshow("image2", img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__=='__main__':
    main()