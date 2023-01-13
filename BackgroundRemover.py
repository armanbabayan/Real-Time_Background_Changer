import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import numpy as np
import glob
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

segmentor = SelfiSegmentation()
fpsReader = cvzone.FPS()
fgbg = cv2.createBackgroundSubtractorMOG2()

listImg = glob.glob("BackgroundImages/*.jpg")
imgList = []
print(listImg)
for imgPath in listImg:
    img = cv2.imread(imgPath)
    imgList.append(img)

indexImg = 0
print("Image list", len(imgList))
off = False
while True:
    success, img = cap.read()
    imgOut = segmentor.removeBG(img, (255, 0, 255), threshold=0.83)
    resized_image = cv2.resize(imgList[indexImg], (640, 480), interpolation=cv2.INTER_AREA)
    imgOut = segmentor.removeBG(img, resized_image, threshold=0.8)

    imgStack = cvzone.stackImages([img, imgOut], 2, 1)
    _, imgStack = fpsReader.update(imgStack)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('n'):
        if 0 <= indexImg < len(imgList)-1:
            indexImg += 1
        else:
            indexImg = 0

    # Video capturing starts
    fgmask = fgbg.apply(img)
    kernel = np.ones((5, 5), np.uint8)
    a = 0
    bounding_rect = []
    fgmask = cv2.erode(fgmask, kernel, iterations=5)
    fgmask = cv2.dilate(fgmask, kernel, iterations=5)
    cv2.imshow("image", imgOut)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(0, len(contours)):
        bounding_rect.append(cv2.boundingRect(contours[i]))
    for i in range(0, len(contours)):
        if bounding_rect[i][2] >= 40 or bounding_rect[i][3] >= 40:
            a = a + (bounding_rect[i][2]) * bounding_rect[i][3]
        if (a >= int(img.shape[0]) * int(img.shape[1]) / 3):

            cv2.imshow("image", imgOut)

            key = cv2.destroyAllWindows()


