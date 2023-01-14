import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import numpy as np
import glob

# initialize video capture
cap = cv2.VideoCapture(0)
# set vidio capture resolutions
cap.set(3, 640)
cap.set(4, 480)
# SelfiSegmentation is used to remove the background of the frame
# and replace it with our images in the directory.
segmentor = SelfiSegmentation()
# display the frames per second(fps) in the output frames
fpsReader = cvzone.FPS()
fgbg = cv2.createBackgroundSubtractorMOG2()
# list of the files with '.jpg' extensions
listImg = glob.glob("BackgroundImages/*.jpg")
# empty list for storing images
imgList = []
# iterate over the list of paths and read images
for imgPath in listImg:
    img = cv2.imread(imgPath)
    imgList.append(img)
# set initial index of the background image
indexImg = 0
# infinite loop for reading framesq
while True:
    success, img = cap.read()
    # resize given background image and
    # convert it to the same size as the frame
    resized_image = cv2.resize(imgList[indexImg], (640, 480), interpolation=cv2.INTER_AREA)
    # remove background from the frame
    imgOut = segmentor.removeBG(img, resized_image, threshold=0.8)
    # stack ground truth and output images in same window
    imgStack = cvzone.stackImages([img, imgOut], 2, 1)
    _, imgStack = fpsReader.update(imgStack)
    # key for display a frame for 1 ms
    key = cv2.waitKey(1)

    if key == ord('q'):
        break
    # change background images pressing 'n'
    elif key == ord('n'):
        # if image index is in range 0 to len(imgList)-1
        # change frames in ascending order else start from 0
        if 0 <= indexImg < len(imgList)-1:
            indexImg += 1
        else:
            indexImg = 0

    """
    Background Subtraction is a technique used for generating a foreground mask. 
    It is a two-step process: Background Initialisation and Updation.
    Background Initialisation: In the Background Initialisation process, 
                                an initial model is computed.
    Background Updation: In Background Updation, 
                        the model is updated to keep new changes in the frame.
    """
    fgmask = fgbg.apply(img)
    kernel = np.ones((5, 5), np.uint8)
    a = 0
    bounding_rect = []
    fgmask = cv2.erode(fgmask, kernel, iterations=5)
    fgmask = cv2.dilate(fgmask, kernel, iterations=5)

    """
    Erosion and Dilation: Erosion is a process of eroding the 
    boundaries of the foreground object. 
    As the kernel slides over the image, depending on the values of 
    pixels (0 or 1), the value is eroded while Dilation is a process 
    of increasing the size of the foreground image thus increasing 
    the white region.
    """
    cv2.imshow("image", imgOut)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(0, len(contours)):
        bounding_rect.append(cv2.boundingRect(contours[i]))
    for i in range(0, len(contours)):
        if bounding_rect[i][2] >= 40 or bounding_rect[i][3] >= 40:
            a = a + (bounding_rect[i][2]) * bounding_rect[i][3]
        if (a >= int(img.shape[0]) * int(img.shape[1]) / 3):
            if 0 <= indexImg < len(imgList) - 1:
                indexImg += 1
            else:
                indexImg = 0

            cv2.imshow("image", imgOut)



