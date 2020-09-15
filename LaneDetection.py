import numpy as np
import cv2
import utlis
cameraFeed = False
video_Path = 'test3.jpg'
cameraNo = 1
frameWidth = 600
frameHeight = 350
if cameraFeed:
    intialTracbarValues = [24, 55, 12, 100]
else:
    intialTracbarValues = [42, 63, 14, 87]
if cameraFeed:
    capture = cv2.VideoCapture(cameraNo)
    capture.set(3, frameWidth)
    capture.set(4, frameHeight)
else:
    capture = cv2.VideoCapture(video_Path)
count = 0
no_Of_ArrayVals = 10
global arrayCurve, arrayCounter
arrayCounter = 0
arrayCurve = np.zeros([no_Of_ArrayVals])
myVals = []
utlis.initializeTrackbars(intialTracbarValues)
while True:

    success, img = capture.read()
    if cameraFeed == False:
        img = cv2.resize(img, (frameWidth, frameHeight), None)
    imgWarpPoints = img.copy()
    imgFinal = img.copy()
    imgCanny = img.copy()
    imgUndis = utlis.undistort(img)
    imgThres, imgCanny, imgColor = utlis.thresholding(imgUndis)
    src = utlis.valTrackbars()
    imgWarp = utlis.perspective_warp(imgThres, dst_size=(frameWidth, frameHeight), src=src)
    imgWarpPoints = utlis.drawPoints(imgWarpPoints, src)
    imgSliding, curves, lanes, ploty = utlis.sliding_window(imgWarp, draw_windows=True)

    try:
        curverad = utlis.get_curve(imgFinal, curves[0], curves[1])
        lane_curve = np.mean([curverad[0], curverad[1]])
        imgFinal = utlis.draw_lanes(img, curves[0], curves[1], frameWidth, frameHeight, src=src)

        currentCurve = lane_curve // 50
        if int(np.sum(arrayCurve)) == 0:
            averageCurve = currentCurve
        else:
            averageCurve = np.sum(arrayCurve) // arrayCurve.shape[0]
        if abs(averageCurve - currentCurve) > 200:
            arrayCurve[arrayCounter] = averageCurve
        else:
            arrayCurve[arrayCounter] = currentCurve
        arrayCounter += 1
        if arrayCounter >= noOfArrayValues: arrayCounter = 0
        cv2.putText(imgFinal, str(int(averageCurve)), (frameWidth // 2 - 70, 70), cv2.FONT_HERSHEY_DUPLEX, 1.75,
                    (0, 0, 255), 2, cv2.LINE_AA)

    except:
        lane_curve = 00
        pass

    imgFinal = utlis.drawLines(imgFinal, lane_curve)
    imgThres = cv2.cvtColor(imgThres, cv2.COLOR_GRAY2BGR)
    imgBlank = np.zeros_like(img)
    imgStacked = utlis.stackImages(0.7, ([img, imgUndis, imgWarpPoints],
                                         [imgColor, imgCanny, imgThres],
                                         [imgWarp, imgSliding, imgFinal]
                                         ))

    cv2.imshow("PipeLine", imgStacked)
    cv2.imshow("Result", imgFinal)
    if cv2.waitKey(2) & 0xFF == ord('a'):
      break

capture.release()
cv2.destroyAllWindows()
cv2.destroyAllWindows()
