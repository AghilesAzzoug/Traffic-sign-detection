import os
import re
import sys
from threading import Thread

import cv2


# parralel sign detector
class SignDetector(Thread):
    def __init__(self, classifier, scaledframe):
        Thread.__init__(self)
        self.classifier = classifier
        self.scaledframe = scaledframe
        self._return = None

    def run(self):
        signs = self.classifier.detectMultiScale(
            self.scaledframe,
            1.1,
            5,
            0,
            (10, 10),
            (200, 200))
        self._return = signs

    def join(self, timeout=None):
        Thread.join(self)
        return self._return


def read_paths(path):
    images = [[] for _ in range(2)]
    for dirname, dirnames, _ in os.walk(path):
        for subdirname in dirnames:
            filepath = os.path.join(dirname, subdirname)
            for filename in os.listdir(filepath):
                try:
                    imgpath = str(os.path.join(filepath, filename))
                    images[0].append(imgpath)
                    limit = re.findall('[0-9]+', filename)
                    images[1].append(limit[0])
                except IOError:
                    print("I/O error " + str(IOError))
                except:
                    print("Unexpected error:" + str(sys.exc_info()[0]))
                    raise
    return images


# loads image from "data" folder
def get_keyPoints(imgpath):
    images = read_paths(imgpath)
    imglist = [[], [], [], []]
    cur_img = 0

    #  Scale-Invariant Feature Transform
    sift = cv2.xfeatures2d.SIFT_create()
    for i in images[0]:
        img = cv2.imread(i, 0)
        imglist[0].append(img)
        imglist[1].append(images[1][cur_img])
        cur_img += 1
        keypoints, des = sift.detectAndCompute(img, None)
        imglist[2].append(keypoints)
        imglist[3].append(des)
    return imglist


# Fast Library for Approximate Nearest Neighbors
def detect_speed(img):
    global IMAGES, FLANNTHRESHOLD, MIN_KEY_POINT

    """Run FLANN-detector for given image with given image list"""
    # Find the keypoint descriptors with SIFT
    _, des = SIFT.detectAndCompute(img, None)
    if des is None:
        return "Unknown", 0
    if len(des) < MIN_KEY_POINT:
        return "Unknown", 0

    biggest_amnt = 0
    biggest_speed = 0
    cur_img = 0
    try:
        for _ in IMAGES[0]:
            des2 = IMAGES[3][cur_img]
            matches = FLANN.knnMatch(des2, des, k=2)
            matchamnt = 0
            # Find matches with Lowe's ratio test
            for _, (moo, noo) in enumerate(matches):
                if moo.distance < FLANNTHRESHOLD * noo.distance:
                    matchamnt += 1
            if matchamnt > biggest_amnt:
                biggest_amnt = matchamnt
                biggest_speed = IMAGES[1][cur_img]
            cur_img += 1
        if biggest_amnt > MIN_KEY_POINT:
            return biggest_speed, biggest_amnt
        else:
            return "Unknown", 0
    except Exception as exept:
        print(str(exept))
        return "Unknown", 0


def run_logic():
    lastlimit = "00"
    lastdetect = "00"
    downscale = DOWNSCALE
    matches = 0
    try:
        if CAP.isOpened():
            rval, frame = CAP.read()
            print("Camera opened and frame read")
        else:
            rval = False
            print("Camera not opened")
        while rval:
            origframe = frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.equalizeHist(frame, frame)

            scaledsize = (int(frame.shape[1] / downscale), int(frame.shape[0] / downscale))

            scaledframe = cv2.resize(frame, scaledsize)

            # Detect speed signs
            speed_thread = SignDetector(SPEEDCLASSIFIER, scaledframe=scaledframe)
            speed_thread.start()
            # Threading to detect stop signs
            stop_thread = SignDetector(STOPCLASSIFIER, scaledframe=scaledframe)
            stop_thread.start()

            speed_signs = speed_thread.join()
            stop_signs = stop_thread.join()

            for stop in stop_signs:
                xpos, ypos, width, height = [i * downscale for i in stop]
                cv2.rectangle(
                    origframe,
                    (xpos, ypos),
                    (xpos + width, ypos + height),
                    (0, 0, 255))
                cv2.putText(
                    origframe,
                    "STOP !",
                    (xpos, ypos - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1, )

            for sign in speed_signs:
                xpos, ypos, width, height = [i * downscale for i in sign]

                crop_img = frame[ypos:ypos + height, xpos:xpos + width]
                sized = cv2.resize(crop_img, (128, 128))
                comp = "Unknown"

                comp, amnt = detect_speed(sized)
                if comp != "Unknown":
                    if comp != lastlimit:
                        if comp == lastdetect:
                            possiblematch = comp
                            matches = matches + 1
                            if matches >= N_MATCHES:
                                lastlimit = possiblematch
                                matches = 0
                        else:
                            matches = 0
                    cv2.rectangle(
                        origframe,
                        (xpos, ypos),
                        (xpos + width, ypos + height),
                        (255, 0, 0))
                    cv2.putText(
                        origframe,
                        "Speed limit: " + comp,
                        (xpos, ypos - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        1, )
                else:
                    comp = lastdetect
                lastdetect = comp
            cv2.putText(
                origframe,
                "Current speed limit: " + str(lastlimit) + " km/h.",
                (5, 50),
                cv2.QT_FONT_NORMAL,
                1,
                (0, 0, 0),
                2
            )

            if True:
                cv2.imshow("View", origframe)

            _ = cv2.waitKey(20)
            rval, frame = CAP.read()
    except (KeyboardInterrupt, Exception) as exept:
        print(str(exept))
        print("Shutting down!")


IMAGES = get_keyPoints("data")
# Scale-Invariant Feature Transform
SIFT = cv2.xfeatures2d.SIFT_create()
FLANN = None
CAP = None
MIN_KEY_POINT = 5
FLANNTHRESHOLD = 0.8
CHECKS = 50
TREES = 5
N_MATCHES = 2
DOWNSCALE = 1
CAM_INDEX = 0
if __name__ == "__main__":
    CAP = cv2.VideoCapture(CAM_INDEX)
    SPEEDCLASSIFIER = cv2.CascadeClassifier("speed_signs_lbp.xml")
    STOPCLASSIFIER = cv2.CascadeClassifier("stop_signs_lbp.xml")

    FLANN = cv2.FlannBasedMatcher(dict(algorithm=0, trees=TREES), dict(checks=CHECKS))

    run_logic()
