# the method to find curves/contours in an image
from __future__ import print_function
import numpy as np
# import convolution
import imutils
import argparse
import cv2


class autoROI():
    def __init__(self, img):
        self.k = 1
        self.image = img
        cv2.imshow("Image", self.image)

    def start(self):
        ROI = []
        blur = 11
        while not ROI:  # while ROI is empty
            print('Blurring constant = %d' % blur)
            print(ROI)
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (blur, blur), 0)
            cv2.imshow("Blurred", blurred)
            # print(self.image.shape[:2])

            edged = cv2.Canny(blurred, 30, 150)
            cv2.imshow("Edges", edged)

            # find the contours of the image
            # tuples that findContours returns: image after applying contour detection, contours themselves, hierarchy of the contours
            # parameters: image (copy because findContours will destroy this), type of contours (following the outline), how the contours are approximated (with compression)
            cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            print("Found {} elements in this image".format(len(cnts)))

            elements = self.image.copy()
            # draw the found contours
            # parameters: image we want to draw on, list of contours, -1 means all of the contours, color of the line, thickness of drawn line
            cv2.drawContours(elements, cnts, -1, (0, 255, 0), 2)
            cv2.imshow("Elements", elements)
            print('DEBUG: select the window and click a key to proceed')
            cv2.waitKey(0)

            # crop each individual element
            for (i, e) in enumerate(cnts):
                (x, y, w, h) = cv2.boundingRect(e)  # finds enclosing box that fits the contour

                # print("Element #{}".format(i + 1))
                element = self.image[y - 13:y + h + 13, x - 13:x + w + 13]
                # cv2.imshow("Element", element)

                mask = np.zeros(self.image.shape[:2], dtype="uint8")
                # maybe needs to be changed to enclosing box
                ((centerX, centerY), radius) = cv2.minEnclosingCircle(e)  # fits a circle to the contour
                cv2.circle(mask, (int(centerX), int(centerY)), int(radius), 255, -1)
                mask = mask[y - 2:y + h + 2, x - 2:x + w + 2]
                # cv2.imshow("Masked Element", cv2.bitwise_and(element, element, mask = mask))
                # neglect elements that are too big to be a target
                # if element.shape[0] < 51 and element.shape[0] > 5:
                ROI.append(element)
            # cv2.waitKey(0)

            blur = blur - 2
            if blur == 1:
                cv2.destroyAllWindows()

        # resized = imutils.resize(ROI[0], width = 100, height = 100)
        # kernel to sharpen
        # kernel = np.array((
        #	[0, -1, 0],
        #	[-1, 5, -1],
        #	[0, -1, 0]), dtype="int")
        # opencvOutput = resized.copy()
        # for i in range(50):
        # convoleOutput = convolution.convolve(resized, kernel)
        #	opencvOutput = cv2.filter2D(opencvOutput, -1, kernel)
        # cv2.imshow("Resized using function", opencvOutput)
        # cv2.imshow("chosen", resized)
        # cv2.waitKey(0)
        print('')
        return ROI[0]
        cv2.destroyAllWindows()
