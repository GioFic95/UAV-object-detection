import os

import cv2 as cv


def crop_image(in_path, out_path, name):
    # legge immagine
    img = cv.imread(in_path)
    # cv.imshow('Image', img)
    print("Dimension:", img.shape)

    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # trova contorni
    _, thresh = cv.threshold(imgray, 127, 255, 0)
    # cv.imshow('Imagebn', thresh)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print("Num contours:", len(contours))

    # filtra zone troppo piccole o troppo grandi
    min_area = 100
    max_area = 5000
    contours = [c for c in contours if min_area <= cv.contourArea(c) <= max_area]
    print("Num contours after filtering:", len(contours))

    # ordina i contorni in base alla dimensione dell'area
    contours = sorted(contours, key=cv.contourArea, reverse=True)

    # estrae ROI
    rois = []
    for i, c in enumerate(contours):
        x,y,w,h = cv.boundingRect(c)
        h0 = (244 - h)/2   # y
        w0 = (244 - w)/2   # x

        b_margin = y + h + h0 - img.shape[0]
        t_margin = h0 - y
        r_margin = x + w + w0 - img.shape[1]
        l_margin = w0 - x

        x0 = int(x - w0 - r_margin if r_margin > 0 else x - w0 if l_margin <= 0 else x - w0 + l_margin)
        x1 = int(x + w + w0 + l_margin if l_margin > 0 else x + w + w0 if r_margin <= 0 else x + w + w0 - r_margin)
        y0 = int(y - h0 - b_margin if b_margin > 0 else y - h0 if t_margin <= 0 else y - h0 + t_margin)
        y1 = int(y + h + h0 + t_margin if t_margin > 0 else y + h + h0 if b_margin <= 0 else y + h + h0 - b_margin)
        print(name, i, h0, w0, b_margin, t_margin, r_margin, l_margin, y, y0, y1, x, x0, x1)
        rois += [((x0, y0), (x1, y1))]
        roi = img[y0:y1, x0:x1]
        print(roi.shape)
        # cv.imshow('ROI', roi)
        # cv.waitKey(0) & 0xFF
        file_name = os.path.join(out_path, "crop_" + str(name) + "_" + str(i) + ".png")
        cv.imwrite(file_name, roi)

    return rois
