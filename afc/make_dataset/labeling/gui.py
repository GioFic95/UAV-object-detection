import argparse
import json
import os
import shutil
import sys
import datetime

import cv2
import numpy as np
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QPixmap, QImage, QKeySequence
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QLineEdit, QComboBox, QShortcut, QMessageBox, \
    QInputDialog, QFileDialog
from auto_mode import autoROI
from imutils import resize, rotate


# Adapted from:
# Sapienza Flight Team - Roma - User Interface - ODLC
# @Francesco Corallo  @Michiel Firlefyn


class App(QWidget):  # main window

    def __init__(self):
        super().__init__()
        self.title = 'SFT User Interface - AUVSI SUAS'
        self.left = 100
        self.top = 10
        self.width = 640
        self.height = 480
        self.dialogs = list()
        self.selectDir()
        self.setDirAndBackup()
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # create widget
        self.label = QLabel(self)
        self.k = 0
        print('')
        print('')
        print("Start.")
        print('')
        QMessageBox.about(self, "Info", help_message)
        corrupted_check = 1
        self.on_click_next()
        try:
            self.img_resized
        except:
            self.img_resized = np.zeros((640, 640, 3), np.uint8)
        imglst.append(self.img_resized)
        cvRGBImg = cv2.cvtColor(self.img_resized, cv2.COLOR_BGR2RGB)
        qimg = QImage(cvRGBImg.data, cvRGBImg.shape[1], cvRGBImg.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.label.setPixmap(pixmap)
        W, H = pixmap.width(), pixmap.height()
        H += 100  # adds white space for buttons and text
        self.resize(W, H)

        pos = True
        poslst.append(pos)

        # add buttons
        button_prev = QPushButton("<-", self)
        button_prev.setToolTip("Previous image")
        button_prev.move(0.30 * W, H - 50)
        button_prev.clicked.connect(self.on_click_prev)
        self.shortcut_prev = QShortcut(QKeySequence("Ctrl+è"), self)
        self.shortcut_prev.activated.connect(self.on_click_prev)

        button_auto = QPushButton("AUTO (beta)", self)
        button_auto.setToolTip("automatic detection mode")
        button_auto.move(0.42 * W, H - 50)
        button_auto.clicked.connect(self.on_click_auto)
        self.shortcut_auto = QShortcut(QKeySequence("Ctrl+A"), self)
        self.shortcut_auto.activated.connect(self.on_click_auto)

        button_manual = QPushButton("MANUAL", self)
        button_manual.setToolTip("manual detection mode")
        button_manual.move(0.4375 * W, H - 80)
        button_manual.clicked.connect(self.on_click_man)
        self.shortcut_man = QShortcut(QKeySequence("Ctrl+M"), self)
        self.shortcut_man.activated.connect(self.on_click_man)

        button_next = QPushButton("->", self)
        button_next.setToolTip("Next image")
        button_next.move(0.63 * W, H - 50)
        button_next.clicked.connect(self.on_click_next)
        self.shortcut_next = QShortcut(QKeySequence("Ctrl++"), self)
        self.shortcut_next.activated.connect(self.on_click_next)

        button_zoom = QPushButton("ZOOM", self)
        button_zoom.setToolTip("zoom a portion of the image")
        button_zoom.move(0 * W, H - 100)
        button_zoom.clicked.connect(self.on_click_zoom)
        self.shortcut_zoom = QShortcut(QKeySequence("Ctrl+z"), self)
        self.shortcut_zoom.activated.connect(self.on_click_zoom)

        button_lighten = QPushButton("LIGHTEN", self)
        button_lighten.setToolTip("lighten the image")
        button_lighten.move(0.12 * W, H - 100)
        button_lighten.clicked.connect(self.on_click_lighten)
        self.shortcut_lighten = QShortcut(QKeySequence("Ctrl+l"), self)
        self.shortcut_lighten.activated.connect(self.on_click_lighten)

        button_darken = QPushButton("DARKEN", self)
        button_darken.setToolTip("darken the image")
        button_darken.move(0.26 * W, H - 100)
        button_darken.clicked.connect(self.on_click_darken)
        self.shortcut_darken = QShortcut(QKeySequence("Ctrl+d"), self)
        self.shortcut_darken.activated.connect(self.on_click_darken)

        button_help = QPushButton("help", self)
        button_help.setToolTip("display help")
        button_help.move(0.9 * W, H - 100)
        button_help.clicked.connect(self.on_click_help)

        self.shortcut_close = QShortcut(QKeySequence("Ctrl+w"), self)
        self.shortcut_close.activated.connect(self.close)

    # self.dialog_submitted = DialogSubmitted()
    # self.dialogs.append(self.dialog_submitted)
    # self.dialog_submitted.show()

    def selectDir(self):
        try:
            global directoryPath
            print('Images folder set to: ' + directoryPath)
        except:
            # fileName = QFileDialog.getOpenFileName(self, 'OpenFile')
            print('')
            print('Please select objects images directory ')
            print('')
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            directoryPath = str(
                QFileDialog.getExistingDirectory(self, "Select Objects Images directory", options=options))
            # self.myTextBox.setText(fileName)
            print('Images folder set to: ' + directoryPath)

    def setDirAndBackup(self):
        global counter
        global backup_num
        directory = os.fsencode(directoryPath)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            pics[str(counter)] = filename  # pics['id: ' + str(counter)] = filename
            counter += 1
            continue

        # num = 1
        # code for creating a new backup instead of overwriting the old one
        if os.path.exists(directoryPath + '/../backup'):  # + str(num)):
            ask = 0
            while ask == 0:
                buttonReply = QMessageBox.question(self, 'PyQt5 message',
                                                   "A backup folder exist. Do you want to use it?",
                                                   QMessageBox.No | QMessageBox.Yes)  # , QMessageBox.No)
                # ask = input('\nA backup folder exist. Do you want to use it? answer: y or n \n')
                if buttonReply == QMessageBox.Yes:
                    # num = input('input the number of the desired backup folder:\n')
                    backup_num = '/../backup'  # + str(num)
                    ask = 1
                    backup_used = 1
                elif buttonReply == QMessageBox.No:
                    shutil.rmtree(directoryPath + '/../backup')  # + str(num))
                    os.mkdir(directoryPath + '/../backup')  # + str(num))
                    backup_num = '/../backup'  # + str(num)
                    ask = 1
                    backup_used = 0
                else:
                    print('answer y or n')
                    ask = 0
        else:
            os.mkdir(directoryPath + '/../backup')  # + str(num))
            backup_num = '/../backup'  # + str(num)

        backup_directory = os.fsencode(directoryPath + backup_num)
        for file in os.listdir(backup_directory):
            filename = os.fsdecode(file)
            if 'json' in filename:
                current_file = open(directoryPath + backup_num + '/' + filename)
                current_json = json.load(current_file)
                # current_json_wout_id.pop('id')
                backup_jsons.append(current_json)

    def dialog(self):
        dialog = DialogApp(self.img_cropped, self.k)
        self.dialogs.append(dialog)
        dialog.show()

    def Crop(self, regions):
        # if False in poslst:
        #	ind = len(poslst) - 1 - poslst[::-1].index(False)
        #	imgCrop = imglst[ind][int(regions[1]):int(regions[1]+regions[3]), int(regions[0]):int(regions[0]+regions[2])]
        # else:
        #	imgCrop = imglst[-1][int(regions[1]):int(regions[1]+regions[3]), int(regions[0]):int(regions[0]+regions[2])]
        img_crop = self.img_resized[int(regions[1]):int(regions[1] + regions[3]),
                   int(regions[0]):int(regions[0] + regions[2])]
        print("crop:", regions)
        return img_crop

    @pyqtSlot()
    def on_click_prev(self):
        print("Previous img:")
        if self.k - 1 == 0:
            print("This is the first image!")
            return
        else:
            corrupted_check = 1
            while corrupted_check == 1:
                if self.k <= 0:
                    print("This is the first image!")
                    # self.k = 1
                    break
                else:
                    self.k -= 1
                    try:
                        self.image = cv2.imread(directoryPath + '/' + pics[str(self.k)])  # 'id: ' +
                        self.img_resized = resize(self.image, width=self.width)  # adds white space for buttons and text
                        corrupted_check = 0
                    except:
                        print('skipping img %d: not an image or not-readable image' % self.k)
                        corrupted_check = 1

        print("current = %d . remaining: %d. submitted: %d" % (self.k, counter - 1 - self.k, submitcount))
        print('')
        pos = False
        poslst.append(pos)
        imglst.append(self.img_resized)
        cvRGBImg = cv2.cvtColor(self.img_resized, cv2.COLOR_BGR2RGB)
        qimg = QImage(cvRGBImg.data, cvRGBImg.shape[1], cvRGBImg.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.label.setPixmap(pixmap)

    def on_click_auto(self):
        print("Automatic mode started")
        my_autoROI = autoROI(self.image)
        result = my_autoROI.start()
        cv2.destroyAllWindows()
        self.img_cropped = result
        self.dialog()

    def on_click_man(self):
        print("Manual mode started")
        print('WARNING! Object must fill 25%+ of the cropped image!!! ')
        self.regions = cv2.selectROI(self.img_resized)
        cv2.destroyAllWindows()
        # self.regions = deepcopy(r)
        self.img_cropped = self.Crop(self.regions)
        self.dialog()

    def on_click_next(self):
        print("Next img:")
        if self.k + 1 == counter:
            print("No more images!")
            return
        else:
            corrupted_check = 1
            while corrupted_check == 1:
                if self.k == counter:
                    print("No more images !")
                    # self.k = counter-1
                    break
                else:
                    self.k += 1
                    try:
                        self.image = cv2.imread(directoryPath + '/' + pics[str(self.k)])  # 'id: ' +
                        self.img_resized = resize(self.image, width=self.width)  # adds white space for buttons and text
                        corrupted_check = 0
                    except:
                        print('skipping img %d: not an image or not-readable image' % self.k)
                        corrupted_check = 1

        print("current = %d . remaining: %d. submitted: %d" % (self.k, counter - 1 - self.k, submitcount))
        print('')
        pos = False
        poslst.append(pos)
        imglst.append(self.img_resized)
        cvRGBImg = cv2.cvtColor(self.img_resized, cv2.COLOR_BGR2RGB)
        qimg = QImage(cvRGBImg.data, cvRGBImg.shape[1], cvRGBImg.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.label.setPixmap(pixmap)

    def on_click_zoom(self):
        self.regions = cv2.selectROI(self.img_resized)
        cv2.destroyAllWindows()
        self.img_cropped = self.Crop(self.regions)
        self.img_resized = resize(self.img_cropped, width=self.width)
        cvRGBImg = cv2.cvtColor(self.img_resized, cv2.COLOR_BGR2RGB)
        qimg = QImage(cvRGBImg.data, cvRGBImg.shape[1], cvRGBImg.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.label.setPixmap(pixmap)

    def on_click_lighten(self):
        M = np.ones(self.img_resized.shape, dtype="uint8") * 50
        self.added = cv2.add(self.img_resized, M)
        self.img_resized = resize(self.added, width=self.width)
        cvRGBImg = cv2.cvtColor(self.img_resized, cv2.COLOR_BGR2RGB)
        qimg = QImage(cvRGBImg.data, cvRGBImg.shape[1], cvRGBImg.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.label.setPixmap(pixmap)

    def on_click_darken(self):
        M = np.ones(self.img_resized.shape, dtype="uint8") * 50
        self.subtracted = cv2.subtract(self.img_resized, M)
        self.img_resized = resize(self.subtracted, width=self.width)
        cvRGBImg = cv2.cvtColor(self.img_resized, cv2.COLOR_BGR2RGB)
        qimg = QImage(cvRGBImg.data, cvRGBImg.shape[1], cvRGBImg.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.label.setPixmap(pixmap)

    def on_click_help(self):
        QMessageBox.about(self, "Info", help_message)


class DialogApp(QWidget):  # Dialog window with cropped image

    def __init__(self, img_cropped, k):
        super().__init__()
        self.title = 'Insert characteristics manually'
        self.left = 100
        self.top = 100
        self.width = 160
        self.height = 120
        self.imgCrop = img_cropped
        self.initUI()
        self.k = k

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # display the cropped image
        self.label1 = QLabel(self)
        self.label1.move(30, 30)
        try:  # check if automatic mode has found something displayable
            self.img_resized1 = resize(self.imgCrop, width=self.width)
            cvRGBImg1 = cv2.cvtColor(self.img_resized1, cv2.COLOR_BGR2RGB)
            qimg1 = QImage(cvRGBImg1.data, cvRGBImg1.shape[1], cvRGBImg1.shape[0], QImage.Format_RGB888)
            pixmap1 = QPixmap.fromImage(qimg1)
            self.label1.setPixmap(pixmap1)
            W1, H1 = pixmap1.width(), pixmap1.height()
            H1 += 300  # adds white space for buttons and text
            W1 += 600
            self.resize(W1, H1)
            # print(W1, H1)

            # rotate
            button_rotate = QPushButton("rotate", self)
            button_rotate.setToolTip("rotate image 90 deg")
            button_rotate.move(0.05 * W1, 0.5 * H1)
            button_rotate.clicked.connect(self.on_click_rotate)
            self.shortcut_rotate = QShortcut(QKeySequence("Ctrl+r"), self)
            self.shortcut_rotate.activated.connect(self.on_click_rotate)

            label_lcolor = QLabel(self)  # letter color
            label_lcolor.setText('Letter Color:')
            label_lcolor.move(0.4255 * W1, 0.0444 * H1)
            label_lcolor.setStyleSheet("font: {}pt Comic Sans MS".format(0.03 * H1))

            self.textbox_lcolor = QLineEdit(self)
            self.textbox_lcolor.move(0.65 * W1, 0.0444 * H1)
            # textbox_color.resize(100,40)

            # dropdown lcolor
            self.drop_lcolor = QComboBox(self)
            self.drop_lcolor.addItem("-")
            self.drop_lcolor.addItem("WHITE")
            self.drop_lcolor.addItem("BLACK")
            self.drop_lcolor.addItem("GRAY")
            self.drop_lcolor.addItem("RED")
            self.drop_lcolor.addItem("BLUE")
            self.drop_lcolor.addItem("GREEN")
            self.drop_lcolor.addItem("YELLOW")
            self.drop_lcolor.addItem("PURPLE")
            self.drop_lcolor.addItem("BROWN")
            self.drop_lcolor.addItem("ORANGE")
            self.drop_lcolor.move(0.84 * W1, 0.03 * H1)
            self.drop_lcolor.setStyleSheet('''* QComboBox QAbstractItemView { min-width: 100px;}''')
            # comboBox.activated[str].connect(self.update_fields)
            ## Colors

            label_bcolor = QLabel(self)  # background color
            label_bcolor.setText('Background Color:')
            label_bcolor.move(0.4255 * W1, 0.2 * H1)
            label_bcolor.setStyleSheet("font: {}pt Comic Sans MS".format(0.03 * H1))

            self.textbox_bcolor = QLineEdit(self)
            self.textbox_bcolor.move(0.65 * W1, 0.2 * H1)
            # textbox_color.resize(100,40)

            self.drop_bcolor = QComboBox(self)
            self.drop_bcolor.addItem("-")
            self.drop_bcolor.addItem("WHITE")
            self.drop_bcolor.addItem("BLACK")
            self.drop_bcolor.addItem("GRAY")
            self.drop_bcolor.addItem("RED")
            self.drop_bcolor.addItem("BLUE")
            self.drop_bcolor.addItem("GREEN")
            self.drop_bcolor.addItem("YELLOW")
            self.drop_bcolor.addItem("PURPLE")
            self.drop_bcolor.addItem("BROWN")
            self.drop_bcolor.addItem("ORANGE")
            self.drop_bcolor.move(0.84 * W1, 0.1856 * H1)
            self.drop_bcolor.setStyleSheet('''* QComboBox QAbstractItemView { min-width: 100px;}''')

            label_letter = QLabel(self)  # letter
            label_letter.setText('Alphanumerical:')
            label_letter.move(0.4255 * W1, 0.3556 * H1)
            label_letter.setStyleSheet("font: {}pt Comic Sans MS".format(0.03 * H1))

            self.textbox_letter = QLineEdit(self)
            self.textbox_letter.move(0.65 * W1, 0.3556 * H1)
            self.textbox_letter.textChanged.connect(self.on_text_changed)
            # textbox_color.resize(100,40)df

            self.drop_letter = QComboBox(self)
            self.drop_letter.addItem("-           ")
            self.drop_letter.addItem("A")
            self.drop_letter.addItem("B")
            self.drop_letter.addItem("C")
            self.drop_letter.addItem("D")
            self.drop_letter.addItem("E")
            self.drop_letter.addItem("F")
            self.drop_letter.addItem("G")
            self.drop_letter.addItem("H")
            self.drop_letter.addItem("I")
            self.drop_letter.addItem("J")
            self.drop_letter.addItem("K")
            self.drop_letter.addItem("L")
            self.drop_letter.addItem("M")
            self.drop_letter.addItem("N")
            self.drop_letter.addItem("O")
            self.drop_letter.addItem("P")
            self.drop_letter.addItem("Q")
            self.drop_letter.addItem("R")
            self.drop_letter.addItem("S")
            self.drop_letter.addItem("T")
            self.drop_letter.addItem("U")
            self.drop_letter.addItem("V")
            self.drop_letter.addItem("W")
            self.drop_letter.addItem("X")
            self.drop_letter.addItem("Y")
            self.drop_letter.addItem("Z")
            self.drop_letter.move(0.84 * W1, 0.3412 * H1)
            self.drop_letter.setStyleSheet('''* QComboBox QAbstractItemView { min-width: 100px;}''')

            label_bshape = QLabel(self)  # background shape
            label_bshape.setText('Background Shape:')
            label_bshape.move(0.4255 * W1, 0.5111 * H1)
            label_bshape.setStyleSheet("font: {}pt Comic Sans MS".format(0.03 * H1))

            self.textbox_bshape = QLineEdit(self)
            self.textbox_bshape.move(0.65 * W1, 0.5111 * H1)
            # textbox_color.resize(100,40)

            self.drop_bshape = QComboBox(self)
            self.drop_bshape.addItem("-")
            self.drop_bshape.addItem("CIRCLE")
            self.drop_bshape.addItem("SEMICIRCLE")
            self.drop_bshape.addItem("QUARTER_CIRCLE")
            self.drop_bshape.addItem("TRIANGLE")
            self.drop_bshape.addItem("SQUARE")
            self.drop_bshape.addItem("RECTANGLE")
            self.drop_bshape.addItem("TRAPEZOID")
            self.drop_bshape.addItem("PENTAGON")
            self.drop_bshape.addItem("HEXAGON")
            self.drop_bshape.addItem("HEPTAGON")
            self.drop_bshape.addItem("OCTAGON")
            self.drop_bshape.addItem("STAR")
            self.drop_bshape.addItem("CROSS")
            self.drop_bshape.move(0.84 * W1, 0.4967 * H1)
            self.drop_bshape.setStyleSheet('''* QComboBox QAbstractItemView { min-width: 100px;}''')

            button_submit = QPushButton("Submit", self)
            button_submit.setToolTip("Submit results")
            button_submit.move(0.7895 * W1, 0.6667 * H1)
            button_submit.resize(0.1316 * W1, 0.2222 * H1)
            button_submit.clicked.connect(self.on_click_submit)
            self.shortcut_submit = QShortcut(QKeySequence("Ctrl+s"), self)
            self.shortcut_submit.activated.connect(self.on_click_submit)
        except:
            print('cannot display image')
            print('')
        # self.close()

        self.shortcut_close = QShortcut(QKeySequence("Ctrl+w"), self)
        self.shortcut_close.activated.connect(self.close)

    def Action1(self):
        print("Red selected")

    def on_text_changed(self):
        # print('text_ changed')
        pass

    def on_click_rotate(self):
        self.imgCrop = rotate(self.imgCrop, 90)
        self.img_resized1 = resize(self.imgCrop, width=self.width)
        cvRGBImg1 = cv2.cvtColor(self.img_resized1, cv2.COLOR_BGR2RGB)
        qimg1 = QImage(cvRGBImg1.data, cvRGBImg1.shape[1], cvRGBImg1.shape[0], QImage.Format_RGB888)
        pixmap1 = QPixmap.fromImage(qimg1)
        self.label1.setPixmap(pixmap1)
        self.update()

    @pyqtSlot()
    def on_click_submit(self):
        if not '-' in self.drop_bshape.currentText():
            chosen_bshape = self.drop_bshape.currentText().upper()
        else:
            chosen_bshape = self.textbox_bshape.text().upper()
            if chosen_bshape == "":
                chosen_bshape = None

        if '-' not in self.drop_bcolor.currentText():
            chosen_bcolor = self.drop_bcolor.currentText()
        else:
            chosen_bcolor = self.textbox_bcolor.text().upper()
            if chosen_bcolor == "":
                chosen_bcolor = None

        if '-' not in self.drop_letter.currentText():
            chosen_letter = self.drop_letter.currentText()
        else:
            chosen_letter = self.textbox_letter.text().upper()
            if chosen_letter == "":
                chosen_letter = None

        if '-' not in self.drop_lcolor.currentText():
            chosen_lcolor = self.drop_lcolor.currentText()
        else:
            chosen_lcolor = self.textbox_lcolor.text().upper()
            if chosen_lcolor == "":
                chosen_lcolor = None

        odlc_dict = {
            # todo: name, position, rotation...
            'shape': chosen_bshape,
            'shapeColor': chosen_bcolor,
            'alphanumeric': chosen_letter,
            'alphanumericColor': chosen_lcolor}
        odlc_json = json.dumps(odlc_dict)

        # check with local submitted list, and with backup if object has been yet submitted
        if (not odlc_json in submitted) and (not odlc_json in backup_jsons):
            global submitcount
            submitcount += 1
            print("OBJECT SUBMITTED.     number of submitted objects: " + str(submitcount))
            submitted.append(odlc_json)
            # save to backup folder:
            out = directoryPath + backup_num + '/' + str(len(backup_jsons)+submitcount) + "_" + ".json"
            json_backup = open(out, "w")
            json_backup.write(odlc_json)

        elif odlc_json in submitted:
            print('This object has been already submitted !')

        elif odlc_json in backup_jsons:
            print('This object is present in the backup folder')

        # App().dialog_submitted.refresh_window(self.img_resized1)  # todo reset whole image after zoom ?
        self.close()

    def decode_error(self, response):
        print(response)
        if '400' in str(response):
            print('The request was bad/invalid, the server does not know how to respond to such a request')
        elif '401' in str(response):
            print('The request is unauthorized')
        elif '403' in str(response):
            print('The request is forbidden')
        elif '404' in str(response):
            print('The request was made to an invalid URL')
        elif '405' in str(response):
            print('The request used an invalid method (e.g., GET when only POST is supported')
        elif '500' in str(response):
            print(
                'The server encountered an internal error and was unable to process the request. This indicates a configuration error on the server side.')


class DialogSubmitted(QWidget):  # Dialog window with cropped image

    def __init__(self):
        super().__init__()
        self.title = 'Submitted objects'
        self.left = 800
        self.top = 100
        self.width = 300
        self.height = 300
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.label_letter = QLabel(self)
        self.label_letter.setText('Prova')
        self.label_letter.move(0.4255 * self.width, 0.3556 * self.height)
        self.label_letter.setStyleSheet("font: {}pt Comic Sans MS".format(0.1 * self.height))

    def refresh_window(self, image):
        # self.label = QLabel(self)
        # self.label.move(30, 30)
        # self.img_resized = resize(image, 10)
        # cvRGBImg = cv2.cvtColor(self.img_resized, cv2.COLOR_BGR2RGB)
        # qimg = QImage(cvRGBImg.data, cvRGBImg.shape[1], cvRGBImg.shape[0], QImage.Format_RGB888)
        # pixmap = QPixmap.fromImage(qimg)
        # self.label.setPixmap(pixmap)

        self.label_letter = QLabel(self)
        self.label_letter.setText('modificato')
        self.label_letter.move(0.4255 * self.width, 0.1 * self.height)
        self.label_letter.setStyleSheet("font: {}pt Comic Sans MS".format(0.1 * self.height))
        self.update()


# python gui.py -dir ./test_img
if __name__ == '__main__':
    # arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-dir', '--directory', required=False, help='Path to images directory')
    args = vars(ap.parse_args())

    # variables from arguments
    directoryPath = args['directory']

    pics = {}
    counter = 1

    imglst = []
    poslst = []

    submitted = []
    submitcount = 0

    backup_jsons = []

    backup_used = 0

    # login
    # session = requests.Session()
    # session.post('http://' + url + '/api/login', data=json.dumps({'username': username, 'password': password}))
    # cookie = session.cookies

    help_message = """Rules:\nObject must fill 25%+ of the cropped image.\n
    Tips:
    - If the program crashes for some reason, you can restart it and it automatically detects an existing backup folder
    and asks if you want to use it for checking already submitted objects;
    - Also, to save time, if you don't remember if an object has been already submitted, you can manually check in the
    backup folder;
    - The backup folder is also useful if something goes wrong: copy it on a USB and give it to judges;
    - When filling object: the dropdown menù has precedence over the text input;
    - You can also leave blank fields if you fail detecting some details.\n
    Shortcuts:
    ctrl+M = manual
    ctrl+A = auto
    ctrl+[ = previous
    ctrl+] = next
    ctrl+S = submit
    ctrl+R = rotate
    ctrl+W = close window\n
    Check README.txt for more\n\n"""

    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
