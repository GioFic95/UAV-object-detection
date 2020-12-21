import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QPixmap, QImage, QKeySequence, QFont
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QLineEdit, QComboBox, QShortcut, QMessageBox, \
    QInputDialog, QFileDialog, QSpinBox
from imutils import resize


# Adapted from:
# Sapienza Flight Team - Roma - User Interface - ODLC
# @Francesco Corallo  @Michiel Firlefyn


def add_grid(img):
    out = img.copy()
    cv2.line(out, (int(img.shape[1] / 3), 0), (int(img.shape[1] / 3), img.shape[0]), (255, 0, 0), 2)
    cv2.line(out, (int(img.shape[1]*2 / 3), 0), (int(img.shape[1]*2 / 3), img.shape[0]), (255, 0, 0), 2)
    cv2.line(out, (0, int(img.shape[0] / 3)), (img.shape[1], int(img.shape[0] / 3)), (255, 0, 0), 2)
    cv2.line(out, (0, int(img.shape[0]*2 / 3)), (img.shape[1], int(img.shape[0]*2 / 3)), (255, 0, 0), 2)
    return out


class App(QWidget):  # main window

    def __init__(self):
        super().__init__()
        self.title = 'Labeling GUI'
        self.left = 100
        self.top = 10
        self.width = 1200  # 640
        self.height = 900  # 480
        self.dialogs = list()
        self.selectDir()
        self.selectSave()
        self.setDirAndSave()
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
        # QMessageBox.about(self, "Info", help_message)
        self.info_label = QLabel(self)
        self.info_label.move(50, self.height - 50)
        self.info_label.setFont(QFont('Arial', 20))
        self.on_click_next()
        try:
            while pics[str(self.k)] in submitted:
                print("pic:", pics[str(self.k)])
                self.on_click_next()
        except KeyError:
            print("sorry")
            self.img_resized = np.zeros((self.height-100, self.width, 3), np.uint8)
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
        self.shortcut_prev = QShortcut(QKeySequence("Ctrl+Left"), self)
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
        self.shortcut_next = QShortcut(QKeySequence("Ctrl+Right"), self)
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

    def selectDir(self):
        try:
            global directoryPath
            print('Images folder set to: ' + directoryPath)
        except:
            print('')
            print('Please select objects images directory ')
            print('')
            directoryPath = str(
                QFileDialog.getExistingDirectory(self,
                                                 "Select Objects Images directory",
                                                 directory=os.getcwd()))
            print('Images folder set to: ' + directoryPath)

    def selectSave(self):
        try:
            global save_path
            print('Save file set to: ' + save_path)
        except:
            print('')
            print('Please select save file')
            print('')
            save_path = str(QFileDialog.getOpenFileName(self,
                                                        "Select results file",
                                                        directory=os.getcwd(),
                                                        filter="*.tsv")[0])
            print('Save file set to: ' + save_path)

    def setDirAndSave(self):
        global counter
        directory = os.fsencode(directoryPath)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            pics[str(counter)] = filename
            counter += 1
            continue

        # if the save dir isn't empty, load the names of the saved images
        global save_path
        if not os.path.exists(save_path):
            path, file = os.path.split(save_path)
            if not os.path.exists(path):
                if path != "":
                    os.mkdir(path)
                else:
                    if not os.path.exists("results"):
                        os.mkdir("results")
                    path = "results"
            if file == "":
                save_path = os.path.join(path, "results.tsv")
            else:
                save_path = os.path.join(path, file)
            # columns = "name\tshape\tshapeColor\talphanumeric\talphanumericColor\tboundingBox\trotation\n"
            columns = "name\tshape\tshapeColor\tboundingBox\trotation\n"
            with open(save_path, 'w') as save_file:
                save_file.write(columns)
        else:
            df = pd.read_csv(save_path, sep='\t')
            print(df.dtypes)
            names = df["name"].values
            global submitted
            submitted.update([name.split("_")[0] for name in names])
            print("submitted:", submitted)
        print("save_path:", save_path)

    def dialog(self):
        dialog = DialogApp(self.img_cropped, self.k)
        self.dialogs.append(dialog)
        dialog.show()

    def Crop(self, regions, img=None):
        if img is None:
            img = self.img_resized
        img_crop = img[int(regions[1]):int(regions[1] + regions[3]),
                       int(regions[0]):int(regions[0] + regions[2])]
        print("crop:", regions)
        return img_crop

    @pyqtSlot()
    def on_click_prev(self):
        print("Previous img:")
        if self.k - 1 == 0:
            print("This is the first image!")
            self.info_label.setText("First image!")
            self.info_label.setStyleSheet("background-color: red;")
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
        try:
            self.info_label.setText(pics[str(self.k)])
            self.info_label.setStyleSheet("background-color: transparent;")
        except KeyError:
            pass

    def on_click_auto(self):
        print("Automatic mode started")
        # my_autoROI = autoROI(self.image)
        # result = my_autoROI.start()
        # cv2.destroyAllWindows()
        # self.img_cropped = result
        # self.dialog()

    def on_click_man(self):
        print("Manual mode started")
        print('WARNING! Object must fill 25%+ of the cropped image!!! ')
        self.regions = cv2.selectROI(self.img_resized)
        cv2.destroyAllWindows()
        self.img_cropped = self.Crop(self.regions)
        if self.regions[2] == 0 and self.regions[3] == 0:
            print("self.regions empty")
            return
        self.dialog()

    def on_click_next(self):
        print("Next img:")
        if self.k + 1 == counter:
            print("No more images!")
            self.k += 1
            return
        else:
            corrupted_check = 1
            while corrupted_check == 1:
                if self.k == counter:
                    print("No more images !")
                    self.info_label.setText("No more images!")
                    self.info_label.setStyleSheet("background-color: red;")
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
        try:
            self.info_label.setText(pics[str(self.k)])
            self.info_label.setStyleSheet("background-color: transparent;")
        except KeyError:
            pass

    def on_click_zoom(self):
        temp_resized = resize(self.image, width=self.width)
        self.zoom = cv2.selectROI(temp_resized)
        cv2.destroyAllWindows()
        if self.zoom[2] == 0 and self.zoom[3] == 0:
            print("zoom regions empty")
            return
        self.img_cropped = self.Crop(self.zoom, img=temp_resized)
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
        self.rotation = 0

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # display the cropped image
        self.label1 = QLabel(self)
        self.label1.move(30, 30)
        try:  # check if automatic mode has found something displayable
            self.img_resized1 = resize(self.imgCrop, width=self.width)
            grid_img = add_grid(self.img_resized1)
            cvRGBImg1 = cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB)
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

            label_rotate = QLabel(self)  # letter color
            label_rotate.setText('Shape rotation:')
            label_rotate.move(0.05 * W1, 0.6 * H1)
            label_rotate.setStyleSheet("font: {}pt Comic Sans MS".format(0.03 * H1))
            self.textbox_rotation = QSpinBox(self)
            self.textbox_rotation.setRange(-360, 360)
            self.textbox_rotation.setWrapping(True)
            self.textbox_rotation.move(0.05 * W1, 0.7 * H1)
            self.textbox_rotation.valueChanged.connect(self.on_rotation_changed)

            # label_lcolor = QLabel(self)  # letter color
            # label_lcolor.setText('Letter Color:')
            # label_lcolor.move(0.4255 * W1, 0.0444 * H1)
            # label_lcolor.setStyleSheet("font: {}pt Comic Sans MS".format(0.03 * H1))

            # dropdown lcolor
            # self.drop_lcolor = QComboBox(self)
            # self.drop_lcolor.addItem("-")
            # self.drop_lcolor.addItem("BLACK")
            # self.drop_lcolor.addItem("BLUE")
            # self.drop_lcolor.addItem("BROWN")
            # self.drop_lcolor.addItem("GRAY")
            # self.drop_lcolor.addItem("GREEN")
            # self.drop_lcolor.addItem("ORANGE")
            # self.drop_lcolor.addItem("PURPLE")
            # self.drop_lcolor.addItem("RED")
            # self.drop_lcolor.addItem("WHITE")
            # self.drop_lcolor.addItem("YELLOW")
            # self.drop_lcolor.move(0.84 * W1, 0.03 * H1)
            # self.drop_lcolor.setStyleSheet('''* QComboBox QAbstractItemView { min-width: 100px;}''')

            label_bcolor = QLabel(self)  # background color
            label_bcolor.setText('Background Color:')
            label_bcolor.move(0.4255 * W1, 0.2 * H1)
            label_bcolor.setStyleSheet("font: {}pt Comic Sans MS".format(0.03 * H1))

            self.drop_bcolor = QComboBox(self)
            self.drop_bcolor.addItem("-")
            self.drop_bcolor.addItem("BLACK")
            self.drop_bcolor.addItem("BLUE")
            self.drop_bcolor.addItem("BROWN")
            self.drop_bcolor.addItem("GRAY")
            self.drop_bcolor.addItem("GREEN")
            self.drop_bcolor.addItem("ORANGE")
            self.drop_bcolor.addItem("PURPLE")
            self.drop_bcolor.addItem("RED")
            self.drop_bcolor.addItem("WHITE")
            self.drop_bcolor.addItem("YELLOW")
            self.drop_bcolor.move(0.84 * W1, 0.1856 * H1)
            self.drop_bcolor.setStyleSheet('''* QComboBox QAbstractItemView { min-width: 100px;}''')

            # label_letter = QLabel(self)  # letter
            # label_letter.setText('Alphanumerical:')
            # label_letter.move(0.4255 * W1, 0.3556 * H1)
            # label_letter.setStyleSheet("font: {}pt Comic Sans MS".format(0.03 * H1))

            # self.drop_letter = QComboBox(self)
            # self.drop_letter.addItem("-")
            # chars = "0123456789AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz"
            # for i in chars:  # only uppercase?
            #     self.drop_letter.addItem(i)
            # self.drop_letter.move(0.84 * W1, 0.3412 * H1)
            # self.drop_letter.setStyleSheet('''* QComboBox QAbstractItemView { min-width: 100px;}''')

            label_bshape = QLabel(self)  # background shape
            label_bshape.setText('Background Shape:')
            label_bshape.move(0.4255 * W1, 0.5111 * H1)
            label_bshape.setStyleSheet("font: {}pt Comic Sans MS".format(0.03 * H1))

            self.drop_bshape = QComboBox(self)
            self.drop_bshape.addItem("-")
            self.drop_bshape.addItem("CIRCLE")
            self.drop_bshape.addItem("CROSS")
            self.drop_bshape.addItem("HEPTAGON")
            self.drop_bshape.addItem("HEXAGON")
            self.drop_bshape.addItem("OCTAGON")
            self.drop_bshape.addItem("PENTAGON")
            self.drop_bshape.addItem("QUARTER_CIRCLE")
            self.drop_bshape.addItem("RECTANGLE")
            self.drop_bshape.addItem("SEMICIRCLE")
            self.drop_bshape.addItem("SQUARE")
            self.drop_bshape.addItem("STAR")
            self.drop_bshape.addItem("TRAPEZOID")
            self.drop_bshape.addItem("TRIANGLE")
            self.drop_bshape.move(0.84 * W1, 0.4967 * H1)
            self.drop_bshape.setStyleSheet('''* QComboBox QAbstractItemView { min-width: 100px;}''')

            button_submit = QPushButton("Submit", self)
            button_submit.setAutoDefault(True)
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

    def on_rotation_changed(self):
        rotation = self.textbox_rotation.value()
        self.img_resized1 = ndimage.rotate(self.img_resized1, rotation-self.rotation, reshape=False)
        grid_img = add_grid(self.img_resized1)
        cvRGBImg1 = cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB)
        qimg1 = QImage(cvRGBImg1.data, cvRGBImg1.shape[1], cvRGBImg1.shape[0], QImage.Format_RGB888)
        pixmap1 = QPixmap.fromImage(qimg1)
        self.label1.setPixmap(pixmap1)
        self.update()

        self.rotation = rotation
        # print('rotation:', self.rotation)

    def on_click_rotate(self):
        self.textbox_rotation.setValue((self.rotation+90) % 360)

    @pyqtSlot()
    def on_click_submit(self):
        empty = []

        # if '-' not in self.drop_lcolor.currentText():
        #     chosen_lcolor = self.drop_lcolor.currentText()
        # else:
        #     chosen_lcolor = ""
        #     empty += ["'Letter Color'"]
        #     print("no lcolor choosen")

        if '-' not in self.drop_bcolor.currentText():
            chosen_bcolor = self.drop_bcolor.currentText()
        else:
            chosen_bcolor = ""
            empty += ["'Background Color'"]
            print("no bcolor chosen")

        # if '-' not in self.drop_letter.currentText():
        #     chosen_letter = self.drop_letter.currentText()
        # else:
        #     chosen_letter = ""
        #     empty += ["'Alphanumerical'"]
        #     print("no char chosen")

        if '-' not in self.drop_bshape.currentText():
            chosen_bshape = self.drop_bshape.currentText().upper()
        else:
            chosen_bshape = ""
            empty += ["'Background Shape'"]
            print("no shape chosen")

        if empty:
            reply = QMessageBox.question(self, 'Empty Fields',
                                         f"The field(s) {', '.join(empty)} are still empty,\n"
                                         f"do you want to submit anyway?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return

        temp_resized = resize(ex.image, width=ex.width)

        # resized zoom to original zoom
        # ? : ex.zoom[2] = ex.regions[0] : temp_resized.shape[1]
        x = ex.zoom[2] * ex.regions[0] / ex.img_resized.shape[1]
        y = ex.zoom[3] * ex.regions[1] / ex.img_resized.shape[0]

        # ? : ex.regions[2] = ex.zoom[2] : temp_resized.shape[1]
        w = ex.zoom[2] * ex.regions[2] / ex.img_resized.shape[1]
        h = ex.zoom[3] * ex.regions[3] / ex.img_resized.shape[0]

        # resized image to original image
        # ? : (ex.zoom[0] + x) = ex.image.shape[0] : temp_resized.shape[0]
        # ? : w = ex.image.shape[0] : temp_resized.shape[0]
        regions = [
            (ex.zoom[0] + x) * ex.image.shape[1] / temp_resized.shape[1],
            (ex.zoom[1] + y) * ex.image.shape[0] / temp_resized.shape[0],
            w * ex.image.shape[1] / temp_resized.shape[1],
            h * ex.image.shape[0] / temp_resized.shape[0]
        ]
        print("shapes:", ex.image.shape, ex.img_resized.shape, temp_resized.shape)
        print("regions:\n", ex.zoom, "\n", ex.regions, "\n", regions)

        # odlc_tsv = f"{pics[str(ex.k)]}\t{chosen_bshape}\t{chosen_bcolor}\t{chosen_letter}\t{chosen_lcolor}\t" \
        odlc_tsv = f"{pics[str(ex.k)]}\t{chosen_bshape}\t{chosen_bcolor}\t" \
                   f"[{regions[0]},{regions[1]},{regions[0]+regions[2]},{regions[1]+regions[3]}]\t{self.rotation}\n"

        global submitcount
        submitcount += 1
        print("OBJECT SUBMITTED.     number of submitted objects: " + str(submitcount))
        with open(save_path, 'a') as save_file:
            save_file.write(odlc_tsv)

        self.close()


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
        self.label_letter = QLabel(self)
        self.label_letter.setText('modificato')
        self.label_letter.move(0.4255 * self.width, 0.1 * self.height)
        self.label_letter.setStyleSheet("font: {}pt Comic Sans MS".format(0.1 * self.height))
        self.update()


# python gui.py -dir ./test_img -save results/results.tsv
if __name__ == '__main__':
    # arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-dir', '--directory', required=False, help='Path to images directory')
    ap.add_argument('-save', '--save', required=False, help='Path to save directory')
    args = vars(ap.parse_args())

    # variables from arguments
    directoryPath = args['directory']
    save_path = args['save']

    pics = {}
    counter = 1

    imglst = []
    poslst = []

    submitted = set()
    submitcount = 0

    help_message = "If you want to continue your labeling work in a different moment, the script will resume directly " \
                   "from the next picture with respect to the last one from which you submitted some labels, so, " \
                   "remember to fill the labels for all the shapes in a photo before quitting.\n\nYou can use the esc " \
                   "key to quit (undo) from the zoom/regions dialogue and enter to confirm the selection.\n\nYou can " \
                   "use the tab key to move across fields, to fill them faster, and up/down arrows to " \
                   "increment/decrement the rotation.\n\nShortcuts:\n" \
                   "ctrl+M = manual\n" \
                   "ctrl+Left = previous\n" \
                   "ctrl+Right = next\n" \
                   "ctrl+S = submit\n" \
                   "ctrl+R = rotate\n" \
                   "ctrl+W = close window\n" \
                   "Check readme.md for more\n\n"

    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
