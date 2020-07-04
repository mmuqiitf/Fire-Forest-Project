import sys
import cv2
import numpy as np
import math
import xlsxwriter
from konvolusi import convolve as conv
from itertools import product
import csv
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow, QFileDialog
from PyQt5.uic import loadUi
from matplotlib import pyplot as plt

class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('project.ui', self)
        self.image = None
        self.image_contrast = None
        self.image_output = None
        self.btnLoad.clicked.connect(self.loadClicked)
        self.btnSave.clicked.connect(self.saveClicked)
        self.btnIdentifikasi.clicked.connect(self.detectProcess)
        self.actionMean_Filter.triggered.connect(self.meanClicked)

    @pyqtSlot()
    def loadClicked(self):
        flname, filter = QFileDialog.getOpenFileName(self, 'Open File', 'C:\\', "Image Files (*.png *.jpg *.jpeg)")
        if flname:
            self.loadImage(flname)
        else:
            print('Invalid Image')

    def loadImage(self, flname):
        self.image = cv2.imread(flname, cv2.IMREAD_COLOR)
        img = self.image
        # self.exportCSV(img, 'array_image')
        self.displayImage()

    def contrast(self, img):
        contrast = 1.4
        h, w = img.shape[:2]
        for i in np.arange(h):
            for j in np.arange(w):
                a = img.item(i, j)
                b = math.ceil(a * contrast)
                if b > 255:
                    b = 255
                elif b < 0:
                    b = 0
                else:
                    b = b
                img.itemset((i, j), b)
        self.image_contrast = img
        cv2.imshow("Contrast", self.image_contrast)

    def meanClicked(self):
        img = self.image
        kernel = np.ones((3, 3), np.float32) / 9
        mean = cv2.filter2D(self.image, -1, kernel)
        self.image = mean
        cv2.imshow("Mean", mean)
        cv2.imshow("Original Image", img)

    def detectProcess(self):
        # Hasil mean filtering
        img = self.image
        # Ubah mode ke HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cv2.imshow("HSV", hsb)
        # Ambang batas untuk warna Red
        lower = np.array([15, 31, 111], dtype="uint8")
        upper = np.array([90, 255, 255], dtype="uint8")
        # Thresholding dengan ambang batas
        mask = cv2.inRange(hsv, lower, upper)
        output = cv2.bitwise_and(img, hsv, mask=mask)
        # Ubah mode ke greyscale
        h, w = img.shape[:2]
        gray = np.zeros((h, w), np.uint8)
        for i in range(h):
            for j in range(w):
                gray[i, j] = np.clip(
                    0.299 * output[i, j, 0]
                    + 0.587 * output[i, j, 1]
                    + 0.114 * output[i, j, 2],
                    0,
                    255,
                )
        self.image = output
        cv2.imshow("Greyscale", gray)
        # Contrast
        self.contrast(gray)
        # Sobel Edge Detection
        self.sobelDetection(self.image_contrast)
        no_red = cv2.countNonZero(mask)
        print('Node Red : ' + str(no_red))
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1.5
        color = (255, 255, 255)
        thickness = 2
        if int(no_red) >= 20000:
            print('Fire detected')
            cv2.putText(output, "Fire Detected", org, font, fontScale, color, thickness, cv2.LINE_AA)
        else:
            print('Non Fire')
            cv2.putText(output, "Non Fire", org, font, fontScale, color, thickness, cv2.LINE_AA)

        self.displayImage(2)
        cv2.waitKey(0)

    def sobelDetection(self, img):
        print(img)
        Sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        Sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        img_x = conv(img, Sx)
        img_y = conv(img, Sy)
        img_out = np.sqrt(img_x * img_x + img_y * img_y)
        img_out = np.ceil((img_out / np.max(img_out)) * 255)
        plt.imshow(img_out, cmap="gray", interpolation="bicubic")
        plt.xticks([]), plt.yticks([])
        plt.show()

    def exportXLSX(self, array, flname):
        workbook = xlsxwriter.Workbook(str(flname) + '.xlsx')
        worksheet = workbook.add_worksheet()
        # array = self.image
        row = 0
        for col, data in enumerate(array):
            worksheet.write_column(row, col, data)
        workbook.close()

    def exportCSV(self, array, flname):
        with open(str(flname) + '.csv', 'w', newline='') as f_output:
            csv_output = csv.writer(f_output)
            csv_output.writerow(["Image Name ", "R", "G", "B"])
            width, height = array.shape[:2]
            print(f'{array}, Width {width}, Height {height}')  # show
            # Read the details of each pixel and write them to the file
            csv_output.writerows([array, array[x, y]] for x, y in product(range(width), range(height)))

    def saveClicked(self):
        flname, filter = QFileDialog.getSaveFileName(self, 'Save File', 'C:\\',
                                                     "Image Files (*.jpg)")
        if flname:
            cv2.imwrite(flname, self.image)
        else:
            print('Error')

    def displayImage(self, windows=1):
        qformat = QImage.Format_Indexed8
        if len(self.image.shape) == 3:
            if (self.image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(
            self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qformat)
        img = img.rgbSwapped()
        if windows == 1:
            self.imgLabel.setPixmap(QPixmap.fromImage(img))
            self.imgLabel.setAlignment(
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.imgLabel.setScaledContents(True)
        elif windows == 2:
            self.hasilLabel.setPixmap(QPixmap.fromImage(img))
            self.hasilLabel.setAlignment(
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.hasilLabel.setScaledContents(True)


app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('Main Window')
window.show()
sys.exit(app.exec_())
