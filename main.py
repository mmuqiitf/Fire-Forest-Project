import sys
import cv2
import numpy as np
import math
import xlsxwriter
from konvolusi import convolve as conv
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
        self.btnLoad.clicked.connect(self.loadClicked)
        self.btnSave.clicked.connect(self.saveClicked)
        self.btnIdentifikasi.clicked.connect(self.detectProcess)
        self.actionGrayscale.triggered.connect(self.grayClicked)
        self.btnXls.clicked.connect(self.exportXLSX)
        self.actionContrast.triggered.connect(self.contrastClicked)
        self.actionMean_Filter.triggered.connect(self.meanClicked)

    @pyqtSlot()
    def loadClicked(self):
        flname, filter = QFileDialog.getOpenFileName(self, 'Open File', 'E:\\Muqiit\\Kuliah\\PCD\\Project', "Image Files (*.png *.jpg *.jpeg)")
        if flname:
            self.loadImage(flname)
        else:
            print('Invalid Image')

    def loadImage(self, flname):
        self.image = cv2.imread(flname, cv2.IMREAD_COLOR)
        img = self.image
        self.displayImage()

    def grayClicked(self):
        h, w = self.image.shape[:2]
        gray = np.zeros((h, w), np.uint8)
        for i in range(h):
            for j in range(w):
                gray[i, j] = np.clip(
                    0.299 * self.image[i, j, 0] + 0.587 * self.image[i, j, 1] + 0.114 * self.image[i, j, 2], 0, 255)
        self.image = gray
        print(self.image)
        self.displayImage(2)
        plt.hist(self.image.ravel(), 255, [0, 255])
        plt.show()

    def contrastClicked(self):
        img = self.image
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
        self.image = img
        self.displayImage(2)

    def meanClicked(self):
        img = self.image
        mean = cv2.blur(img, (5, 5))
        cv2.imshow("Mean", mean)

    def mean(self):
        cv2.mean(self.image)

    def detectProcess(self):
        img = self.image
        # variabel untuk sobel
        scale = 1
        delta = 0
        ddepth = cv2.CV_16S
        # Blur menggunakan mean filter
        blur = cv2.blur(img, (5, 5))
        # Ubah mode ke HSV
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        # Ambang batas untuk warna Red
        lower = np.array([5, 5, 111])
        upper = np.array([90, 255, 255])
        # Mencari pixel diantara ambang batas
        mask = cv2.inRange(hsv, lower, upper)
        output = cv2.bitwise_and(img, hsv, mask=mask)
        # Ubah mode ke greyscale
        output_for_sobel = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        # Sobel Edge Detection
        grad_x = cv2.Sobel(output_for_sobel, ddepth, 1, 0, ksize=3, scale=scale, delta=delta,
                           borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(output_for_sobel, ddepth, 0, 1, ksize=3, scale=scale, delta=delta,
                           borderType=cv2.BORDER_DEFAULT)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        # Menghitung node red
        no_red = cv2.countNonZero(mask)
        print('Node Red : ' + str(no_red))
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1
        color = (255, 255, 255)
        thickness = 2

        if int(no_red) >= 20000:
            print('Fire detected')
            cv2.putText(output, "Fire Detected", org, font, fontScale, color, thickness, cv2.LINE_AA)
        else:
            print('Non Fire')
            cv2.putText(output, "Non Fire", org, font, fontScale, color, thickness, cv2.LINE_AA)

        # cv2.imshow("Output", output)
        # cv2.imshow("Original Imahe", img)
        cv2.imshow("Sobel Edge Detection", output_for_sobel)
        self.image = output
        self.displayImage(2)
        self.image = grad
        self.displayImage(3)
        cv2.waitKey(0)

    def exportXLSX(self):
        workbook = xlsxwriter.Workbook('arrays.xlsx')
        worksheet = workbook.add_worksheet()
        array = self.image
        row = 0
        for col, data in enumerate(array):
            worksheet.write_column(row, col, data)
        workbook.close()

    def saveClicked(self):
        flname, filter = QFileDialog.getSaveFileName(self, 'Save File', 'C:\\',
                                                     "Image Files (*.jpg)")
        if flname:
            cv2.imwrite(flname, self.image)
        else:
            print('Error')

    def displayImage(self, windows=1):
        qformat = QImage.Format_Indexed8
        if len(self.image.shape) == 3:  # row[0],col[1],channel[2]
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
        elif windows == 3:
            self.tepiHasilLabel.setPixmap(QPixmap.fromImage(img))
            self.tepiHasilLabel.setAlignment(
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.tepiHasilLabel.setScaledContents(True)


app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('Main Window')
window.show()
sys.exit(app.exec_())
