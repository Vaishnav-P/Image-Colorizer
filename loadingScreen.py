# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'form.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_loadingScreen(object):
    def setupUi(self, loadingScreen):
        loadingScreen.setObjectName("loadingScreen")
        loadingScreen.resize(680, 400)
        self.centralwidget = QtWidgets.QWidget(loadingScreen)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setContentsMargins(10, 10, 10, 10)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.dropShadowFrame = QtWidgets.QFrame(self.centralwidget)
        self.dropShadowFrame.setStyleSheet("QFrame{\n"
"background-color:rgb(56,58,89);\n"
"color:rgb(220,220,220); \n"
"border-radius:10px;\n"
"}")
        self.dropShadowFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.dropShadowFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.dropShadowFrame.setObjectName("dropShadowFrame")
        self.label = QtWidgets.QLabel(self.dropShadowFrame)
        self.label.setGeometry(QtCore.QRect(0, 130, 661, 91))
        font = QtGui.QFont()
        font.setPointSize(40)
        self.label.setFont(font)
        self.label.setStyleSheet("color: rgb(115, 210, 22);")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.dropShadowFrame)
        self.label_2.setGeometry(QtCore.QRect(0, 200, 661, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("color: rgb(172, 62, 183);\n"
"border-color: rgb(32, 74, 135);")
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.progressBar = QtWidgets.QProgressBar(self.dropShadowFrame)
        self.progressBar.setGeometry(QtCore.QRect(50, 260, 571, 23))
        self.progressBar.setStyleSheet("QProgressBar{\n"
"    \n"
"    background-color: rgb(98, 114, 164);\n"
"    color:rgb(200,200,200);\n"
"    border-style:none;\n"
"    border-radius:10px;\n"
"    text-align:center;\n"
"}\n"
"QProgressBar::chunk{\n"
"    border-radius:10px;\n"
"    \n"
"    background-color: qlineargradient(spread:pad, x1:0, y1:0.58, x2:1, y2:0.568, stop:0 rgba(159, 225, 95, 255), stop:1 rgba(78, 154, 6, 255));\n"
"\n"
"}")
        self.progressBar.setProperty("value", 24)
        self.progressBar.setTextVisible(True)
        self.progressBar.setObjectName("progressBar")
        self.label_3 = QtWidgets.QLabel(self.dropShadowFrame)
        self.label_3.setGeometry(QtCore.QRect(0, 290, 661, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("color: rgb(52, 101, 164);\n"
"border-color: rgb(32, 74, 135);")
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.dropShadowFrame)
        self.label_4.setGeometry(QtCore.QRect(180, 330, 471, 41))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_4.setFont(font)
        self.label_4.setStyleSheet("color: rgb(52, 101, 164);\n"
"border-color: rgb(32, 74, 135);")
        self.label_4.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_4.setObjectName("label_4")
        self.verticalLayout.addWidget(self.dropShadowFrame)
        loadingScreen.setCentralWidget(self.centralwidget)

        self.retranslateUi(loadingScreen)
        QtCore.QMetaObject.connectSlotsByName(loadingScreen)

    def retranslateUi(self, loadingScreen):
        _translate = QtCore.QCoreApplication.translate
        loadingScreen.setWindowTitle(_translate("loadingScreen", "loadingScreen"))
        self.label.setText(_translate("loadingScreen", "<strong>IMAGE</strong> Colorizer"))
        self.label_2.setText(_translate("loadingScreen", "Color Your <strong>WORLD</strong>"))
        self.label_3.setText(_translate("loadingScreen", "<html><head/><body><p>loading..</p></body></html>"))
        self.label_4.setText(_translate("loadingScreen", "<strong>Created: </strong> Vaishnav P"))


# if __name__ == "__main__":
#     import sys
#     app = QtWidgets.QApplication(sys.argv)
#     loadingScreen = QtWidgets.QMainWindow()
#     ui = Ui_loadingScreen()
#     ui.setupUi(loadingScreen)
#     loadingScreen.show()
#     sys.exit(app.exec_())
