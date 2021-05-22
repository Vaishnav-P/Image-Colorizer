import sys
import platform
from PyQt5 import QtCore,QtGui,QtWidgets
from form import Ui_gui
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import *
import test2_b as Colorizer
import time
from qt_material import apply_stylesheet
from loadingScreen import Ui_loadingScreen
counter = 0
class loadingScreen(QtWidgets.QMainWindow):
	def __init__(self):
		QtWidgets.QMainWindow.__init__(self)
		self.ui = Ui_loadingScreen()
		self.ui.setupUi(self)

		self.setWindowFlag(QtCore.Qt.FramelessWindowHint)	
		self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
		self.centerWindow()

		self.shadow = QGraphicsDropShadowEffect(self)
		self.shadow.setBlurRadius(20)
		self.shadow.setXOffset(0)
		self.shadow.setYOffset(0)
		self.shadow.setColor(QColor(0,0,0,60))
		self.ui.dropShadowFrame.setGraphicsEffect(self.shadow)

		self.timer = QtCore.QTimer()
		self.timer.timeout.connect(self.progress)
		self.timer.start(35)
		self.ui.label_2.setText("<strong>WELCOME</strong> TO Image Colorizer")
		QtCore.QTimer.singleShot(1500, lambda: self.ui.label_2.setText("<strong>LOADING</strong> MODULES"))
		QtCore.QTimer.singleShot(3000, lambda: self.ui.label_2.setText("<strong>LOADING</strong> USER INTERFACE"))
		self.show()

	def progress(self):
		global counter

		self.ui.progressBar.setValue(counter)

		if counter > 100:

			self.timer.stop()

			self.main = Main()
			self.main.show()
			self.close()

		counter+=1
	
	def centerWindow(self):
		frameGm = self.frameGeometry()
		screen = QtWidgets.QApplication.desktop().screenNumber(QtWidgets.QApplication.desktop().cursor().pos())
		centerPoint =QtWidgets.QApplication.desktop().screenGeometry(screen).center()
		frameGm.moveCenter(centerPoint)
		self.move(frameGm.topLeft())	

class Main(QtWidgets.QMainWindow):

	def __init__(self):
		QtWidgets.QMainWindow.__init__(self)
		self.ui = Ui_gui()
		self.ui.setupUi(self)
		self.centerWindow()
		self.ui.browse.clicked.connect(self.browsefiles)
		self.ui.ok.clicked.connect(self.colorize)


	def browsefiles(self):

		fname = QFileDialog.getOpenFileName(self,'Browse','..')
		self.ui.input1.setText(fname[0])

	def colorize(self):
		
		saveFile = self.ui.input2.text()
		fileName = self.ui.input1.text()
		if saveFile == '':
			saveFile = 'default'
		self.ui.progressBar.setProperty('value',0)
		if(fileName!=''):
			self.increment()
			completed = Colorizer.colorize(fileName,saveFile)
			if(completed):
				msg = QMessageBox()
				msg.setWindowTitle('Image Colorizer')
				msg.setText("Success!")
				msg.setIcon(QMessageBox.Information)
				x = msg.exec_()
			else:
				msg = QMessageBox()
				msg.setWindowTitle('Image Colorizer')
				msg.setText("Error!")
				msg.setIcon(QMessageBox.Critical)
				x = msg.exec_()

	def increment(self,*args):
		for i in range(100):
			progress = i+1
			self.ui.progressBar.setProperty('value',progress)
			time.sleep(0.1)

	def centerWindow(self):
		frameGm = self.frameGeometry()
		screen = QtWidgets.QApplication.desktop().screenNumber(QtWidgets.QApplication.desktop().cursor().pos())
		centerPoint =QtWidgets.QApplication.desktop().screenGeometry(screen).center()
		frameGm.moveCenter(centerPoint)
		self.move(frameGm.topLeft())	

if __name__ == "__main__":
	app =QtWidgets.QApplication(sys.argv)
	window = loadingScreen()
	window.show()
	sys.exit(app.exec_())