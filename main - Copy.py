import numpy as np
import cv2
import sys
import csv
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QHBoxLayout, QFrame, QMenuBar, QAction, QSpacerItem, QSizePolicy
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QIcon
from PyQt5.QtCore import QTimer, Qt
import pyqtgraph as pg

class HeartRateMonitor(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

        self.realWidth = 320
        self.realHeight = 240
        self.videoWidth = 160
        self.videoHeight = 120
        self.videoChannels = 3
        self.videoFrameRate = 15

        self.levels = 3
        self.alpha = 170
        self.minFrequency = 1.0
        self.maxFrequency = 2.0
        self.bufferSize = 150
        self.bufferIndex = 0

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.loadingTextLocation = (20, 30)
        self.bpmTextLocation = (self.videoWidth // 2 + 5, 30)
        self.fontScale = 1
        self.fontColor = (255, 255, 255)
        self.lineType = 2
        self.boxColor = (0, 255, 0)
        self.boxWeight = 3

        self.webcam = cv2.VideoCapture(0)
        self.webcam.set(3, self.realWidth)
        self.webcam.set(4, self.realHeight)

        self.firstFrame = np.zeros((self.videoHeight, self.videoWidth, self.videoChannels))
        self.firstGauss = self.buildGauss(self.firstFrame, self.levels + 1)[self.levels]
        self.videoGauss = np.zeros((self.bufferSize, self.firstGauss.shape[0], self.firstGauss.shape[1], self.videoChannels))
        self.fourierTransformAvg = np.zeros((self.bufferSize))

        self.frequencies = (1.0 * self.videoFrameRate) * np.arange(self.bufferSize) / (1.0 * self.bufferSize)
        self.mask = (self.frequencies >= self.minFrequency) & (self.frequencies <= self.maxFrequency)

        self.bpmCalculationFrequency = 15
        self.bpmBufferIndex = 0
        self.bpmBufferSize = 10
        self.bpmBuffer = np.zeros((self.bpmBufferSize))

        self.i = 0
        self.monitoring = False

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)

        self.faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Load pre-trained age and gender models (disabled due to missing files)
        self.ageNet = None
        self.genderNet = None
        self.ageList = ['(0-3)', '(4-9)', '(10-15)', '(16-19)', '(20-39)', '(40-59)', '(60-100)']
        self.genderList = ['Male', 'Female']

        # Prepare CSV file for writing heart rate data
        self.csv_file = open('heart_rate_data.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Timestamp', 'Heart Rate (BPM)', 'Age', 'Gender'])

    def initUI(self):
        self.setWindowTitle('Heart Rate Monitor')
        self.setStyleSheet("background-color: #2E3440; color: white;")
        # self.setWindowIcon(QIcon('icon.png'))  # Add an application icon

        self.layout = QVBoxLayout()
        self.videoLayout = QHBoxLayout()
        
        # Menu bar with settings
        self.menuBar = QMenuBar(self)
        self.settingsMenu = self.menuBar.addMenu('Settings')
        
        saveCSVAction = QAction('Save Data', self)
        self.settingsMenu.addAction(saveCSVAction)
        
        # Video label
        self.videoLabel = QLabel()
        self.videoLabel.setFixedSize(320, 240)
        self.videoLabel.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.videoLabel.setStyleSheet("border: 2px solid #4C566A;")
        self.videoLayout.addWidget(self.videoLabel)

        self.hrLayout = QVBoxLayout()
        self.hrLabel = QLabel('HR: Not Available')
        self.hrLabel.setFont(QFont('Arial', 24))
        self.hrLabel.setStyleSheet("color: #88C0D0;")
        self.hrLayout.addWidget(self.hrLabel, alignment=Qt.AlignCenter)

        self.hrTrendPlot = pg.PlotWidget(title="Heart Rate Trend")
        self.hrTrendPlot.setBackground('#3B4252')
        self.hrTrendPlot.getAxis('left').setPen(pg.mkPen('white'))
        self.hrTrendPlot.getAxis('bottom').setPen(pg.mkPen('white'))
        self.hrTrendPlot.setYRange(40, 180)
        self.hrCurve = self.hrTrendPlot.plot(pen=pg.mkPen('r', width=2))
        self.hrData = []

        self.hrLayout.addWidget(self.hrTrendPlot)
        self.videoLayout.addLayout(self.hrLayout)
        self.layout.addLayout(self.videoLayout)

        # Pulse Plot
        self.pulsePlot = pg.PlotWidget(title="Pulse Signal")
        self.pulsePlot.setBackground('#3B4252')
        self.pulsePlot.getAxis('left').setPen(pg.mkPen('white'))
        self.pulsePlot.getAxis('bottom').setPen(pg.mkPen('white'))
        self.pulsePlot.enableAutoRange(axis='y')
        self.pulseCurve = self.pulsePlot.plot(pen=pg.mkPen('b', width=2))
        self.layout.addWidget(self.pulsePlot)
        self.pulseBufferSize = 300
        self.pulseBuffer = np.zeros(self.pulseBufferSize)
        self.pulseIndex = 0


        # Buttons with tooltips
        self.buttonLayout = QHBoxLayout()
        self.startButton = QPushButton('Start')
        self.stopButton = QPushButton('Stop')
        self.startButton.setToolTip('Start heart rate monitoring')
        self.stopButton.setToolTip('Stop heart rate monitoring')

        self.startButton.setStyleSheet("background-color: #5E81AC; color: white;")
        self.stopButton.setStyleSheet("background-color: #BF616A; color: white;")

        self.startButton.clicked.connect(self.startMonitoring)
        self.stopButton.clicked.connect(self.stopMonitoring)

        self.buttonLayout.addWidget(self.startButton)
        self.buttonLayout.addWidget(self.stopButton)
        
        # Spacer for button alignment
        spacer = QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.buttonLayout.addItem(spacer)
        
        self.layout.addLayout(self.buttonLayout)

        self.setLayout(self.layout)

    def buildGauss(self, frame, levels):
        pyramid = [frame]
        for level in range(levels):
            frame = cv2.pyrDown(frame)
            pyramid.append(frame)
        return pyramid

    def reconstructFrame(self, pyramid, index, levels):
        filteredFrame = pyramid[index]
        for level in range(levels):
            filteredFrame = cv2.pyrUp(filteredFrame)
        filteredFrame = filteredFrame[:self.videoHeight, :self.videoWidth]
        return filteredFrame

    def startMonitoring(self):
        self.monitoring = True
        self.timer.start(1000 // self.videoFrameRate)
        self.startButton.setText("Pause")  # Change button text on click

    def stopMonitoring(self):
        self.monitoring = False
        self.timer.stop()
        self.startButton.setText("Start")  # Change back to Start when stopped

    def normalizeSkinColor(self, frame):
        normalizedFrame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        return normalizedFrame

    def applyAdaptiveHistogramEqualization(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        equalizedFrame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return equalizedFrame

    def saveToCSV(self, bpm, age, gender):
        import time
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        self.csv_writer.writerow([timestamp, bpm, age, gender])
        self.csv_file.flush()

    def predictAgeGender(self, face):
        if self.ageNet is None or self.genderNet is None:
            return 'Unknown', 'Unknown'
        
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (104.0, 177.0, 123.0), swapRB=False)
        self.genderNet.setInput(blob)
        genderPreds = self.genderNet.forward()
        gender = self.genderList[genderPreds[0].argmax()]

        self.ageNet.setInput(blob)
        agePreds = self.ageNet.forward()
        age = self.ageList[agePreds[0].argmax()]
        
        return age, gender

    def update(self):
        if not self.monitoring:
            return

        ret, frame = self.webcam.read()
        if not ret:
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = self.applyAdaptiveHistogramEqualization(frame)
        frame = self.normalizeSkinColor(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            self.hrLabel.setText("No face detected")
            return

        x, y, w, h = faces[0]
        detectionFrame = frame[y:y + h, x:x + w]
        faceForPrediction = detectionFrame.copy()
        faceForPrediction = cv2.resize(faceForPrediction, (227, 227))

        age, gender = self.predictAgeGender(faceForPrediction)

        forehead = detectionFrame[0:int(0.3 * h), 0:w]
        detectionFrame = cv2.resize(forehead, (self.videoWidth, self.videoHeight))
        self.videoGauss[self.bufferIndex] = self.buildGauss(detectionFrame, self.levels + 1)[self.levels]
        fourierTransform = np.fft.fft(self.videoGauss, axis=0)

        fourierTransform[self.mask == False] = 0

        if self.bufferIndex % self.bpmCalculationFrequency == 0:
            self.i += 1
            for buf in range(self.bufferSize):
                self.fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
            hz = self.frequencies[np.argmax(self.fourierTransformAvg)]
            bpm = 60.0 * hz
            self.bpmBuffer[self.bpmBufferIndex] = bpm
            self.bpmBufferIndex = (self.bpmBufferIndex + 1) % self.bpmBufferSize
            self.hrData.append(bpm)
            self.hrCurve.setData(self.hrData[-50:])

            # Save heart rate to CSV file
            self.saveToCSV(bpm, age, gender)

        filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
        filtered = filtered * self.alpha

        filteredFrame = self.reconstructFrame(filtered, self.bufferIndex, self.levels)
        outputFrame = detectionFrame + filteredFrame
        outputFrame = cv2.convertScaleAbs(outputFrame)

        self.bufferIndex = (self.bufferIndex + 1) % self.bufferSize

        frame[y:y + int(0.3 * h), x:x + w] = cv2.resize(outputFrame, (w, int(0.3 * h)))
        cv2.rectangle(frame, (x, y), (x + w, y + int(0.3 * h)), self.boxColor, self.boxWeight)

        if self.i > self.bpmBufferSize:
            avg_bpm = self.bpmBuffer.mean()
            bpm_text = "HR: %.1f, Age: %s, Gender: %s" % (avg_bpm, age, gender)
            hr_status = self.getHeartRateStatus(avg_bpm,age,gender)  # Get heart rate status
            self.hrLabel.setText(f"{bpm_text} - Status: {hr_status}")  # Update label with status
        else:
            bpm_text = "Calculating HR..."
            self.hrLabel.setText(bpm_text)

        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        height, width, channel = img.shape
        step = channel * width
        qImg = QImage(img.data, width, height, step, QImage.Format_RGB888)
        self.videoLabel.setPixmap(QPixmap.fromImage(qImg))

        # Raw pulse signal (green channel mean)
        green_mean = np.mean(forehead[:, :, 1])

        self.pulseBuffer[self.pulseIndex] = green_mean
        self.pulseIndex = (self.pulseIndex + 1) % self.pulseBufferSize

        # Plot full pulse waveform
        self.pulseCurve.setData(self.pulseBuffer)


    def getHeartRateStatus(self, bpm, age, gender):
        if age == 'Unknown' or gender == 'Unknown':
            # Default to normal adult range
            if bpm < 60:
                return "Low"
            elif 60 <= bpm <= 100:
                return "Normal"
            elif 100 < bpm <= 120:
                return "High"
            else:
                return "Very High"
        
        if gender.lower() == "male":
            if age == '(0-3)' or age == '(4-9)' or age == '(10-15)' or age == '(16-19)':
                if bpm < 50:
                    return "Very Low"
                elif 50 <= bpm < 70:
                    return "Low"
                elif 70 <= bpm <= 100:
                    return "Normal"
                elif 100 < bpm <= 130:
                    return "High"
                else:
                    return "Very High"
            elif age == '(20-39)':
                if bpm < 55:
                    return "Very Low"
                elif 55 <= bpm < 70:
                    return "Low"
                elif 70 <= bpm <= 100:
                    return "Normal"
                elif 100 < bpm <= 120:
                    return "High"
                else:
                    return "Very High"
            else:  # (40-59), (60-100)
                if bpm < 50:
                    return "Very Low"
                elif 50 <= bpm < 65:
                    return "Low"
                elif 65 <= bpm <= 100:
                    return "Normal"
                elif 100 < bpm <= 120:
                    return "High"
                else:
                    return "Very High"
        
        elif gender.lower() == "female":
            if age == '(0-3)' or age == '(4-9)' or age == '(10-15)' or age == '(16-19)':
                if bpm < 55:
                    return "Very Low"
                elif 55 <= bpm < 75:
                    return "Low"
                elif 75 <= bpm <= 105:
                    return "Normal"
                elif 105 < bpm <= 135:
                    return "High"
                else:
                    return "Very High"
            elif age == '(20-39)':
                if bpm < 60:
                    return "Very Low"
                elif 60 <= bpm < 75:
                    return "Low"
                elif 75 <= bpm <= 105:
                    return "Normal"
                elif 105 < bpm <= 125:
                    return "High"
                else:
                    return "Very High"
            else:  # age > 40
                if bpm < 55:
                    return "Very Low"
                elif 55 <= bpm < 70:
                    return "Low"
                elif 70 <= bpm <= 105:
                    return "Normal"
                elif 105 < bpm <= 125:
                    return "High"
                else:
                    return "Very High"

    def closeEvent(self, event):
        self.webcam.release()
        self.csv_file.close()  # Close the CSV file when the application is closed
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    monitor = HeartRateMonitor()
    monitor.show()
    sys.exit(app.exec_())