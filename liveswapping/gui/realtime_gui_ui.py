# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'realtime_gui.ui'
##
## Created by: Qt User Interface Compiler version 6.9.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QFrame, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QScrollArea, QSizePolicy,
    QSpacerItem, QSpinBox, QVBoxLayout, QWidget)

class Ui_RealtimeGUI(object):
    def setupUi(self, RealtimeGUI):
        if not RealtimeGUI.objectName():
            RealtimeGUI.setObjectName(u"RealtimeGUI")
        RealtimeGUI.resize(750, 700)
        RealtimeGUI.setMinimumSize(QSize(750, 700))
        self.mainLayout = QVBoxLayout(RealtimeGUI)
        self.mainLayout.setSpacing(0)
        self.mainLayout.setObjectName(u"mainLayout")
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.languageFrame = QFrame(RealtimeGUI)
        self.languageFrame.setObjectName(u"languageFrame")
        self.languageFrame.setMinimumSize(QSize(0, 45))
        self.languageFrame.setMaximumSize(QSize(16777215, 45))
        self.languageFrame.setFrameShape(QFrame.NoFrame)
        self.languageLayout = QHBoxLayout(self.languageFrame)
        self.languageLayout.setObjectName(u"languageLayout")
        self.languageLayout.setContentsMargins(20, 10, 20, 10)
        self.languageLabel = QLabel(self.languageFrame)
        self.languageLabel.setObjectName(u"languageLabel")
        font = QFont()
        font.setBold(True)
        self.languageLabel.setFont(font)

        self.languageLayout.addWidget(self.languageLabel)

        self.languageCombo = QComboBox(self.languageFrame)
        self.languageCombo.addItem("")
        self.languageCombo.addItem("")
        self.languageCombo.setObjectName(u"languageCombo")
        self.languageCombo.setMinimumSize(QSize(150, 0))

        self.languageLayout.addWidget(self.languageCombo)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.languageLayout.addItem(self.horizontalSpacer)

        self.versionLabel = QLabel(self.languageFrame)
        self.versionLabel.setObjectName(u"versionLabel")
        font1 = QFont()
        font1.setPointSize(9)
        self.versionLabel.setFont(font1)

        self.languageLayout.addWidget(self.versionLabel)


        self.mainLayout.addWidget(self.languageFrame)

        self.scrollArea = QScrollArea(RealtimeGUI)
        self.scrollArea.setObjectName(u"scrollArea")
        self.scrollArea.setFrameShape(QFrame.NoFrame)
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 748, 851))
        self.contentLayout = QVBoxLayout(self.scrollAreaWidgetContents)
        self.contentLayout.setSpacing(20)
        self.contentLayout.setObjectName(u"contentLayout")
        self.contentLayout.setContentsMargins(25, 25, 25, 25)
        self.sourceGroup = QGroupBox(self.scrollAreaWidgetContents)
        self.sourceGroup.setObjectName(u"sourceGroup")
        self.sourceGroup.setFont(font)
        self.sourceLayout = QVBoxLayout(self.sourceGroup)
        self.sourceLayout.setSpacing(15)
        self.sourceLayout.setObjectName(u"sourceLayout")
        self.sourceFileLayout = QHBoxLayout()
        self.sourceFileLayout.setObjectName(u"sourceFileLayout")
        self.sourceLabel = QLabel(self.sourceGroup)
        self.sourceLabel.setObjectName(u"sourceLabel")
        self.sourceLabel.setMinimumSize(QSize(120, 0))
        self.sourceLabel.setFont(font)

        self.sourceFileLayout.addWidget(self.sourceLabel)

        self.sourceEdit = QLineEdit(self.sourceGroup)
        self.sourceEdit.setObjectName(u"sourceEdit")

        self.sourceFileLayout.addWidget(self.sourceEdit)

        self.sourceBrowseButton = QPushButton(self.sourceGroup)
        self.sourceBrowseButton.setObjectName(u"sourceBrowseButton")
        self.sourceBrowseButton.setMinimumSize(QSize(100, 0))

        self.sourceFileLayout.addWidget(self.sourceBrowseButton)


        self.sourceLayout.addLayout(self.sourceFileLayout)


        self.contentLayout.addWidget(self.sourceGroup)

        self.modelGroup = QGroupBox(self.scrollAreaWidgetContents)
        self.modelGroup.setObjectName(u"modelGroup")
        self.modelGroup.setFont(font)
        self.modelLayout = QVBoxLayout(self.modelGroup)
        self.modelLayout.setSpacing(15)
        self.modelLayout.setObjectName(u"modelLayout")
        self.modelSelectLayout = QHBoxLayout()
        self.modelSelectLayout.setObjectName(u"modelSelectLayout")
        self.modelLabel = QLabel(self.modelGroup)
        self.modelLabel.setObjectName(u"modelLabel")
        self.modelLabel.setMinimumSize(QSize(120, 0))
        self.modelLabel.setFont(font)

        self.modelSelectLayout.addWidget(self.modelLabel)

        self.modelCombo = QComboBox(self.modelGroup)
        self.modelCombo.addItem("")
        self.modelCombo.addItem("")
        self.modelCombo.addItem("")
        self.modelCombo.addItem("")
        self.modelCombo.setObjectName(u"modelCombo")

        self.modelSelectLayout.addWidget(self.modelCombo)


        self.modelLayout.addLayout(self.modelSelectLayout)

        self.modelPathLayout = QHBoxLayout()
        self.modelPathLayout.setObjectName(u"modelPathLayout")
        self.modelPathLabel = QLabel(self.modelGroup)
        self.modelPathLabel.setObjectName(u"modelPathLabel")
        self.modelPathLabel.setMinimumSize(QSize(120, 0))
        self.modelPathLabel.setFont(font)

        self.modelPathLayout.addWidget(self.modelPathLabel)

        self.modelPathEdit = QLineEdit(self.modelGroup)
        self.modelPathEdit.setObjectName(u"modelPathEdit")
        self.modelPathEdit.setReadOnly(True)

        self.modelPathLayout.addWidget(self.modelPathEdit)

        self.modelBrowseButton = QPushButton(self.modelGroup)
        self.modelBrowseButton.setObjectName(u"modelBrowseButton")
        self.modelBrowseButton.setMinimumSize(QSize(100, 0))
        self.modelBrowseButton.setEnabled(False)

        self.modelPathLayout.addWidget(self.modelBrowseButton)


        self.modelLayout.addLayout(self.modelPathLayout)


        self.contentLayout.addWidget(self.modelGroup)

        self.processingGroup = QGroupBox(self.scrollAreaWidgetContents)
        self.processingGroup.setObjectName(u"processingGroup")
        self.processingGroup.setFont(font)
        self.processingLayout = QVBoxLayout(self.processingGroup)
        self.processingLayout.setSpacing(15)
        self.processingLayout.setObjectName(u"processingLayout")
        self.resolutionLayout = QHBoxLayout()
        self.resolutionLayout.setObjectName(u"resolutionLayout")
        self.resolutionLabel = QLabel(self.processingGroup)
        self.resolutionLabel.setObjectName(u"resolutionLabel")
        self.resolutionLabel.setMinimumSize(QSize(120, 0))
        self.resolutionLabel.setFont(font)

        self.resolutionLayout.addWidget(self.resolutionLabel)

        self.resolutionSpinBox = QSpinBox(self.processingGroup)
        self.resolutionSpinBox.setObjectName(u"resolutionSpinBox")
        self.resolutionSpinBox.setMinimumSize(QSize(100, 0))
        self.resolutionSpinBox.setMinimum(64)
        self.resolutionSpinBox.setMaximum(512)
        self.resolutionSpinBox.setValue(128)

        self.resolutionLayout.addWidget(self.resolutionSpinBox)

        self.resolutionInfoLabel = QLabel(self.processingGroup)
        self.resolutionInfoLabel.setObjectName(u"resolutionInfoLabel")
        self.resolutionInfoLabel.setFont(font1)

        self.resolutionLayout.addWidget(self.resolutionInfoLabel)

        self.resolutionSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.resolutionLayout.addItem(self.resolutionSpacer)


        self.processingLayout.addLayout(self.resolutionLayout)

        self.delayLayout = QHBoxLayout()
        self.delayLayout.setObjectName(u"delayLayout")
        self.delayLabel = QLabel(self.processingGroup)
        self.delayLabel.setObjectName(u"delayLabel")
        self.delayLabel.setMinimumSize(QSize(120, 0))
        self.delayLabel.setFont(font)

        self.delayLayout.addWidget(self.delayLabel)

        self.delaySpinBox = QSpinBox(self.processingGroup)
        self.delaySpinBox.setObjectName(u"delaySpinBox")
        self.delaySpinBox.setMinimumSize(QSize(100, 0))
        self.delaySpinBox.setMinimum(0)
        self.delaySpinBox.setMaximum(5000)
        self.delaySpinBox.setValue(0)

        self.delayLayout.addWidget(self.delaySpinBox)

        self.delayInfoLabel = QLabel(self.processingGroup)
        self.delayInfoLabel.setObjectName(u"delayInfoLabel")
        self.delayInfoLabel.setFont(font1)

        self.delayLayout.addWidget(self.delayInfoLabel)

        self.delaySpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.delayLayout.addItem(self.delaySpacer)


        self.processingLayout.addLayout(self.delayLayout)


        self.contentLayout.addWidget(self.processingGroup)

        self.attributesGroup = QGroupBox(self.scrollAreaWidgetContents)
        self.attributesGroup.setObjectName(u"attributesGroup")
        self.attributesGroup.setFont(font)
        self.attributesLayout = QVBoxLayout(self.attributesGroup)
        self.attributesLayout.setSpacing(15)
        self.attributesLayout.setObjectName(u"attributesLayout")
        self.attributeFileLayout = QHBoxLayout()
        self.attributeFileLayout.setObjectName(u"attributeFileLayout")
        self.attributeLabel = QLabel(self.attributesGroup)
        self.attributeLabel.setObjectName(u"attributeLabel")
        self.attributeLabel.setMinimumSize(QSize(120, 0))
        self.attributeLabel.setFont(font)

        self.attributeFileLayout.addWidget(self.attributeLabel)

        self.attributeEdit = QLineEdit(self.attributesGroup)
        self.attributeEdit.setObjectName(u"attributeEdit")

        self.attributeFileLayout.addWidget(self.attributeEdit)

        self.attributeBrowseButton = QPushButton(self.attributesGroup)
        self.attributeBrowseButton.setObjectName(u"attributeBrowseButton")
        self.attributeBrowseButton.setMinimumSize(QSize(100, 0))

        self.attributeFileLayout.addWidget(self.attributeBrowseButton)


        self.attributesLayout.addLayout(self.attributeFileLayout)

        self.attributeStepsLayout = QHBoxLayout()
        self.attributeStepsLayout.setObjectName(u"attributeStepsLayout")
        self.stepsLabel = QLabel(self.attributesGroup)
        self.stepsLabel.setObjectName(u"stepsLabel")
        self.stepsLabel.setMinimumSize(QSize(120, 0))
        self.stepsLabel.setFont(font)

        self.attributeStepsLayout.addWidget(self.stepsLabel)

        self.stepsSpinBox = QDoubleSpinBox(self.attributesGroup)
        self.stepsSpinBox.setObjectName(u"stepsSpinBox")
        self.stepsSpinBox.setMinimumSize(QSize(100, 0))
        self.stepsSpinBox.setMinimum(0)
        self.stepsSpinBox.setMaximum(0)
        self.stepsSpinBox.setSingleStep(0)
        self.stepsSpinBox.setValue(0)

        self.attributeStepsLayout.addWidget(self.stepsSpinBox)

        self.stepsInfoLabel = QLabel(self.attributesGroup)
        self.stepsInfoLabel.setObjectName(u"stepsInfoLabel")
        self.stepsInfoLabel.setFont(font1)

        self.attributeStepsLayout.addWidget(self.stepsInfoLabel)

        self.stepsSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.attributeStepsLayout.addItem(self.stepsSpacer)


        self.attributesLayout.addLayout(self.attributeStepsLayout)


        self.contentLayout.addWidget(self.attributesGroup)

        self.optionsGroup = QGroupBox(self.scrollAreaWidgetContents)
        self.optionsGroup.setObjectName(u"optionsGroup")
        self.optionsGroup.setFont(font)
        self.optionsLayout = QVBoxLayout(self.optionsGroup)
        self.optionsLayout.setSpacing(15)
        self.optionsLayout.setObjectName(u"optionsLayout")
        self.optionsRow1Layout = QHBoxLayout()
        self.optionsRow1Layout.setObjectName(u"optionsRow1Layout")
        self.mouthCheckBox = QCheckBox(self.optionsGroup)
        self.mouthCheckBox.setObjectName(u"mouthCheckBox")
        self.mouthCheckBox.setFont(font)

        self.optionsRow1Layout.addWidget(self.mouthCheckBox)

        self.obsCheckBox = QCheckBox(self.optionsGroup)
        self.obsCheckBox.setObjectName(u"obsCheckBox")
        self.obsCheckBox.setFont(font)

        self.optionsRow1Layout.addWidget(self.obsCheckBox)

        self.optionsRow1Spacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.optionsRow1Layout.addItem(self.optionsRow1Spacer)


        self.optionsLayout.addLayout(self.optionsRow1Layout)

        self.optionsRow2Layout = QHBoxLayout()
        self.optionsRow2Layout.setObjectName(u"optionsRow2Layout")
        self.fpsCheckBox = QCheckBox(self.optionsGroup)
        self.fpsCheckBox.setObjectName(u"fpsCheckBox")
        self.fpsCheckBox.setFont(font)

        self.optionsRow2Layout.addWidget(self.fpsCheckBox)

        self.enhanceCheckBox = QCheckBox(self.optionsGroup)
        self.enhanceCheckBox.setObjectName(u"enhanceCheckBox")
        self.enhanceCheckBox.setFont(font)

        self.optionsRow2Layout.addWidget(self.enhanceCheckBox)

        self.optionsRow2Spacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.optionsRow2Layout.addItem(self.optionsRow2Spacer)


        self.optionsLayout.addLayout(self.optionsRow2Layout)


        self.contentLayout.addWidget(self.optionsGroup)

        self.controlsLayout = QHBoxLayout()
        self.controlsLayout.setSpacing(15)
        self.controlsLayout.setObjectName(u"controlsLayout")
        self.controlsLeftSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.controlsLayout.addItem(self.controlsLeftSpacer)

        self.startButton = QPushButton(self.scrollAreaWidgetContents)
        self.startButton.setObjectName(u"startButton")
        self.startButton.setMinimumSize(QSize(200, 50))
        font2 = QFont()
        font2.setPointSize(14)
        font2.setBold(True)
        self.startButton.setFont(font2)

        self.controlsLayout.addWidget(self.startButton)

        self.stopButton = QPushButton(self.scrollAreaWidgetContents)
        self.stopButton.setObjectName(u"stopButton")
        self.stopButton.setMinimumSize(QSize(120, 50))
        self.stopButton.setEnabled(False)
        self.stopButton.setFont(font2)

        self.controlsLayout.addWidget(self.stopButton)

        self.controlsRightSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.controlsLayout.addItem(self.controlsRightSpacer)


        self.contentLayout.addLayout(self.controlsLayout)

        self.infoLabel = QLabel(self.scrollAreaWidgetContents)
        self.infoLabel.setObjectName(u"infoLabel")
        font3 = QFont()
        font3.setPointSize(9)
        font3.setItalic(True)
        self.infoLabel.setFont(font3)
        self.infoLabel.setAlignment(Qt.AlignCenter)

        self.contentLayout.addWidget(self.infoLabel)

        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.mainLayout.addWidget(self.scrollArea)


        self.retranslateUi(RealtimeGUI)

        QMetaObject.connectSlotsByName(RealtimeGUI)
    # setupUi

    def retranslateUi(self, RealtimeGUI):
        RealtimeGUI.setWindowTitle(QCoreApplication.translate("RealtimeGUI", u"LiveSwapping - Real-time Processing", None))
        self.languageLabel.setText(QCoreApplication.translate("RealtimeGUI", u"Language:", None))
        self.languageCombo.setItemText(0, QCoreApplication.translate("RealtimeGUI", u"English", None))
        self.languageCombo.setItemText(1, QCoreApplication.translate("RealtimeGUI", u"\u0420\u0443\u0441\u0441\u043a\u0438\u0439", None))

        self.versionLabel.setText(QCoreApplication.translate("RealtimeGUI", u"LiveSwapping v0.1", None))
        self.sourceGroup.setTitle(QCoreApplication.translate("RealtimeGUI", u"Source Image", None))
        self.sourceLabel.setText(QCoreApplication.translate("RealtimeGUI", u"Face Image:", None))
        self.sourceEdit.setPlaceholderText(QCoreApplication.translate("RealtimeGUI", u"Select face image to swap from...", None))
        self.sourceBrowseButton.setText(QCoreApplication.translate("RealtimeGUI", u"Browse", None))
        self.modelGroup.setTitle(QCoreApplication.translate("RealtimeGUI", u"AI Model", None))
        self.modelLabel.setText(QCoreApplication.translate("RealtimeGUI", u"Model Type:", None))
        self.modelCombo.setItemText(0, QCoreApplication.translate("RealtimeGUI", u"reswapper128 (StyleTransfer)", None))
        self.modelCombo.setItemText(1, QCoreApplication.translate("RealtimeGUI", u"reswapper256 (StyleTransfer)", None))
        self.modelCombo.setItemText(2, QCoreApplication.translate("RealtimeGUI", u"inswapper128 (InsightFace)", None))
        self.modelCombo.setItemText(3, QCoreApplication.translate("RealtimeGUI", u"DFM Model (Custom)", None))

        self.modelPathLabel.setText(QCoreApplication.translate("RealtimeGUI", u"Model Path:", None))
        self.modelPathEdit.setPlaceholderText(QCoreApplication.translate("RealtimeGUI", u"Model will be downloaded automatically...", None))
        self.modelBrowseButton.setText(QCoreApplication.translate("RealtimeGUI", u"Browse", None))
        self.processingGroup.setTitle(QCoreApplication.translate("RealtimeGUI", u"Processing Settings", None))
        self.resolutionLabel.setText(QCoreApplication.translate("RealtimeGUI", u"Resolution:", None))
        self.resolutionSpinBox.setSuffix(QCoreApplication.translate("RealtimeGUI", u"px", None))
        self.resolutionInfoLabel.setText(QCoreApplication.translate("RealtimeGUI", u"Higher = better quality, slower processing", None))
        self.delayLabel.setText(QCoreApplication.translate("RealtimeGUI", u"Frame Delay:", None))
        self.delaySpinBox.setSuffix(QCoreApplication.translate("RealtimeGUI", u" ms", None))
        self.delayInfoLabel.setText(QCoreApplication.translate("RealtimeGUI", u"Add delay between frames (0 = no delay)", None))
        self.attributesGroup.setTitle(QCoreApplication.translate("RealtimeGUI", u"Face Attributes (Advanced)", None))
        self.attributeLabel.setText(QCoreApplication.translate("RealtimeGUI", u"Attribute Dir:", None))
        self.attributeEdit.setPlaceholderText(QCoreApplication.translate("RealtimeGUI", u"Optional: Select face attribute direction...", None))
        self.attributeBrowseButton.setText(QCoreApplication.translate("RealtimeGUI", u"Browse", None))
        self.stepsLabel.setText(QCoreApplication.translate("RealtimeGUI", u"Attribute Steps:", None))
        self.stepsInfoLabel.setText(QCoreApplication.translate("RealtimeGUI", u"Face attribute modification strength", None))
        self.optionsGroup.setTitle(QCoreApplication.translate("RealtimeGUI", u"Options", None))
        self.mouthCheckBox.setText(QCoreApplication.translate("RealtimeGUI", u"Retain target mouth (preserve speech)", None))
        self.obsCheckBox.setText(QCoreApplication.translate("RealtimeGUI", u"Send to OBS Studio", None))
        self.fpsCheckBox.setText(QCoreApplication.translate("RealtimeGUI", u"Show FPS and delay info", None))
        self.enhanceCheckBox.setText(QCoreApplication.translate("RealtimeGUI", u"Enhance camera resolution", None))
        self.startButton.setText(QCoreApplication.translate("RealtimeGUI", u"Start Real-time", None))
        self.stopButton.setText(QCoreApplication.translate("RealtimeGUI", u"Stop", None))
        self.infoLabel.setText(QCoreApplication.translate("RealtimeGUI", u"Press Stop or close this window to stop real-time processing.", None))
    # retranslateUi

