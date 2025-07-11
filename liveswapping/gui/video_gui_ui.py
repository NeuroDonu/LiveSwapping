# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'video_gui.ui'
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
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFrame,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QProgressBar, QPushButton, QScrollArea, QSizePolicy,
    QSpacerItem, QSpinBox, QVBoxLayout, QWidget)

class Ui_VideoGUI(object):
    def setupUi(self, VideoGUI):
        if not VideoGUI.objectName():
            VideoGUI.setObjectName(u"VideoGUI")
        VideoGUI.resize(700, 650)
        VideoGUI.setMinimumSize(QSize(700, 650))
        self.mainLayout = QVBoxLayout(VideoGUI)
        self.mainLayout.setSpacing(0)
        self.mainLayout.setObjectName(u"mainLayout")
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.languageFrame = QFrame(VideoGUI)
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

        self.scrollArea = QScrollArea(VideoGUI)
        self.scrollArea.setObjectName(u"scrollArea")
        self.scrollArea.setFrameShape(QFrame.NoFrame)
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 698, 771))
        self.contentLayout = QVBoxLayout(self.scrollAreaWidgetContents)
        self.contentLayout.setSpacing(20)
        self.contentLayout.setObjectName(u"contentLayout")
        self.contentLayout.setContentsMargins(25, 25, 25, 25)
        self.filesGroup = QGroupBox(self.scrollAreaWidgetContents)
        self.filesGroup.setObjectName(u"filesGroup")
        self.filesGroup.setFont(font)
        self.filesLayout = QVBoxLayout(self.filesGroup)
        self.filesLayout.setSpacing(15)
        self.filesLayout.setObjectName(u"filesLayout")
        self.modeLayout = QHBoxLayout()
        self.modeLayout.setObjectName(u"modeLayout")
        self.modeLabel = QLabel(self.filesGroup)
        self.modeLabel.setObjectName(u"modeLabel")
        self.modeLabel.setMinimumSize(QSize(120, 0))
        self.modeLabel.setFont(font)

        self.modeLayout.addWidget(self.modeLabel)

        self.modeCombo = QComboBox(self.filesGroup)
        self.modeCombo.addItem("")
        self.modeCombo.addItem("")
        self.modeCombo.setObjectName(u"modeCombo")
        self.modeCombo.setMinimumSize(QSize(200, 0))

        self.modeLayout.addWidget(self.modeCombo)

        self.modeSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.modeLayout.addItem(self.modeSpacer)


        self.filesLayout.addLayout(self.modeLayout)

        self.sourceLayout = QHBoxLayout()
        self.sourceLayout.setObjectName(u"sourceLayout")
        self.sourceLabel = QLabel(self.filesGroup)
        self.sourceLabel.setObjectName(u"sourceLabel")
        self.sourceLabel.setMinimumSize(QSize(120, 0))
        self.sourceLabel.setFont(font)

        self.sourceLayout.addWidget(self.sourceLabel)

        self.sourceEdit = QLineEdit(self.filesGroup)
        self.sourceEdit.setObjectName(u"sourceEdit")

        self.sourceLayout.addWidget(self.sourceEdit)

        self.sourceBrowseButton = QPushButton(self.filesGroup)
        self.sourceBrowseButton.setObjectName(u"sourceBrowseButton")
        self.sourceBrowseButton.setMinimumSize(QSize(100, 0))

        self.sourceLayout.addWidget(self.sourceBrowseButton)


        self.filesLayout.addLayout(self.sourceLayout)

        self.targetLayout = QHBoxLayout()
        self.targetLayout.setObjectName(u"targetLayout")
        self.targetLabel = QLabel(self.filesGroup)
        self.targetLabel.setObjectName(u"targetLabel")
        self.targetLabel.setMinimumSize(QSize(120, 0))
        self.targetLabel.setFont(font)

        self.targetLayout.addWidget(self.targetLabel)

        self.targetEdit = QLineEdit(self.filesGroup)
        self.targetEdit.setObjectName(u"targetEdit")

        self.targetLayout.addWidget(self.targetEdit)

        self.targetBrowseButton = QPushButton(self.filesGroup)
        self.targetBrowseButton.setObjectName(u"targetBrowseButton")
        self.targetBrowseButton.setMinimumSize(QSize(100, 0))

        self.targetLayout.addWidget(self.targetBrowseButton)


        self.filesLayout.addLayout(self.targetLayout)

        self.outputLayout = QHBoxLayout()
        self.outputLayout.setObjectName(u"outputLayout")
        self.outputLabel = QLabel(self.filesGroup)
        self.outputLabel.setObjectName(u"outputLabel")
        self.outputLabel.setMinimumSize(QSize(120, 0))
        self.outputLabel.setFont(font)

        self.outputLayout.addWidget(self.outputLabel)

        self.outputEdit = QLineEdit(self.filesGroup)
        self.outputEdit.setObjectName(u"outputEdit")

        self.outputLayout.addWidget(self.outputEdit)

        self.outputBrowseButton = QPushButton(self.filesGroup)
        self.outputBrowseButton.setObjectName(u"outputBrowseButton")
        self.outputBrowseButton.setMinimumSize(QSize(100, 0))

        self.outputLayout.addWidget(self.outputBrowseButton)


        self.filesLayout.addLayout(self.outputLayout)


        self.contentLayout.addWidget(self.filesGroup)

        self.faceSwapGroup = QGroupBox(self.scrollAreaWidgetContents)
        self.faceSwapGroup.setObjectName(u"faceSwapGroup")
        self.faceSwapGroup.setFont(font)
        self.faceSwapLayout = QVBoxLayout(self.faceSwapGroup)
        self.faceSwapLayout.setSpacing(15)
        self.faceSwapLayout.setObjectName(u"faceSwapLayout")
        self.modelSelectLayout = QHBoxLayout()
        self.modelSelectLayout.setObjectName(u"modelSelectLayout")
        self.modelLabel = QLabel(self.faceSwapGroup)
        self.modelLabel.setObjectName(u"modelLabel")
        self.modelLabel.setMinimumSize(QSize(120, 0))
        self.modelLabel.setFont(font)

        self.modelSelectLayout.addWidget(self.modelLabel)

        self.modelCombo = QComboBox(self.faceSwapGroup)
        self.modelCombo.addItem("")
        self.modelCombo.addItem("")
        self.modelCombo.addItem("")
        self.modelCombo.addItem("")
        self.modelCombo.setObjectName(u"modelCombo")

        self.modelSelectLayout.addWidget(self.modelCombo)


        self.faceSwapLayout.addLayout(self.modelSelectLayout)

        self.modelPathLayout = QHBoxLayout()
        self.modelPathLayout.setObjectName(u"modelPathLayout")
        self.modelPathLabel = QLabel(self.faceSwapGroup)
        self.modelPathLabel.setObjectName(u"modelPathLabel")
        self.modelPathLabel.setMinimumSize(QSize(120, 0))
        self.modelPathLabel.setFont(font)

        self.modelPathLayout.addWidget(self.modelPathLabel)

        self.modelPathEdit = QLineEdit(self.faceSwapGroup)
        self.modelPathEdit.setObjectName(u"modelPathEdit")
        self.modelPathEdit.setReadOnly(True)

        self.modelPathLayout.addWidget(self.modelPathEdit)

        self.modelBrowseButton = QPushButton(self.faceSwapGroup)
        self.modelBrowseButton.setObjectName(u"modelBrowseButton")
        self.modelBrowseButton.setMinimumSize(QSize(100, 0))
        self.modelBrowseButton.setEnabled(False)

        self.modelPathLayout.addWidget(self.modelBrowseButton)


        self.faceSwapLayout.addLayout(self.modelPathLayout)

        self.modelProviderLayout = QHBoxLayout()
        self.modelProviderLayout.setObjectName(u"modelProviderLayout")
        self.modelProviderLabel = QLabel(self.faceSwapGroup)
        self.modelProviderLabel.setObjectName(u"modelProviderLabel")
        self.modelProviderLabel.setMinimumSize(QSize(120, 0))
        self.modelProviderLabel.setFont(font)

        self.modelProviderLayout.addWidget(self.modelProviderLabel)

        self.modelProviderCombo = QComboBox(self.faceSwapGroup)
        self.modelProviderCombo.addItem("")
        self.modelProviderCombo.addItem("")
        self.modelProviderCombo.setObjectName(u"modelProviderCombo")
        self.modelProviderCombo.setMinimumSize(QSize(200, 0))

        self.modelProviderLayout.addWidget(self.modelProviderCombo)

        self.modelProviderSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.modelProviderLayout.addItem(self.modelProviderSpacer)


        self.faceSwapLayout.addLayout(self.modelProviderLayout)


        self.contentLayout.addWidget(self.faceSwapGroup)

        self.upscalerGroup = QGroupBox(self.scrollAreaWidgetContents)
        self.upscalerGroup.setObjectName(u"upscalerGroup")
        self.upscalerGroup.setFont(font)
        self.upscalerLayout = QVBoxLayout(self.upscalerGroup)
        self.upscalerLayout.setSpacing(15)
        self.upscalerLayout.setObjectName(u"upscalerLayout")
        self.upscalerEnableLayout = QHBoxLayout()
        self.upscalerEnableLayout.setObjectName(u"upscalerEnableLayout")
        self.upscalerCheckBox = QCheckBox(self.upscalerGroup)
        self.upscalerCheckBox.setObjectName(u"upscalerCheckBox")
        self.upscalerCheckBox.setFont(font)

        self.upscalerEnableLayout.addWidget(self.upscalerCheckBox)

        self.upscalerEnableSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.upscalerEnableLayout.addItem(self.upscalerEnableSpacer)


        self.upscalerLayout.addLayout(self.upscalerEnableLayout)

        self.upscalerOptionsFrame = QFrame(self.upscalerGroup)
        self.upscalerOptionsFrame.setObjectName(u"upscalerOptionsFrame")
        self.upscalerOptionsFrame.setFrameShape(QFrame.NoFrame)
        self.upscalerOptionsLayout = QVBoxLayout(self.upscalerOptionsFrame)
        self.upscalerOptionsLayout.setSpacing(15)
        self.upscalerOptionsLayout.setObjectName(u"upscalerOptionsLayout")
        self.upscalerSelectLayout = QHBoxLayout()
        self.upscalerSelectLayout.setObjectName(u"upscalerSelectLayout")
        self.upscalerLabel = QLabel(self.upscalerOptionsFrame)
        self.upscalerLabel.setObjectName(u"upscalerLabel")
        self.upscalerLabel.setMinimumSize(QSize(120, 0))
        self.upscalerLabel.setFont(font)

        self.upscalerSelectLayout.addWidget(self.upscalerLabel)

        self.upscalerCombo = QComboBox(self.upscalerOptionsFrame)
        self.upscalerCombo.addItem("")
        self.upscalerCombo.addItem("")
        self.upscalerCombo.setObjectName(u"upscalerCombo")

        self.upscalerSelectLayout.addWidget(self.upscalerCombo)

        self.upscalerSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.upscalerSelectLayout.addItem(self.upscalerSpacer)


        self.upscalerOptionsLayout.addLayout(self.upscalerSelectLayout)

        self.upscalerProviderLayout = QHBoxLayout()
        self.upscalerProviderLayout.setObjectName(u"upscalerProviderLayout")
        self.upscalerProviderLabel = QLabel(self.upscalerOptionsFrame)
        self.upscalerProviderLabel.setObjectName(u"upscalerProviderLabel")
        self.upscalerProviderLabel.setMinimumSize(QSize(120, 0))
        self.upscalerProviderLabel.setFont(font)

        self.upscalerProviderLayout.addWidget(self.upscalerProviderLabel)

        self.upscalerProviderCombo = QComboBox(self.upscalerOptionsFrame)
        self.upscalerProviderCombo.addItem("")
        self.upscalerProviderCombo.addItem("")
        self.upscalerProviderCombo.setObjectName(u"upscalerProviderCombo")
        self.upscalerProviderCombo.setMinimumSize(QSize(200, 0))

        self.upscalerProviderLayout.addWidget(self.upscalerProviderCombo)

        self.upscalerProviderSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.upscalerProviderLayout.addItem(self.upscalerProviderSpacer)


        self.upscalerOptionsLayout.addLayout(self.upscalerProviderLayout)


        self.upscalerLayout.addWidget(self.upscalerOptionsFrame)


        self.contentLayout.addWidget(self.upscalerGroup)

        self.settingsGroup = QGroupBox(self.scrollAreaWidgetContents)
        self.settingsGroup.setObjectName(u"settingsGroup")
        self.settingsGroup.setFont(font)
        self.settingsLayout = QVBoxLayout(self.settingsGroup)
        self.settingsLayout.setSpacing(15)
        self.settingsLayout.setObjectName(u"settingsLayout")
        self.resolutionLayout = QHBoxLayout()
        self.resolutionLayout.setObjectName(u"resolutionLayout")
        self.resolutionLabel = QLabel(self.settingsGroup)
        self.resolutionLabel.setObjectName(u"resolutionLabel")
        self.resolutionLabel.setMinimumSize(QSize(120, 0))
        self.resolutionLabel.setFont(font)

        self.resolutionLayout.addWidget(self.resolutionLabel)

        self.resolutionSpinBox = QSpinBox(self.settingsGroup)
        self.resolutionSpinBox.setObjectName(u"resolutionSpinBox")
        self.resolutionSpinBox.setMinimumSize(QSize(100, 0))
        self.resolutionSpinBox.setMinimum(64)
        self.resolutionSpinBox.setMaximum(512)
        self.resolutionSpinBox.setValue(128)

        self.resolutionLayout.addWidget(self.resolutionSpinBox)

        self.resolutionInfoLabel = QLabel(self.settingsGroup)
        self.resolutionInfoLabel.setObjectName(u"resolutionInfoLabel")
        self.resolutionInfoLabel.setFont(font1)

        self.resolutionLayout.addWidget(self.resolutionInfoLabel)

        self.resolutionSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.resolutionLayout.addItem(self.resolutionSpacer)


        self.settingsLayout.addLayout(self.resolutionLayout)

        self.optionsLayout = QHBoxLayout()
        self.optionsLayout.setObjectName(u"optionsLayout")
        self.mouthCheckBox = QCheckBox(self.settingsGroup)
        self.mouthCheckBox.setObjectName(u"mouthCheckBox")
        self.mouthCheckBox.setFont(font)

        self.optionsLayout.addWidget(self.mouthCheckBox)

        self.optionsSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.optionsLayout.addItem(self.optionsSpacer)


        self.settingsLayout.addLayout(self.optionsLayout)


        self.contentLayout.addWidget(self.settingsGroup)

        self.controlsLayout = QHBoxLayout()
        self.controlsLayout.setSpacing(15)
        self.controlsLayout.setObjectName(u"controlsLayout")
        self.controlsLeftSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.controlsLayout.addItem(self.controlsLeftSpacer)

        self.startButton = QPushButton(self.scrollAreaWidgetContents)
        self.startButton.setObjectName(u"startButton")
        self.startButton.setMinimumSize(QSize(200, 45))
        font2 = QFont()
        font2.setPointSize(12)
        font2.setBold(True)
        self.startButton.setFont(font2)

        self.controlsLayout.addWidget(self.startButton)

        self.stopButton = QPushButton(self.scrollAreaWidgetContents)
        self.stopButton.setObjectName(u"stopButton")
        self.stopButton.setMinimumSize(QSize(100, 45))
        self.stopButton.setEnabled(False)
        self.stopButton.setFont(font2)

        self.controlsLayout.addWidget(self.stopButton)

        self.controlsRightSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.controlsLayout.addItem(self.controlsRightSpacer)


        self.contentLayout.addLayout(self.controlsLayout)

        self.progressBar = QProgressBar(self.scrollAreaWidgetContents)
        self.progressBar.setObjectName(u"progressBar")
        self.progressBar.setVisible(False)
        self.progressBar.setValue(0)
        self.progressBar.setTextVisible(True)

        self.contentLayout.addWidget(self.progressBar)

        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.mainLayout.addWidget(self.scrollArea)


        self.retranslateUi(VideoGUI)

        QMetaObject.connectSlotsByName(VideoGUI)
    # setupUi

    def retranslateUi(self, VideoGUI):
        VideoGUI.setWindowTitle(QCoreApplication.translate("VideoGUI", u"LiveSwapping - Video Processing", None))
        self.languageLabel.setText(QCoreApplication.translate("VideoGUI", u"Language:", None))
        self.languageCombo.setItemText(0, QCoreApplication.translate("VideoGUI", u"English", None))
        self.languageCombo.setItemText(1, QCoreApplication.translate("VideoGUI", u"\u0420\u0443\u0441\u0441\u043a\u0438\u0439", None))

        self.versionLabel.setText(QCoreApplication.translate("VideoGUI", u"LiveSwapping v0.1", None))
        self.filesGroup.setTitle(QCoreApplication.translate("VideoGUI", u"Input Files", None))
        self.modeLabel.setText(QCoreApplication.translate("VideoGUI", u"Processing Mode:", None))
        self.modeCombo.setItemText(0, QCoreApplication.translate("VideoGUI", u"Image to Image", None))
        self.modeCombo.setItemText(1, QCoreApplication.translate("VideoGUI", u"Image to Video", None))

        self.sourceLabel.setText(QCoreApplication.translate("VideoGUI", u"Source Image:", None))
        self.sourceEdit.setPlaceholderText(QCoreApplication.translate("VideoGUI", u"Select face image to swap from...", None))
        self.sourceBrowseButton.setText(QCoreApplication.translate("VideoGUI", u"Browse", None))
        self.targetLabel.setText(QCoreApplication.translate("VideoGUI", u"Target Video:", None))
        self.targetEdit.setPlaceholderText(QCoreApplication.translate("VideoGUI", u"Select video to swap faces in...", None))
        self.targetBrowseButton.setText(QCoreApplication.translate("VideoGUI", u"Browse", None))
        self.outputLabel.setText(QCoreApplication.translate("VideoGUI", u"Output File:", None))
        self.outputEdit.setPlaceholderText(QCoreApplication.translate("VideoGUI", u"Specify output file path (e.g. result.mp4)...", None))
        self.outputBrowseButton.setText(QCoreApplication.translate("VideoGUI", u"Browse", None))
        self.faceSwapGroup.setTitle(QCoreApplication.translate("VideoGUI", u"Face Swap Model", None))
        self.modelLabel.setText(QCoreApplication.translate("VideoGUI", u"AI Model:", None))
        self.modelCombo.setItemText(0, QCoreApplication.translate("VideoGUI", u"reswapper128 (StyleTransfer)", None))
        self.modelCombo.setItemText(1, QCoreApplication.translate("VideoGUI", u"reswapper256 (StyleTransfer)", None))
        self.modelCombo.setItemText(2, QCoreApplication.translate("VideoGUI", u"inswapper128 (InsightFace)", None))
        self.modelCombo.setItemText(3, QCoreApplication.translate("VideoGUI", u"DFM Model (Custom)", None))

        self.modelPathLabel.setText(QCoreApplication.translate("VideoGUI", u"Model Path:", None))
        self.modelPathEdit.setPlaceholderText(QCoreApplication.translate("VideoGUI", u"Model will be downloaded automatically...", None))
        self.modelBrowseButton.setText(QCoreApplication.translate("VideoGUI", u"Browse", None))
        self.modelProviderLabel.setText(QCoreApplication.translate("VideoGUI", u"Provider:", None))
        self.modelProviderCombo.setItemText(0, QCoreApplication.translate("VideoGUI", u"CPU (Fallback)", None))
        self.modelProviderCombo.setItemText(1, QCoreApplication.translate("VideoGUI", u"CUDA (NVIDIA GPU)", None))

        self.upscalerGroup.setTitle(QCoreApplication.translate("VideoGUI", u"Face Enhancement (Upscaler)", None))
        self.upscalerCheckBox.setText(QCoreApplication.translate("VideoGUI", u"Enable Face Enhancement", None))
        self.upscalerLabel.setText(QCoreApplication.translate("VideoGUI", u"Upscaler:", None))
        self.upscalerCombo.setItemText(0, QCoreApplication.translate("VideoGUI", u"GFPGAN", None))
        self.upscalerCombo.setItemText(1, QCoreApplication.translate("VideoGUI", u"RealESRGAN", None))

        self.upscalerProviderLabel.setText(QCoreApplication.translate("VideoGUI", u"Provider:", None))
        self.upscalerProviderCombo.setItemText(0, QCoreApplication.translate("VideoGUI", u"CPU (Fallback)", None))
        self.upscalerProviderCombo.setItemText(1, QCoreApplication.translate("VideoGUI", u"CUDA (NVIDIA GPU)", None))

        self.settingsGroup.setTitle(QCoreApplication.translate("VideoGUI", u"Processing Settings", None))
        self.resolutionLabel.setText(QCoreApplication.translate("VideoGUI", u"Resolution:", None))
        self.resolutionSpinBox.setSuffix(QCoreApplication.translate("VideoGUI", u"px", None))
        self.resolutionInfoLabel.setText(QCoreApplication.translate("VideoGUI", u"Higher = better quality, slower processing", None))
        self.mouthCheckBox.setText(QCoreApplication.translate("VideoGUI", u"Retain target mouth (preserve speech)", None))
        self.startButton.setText(QCoreApplication.translate("VideoGUI", u"Start Processing", None))
        self.stopButton.setText(QCoreApplication.translate("VideoGUI", u"Stop", None))
    # retranslateUi

