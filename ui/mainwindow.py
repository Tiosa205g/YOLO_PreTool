import torch
from .util import *
from matplotlib import pyplot as plt
from PIL import Image
from PySide6.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QHeaderView,
    QTableWidgetItem,

)
from PySide6.QtCore import Qt
from qfluentwidgets_pro import (
    TopFluentWindow,
    QColor,
    FluentIcon,
    TopNavigationItemPosition,
    BodyLabel,
    DropSingleFileWidget,
    DropSingleFolderWidget,
    LineTableWidget,
    ImageLabel,
    ComboBox,
    RoundPushButton,
    Toast,
    SimpleCardWidget,
    ProgressBar,
    ToolTipSlider,
    RangeSlider,
    
)
from qfluentwidgets_pro.components.widgets.acrylic_label import AcrylicLabel

yolo_defaults = {
    "conf": 0.25,
    "iou": 0.7,
    "imgsz": 640,
    "rect": True,
    "half": False,
    "device": None,
    "batch": 1,
    "max_det": 300,
    "vid_stride": 1,
    "stream_buffer": False,
    "visualize": False,
    "augment": False,
    "agnostic_nms": False,
    "classes": None,
    "retina_masks": False,
    "embed": None,
    "project": None,
    "name": None,
    "stream": False,
    "verbose": True,
    "compile": False,
    "end2end": None,
}

class Window(TopFluentWindow):
    def __init__(self):
        super().__init__()
        self.setFixedSize(900, 500)
        self.setupUI()

    def setupUI(self):
        self.setWindowTitle("YOLO目标检测图片推理工具")

        self.titleBar.maxBtn.hide()
        self.titleBar.setDoubleClickEnabled(False)  # 去除双击放大

        self.background = AcrylicLabel(20, QColor(105, 114, 168, 102), parent=self)
        self.background.setImage("ui/img/background.jpg")
        self.background.lower()

        self.importWidget = self._createImportWidget()
        self.importWidget.setObjectName("importWidget")
        self.addSubInterface(
            self.importWidget,
            FluentIcon.DOWNLOAD,
            "载入",
            TopNavigationItemPosition.LEFT,
            expanded=True,
        )

        self.preWidget = self._createPreWidget()
        self.preWidget.setObjectName("preWidget")
        self.addSubInterface(
            self.preWidget,
            FluentIcon.PHOTO,
            "推理",
            TopNavigationItemPosition.LEFT,
            expanded=True,
        )

    def _createImportWidget(self) -> QWidget:
        w = QWidget(self)
        mainLayout = QHBoxLayout(w)

        vLayout = QVBoxLayout()
        modelLabel = BodyLabel("模型文件:")
        modelSelect = DropSingleFileWidget()
        modelSelect.setFixedHeight(125)
        dataLabel = BodyLabel("数据集文件:")
        dataSelect = DropSingleFolderWidget()
        dataSelect.setFixedHeight(125)
        vLayout.addStretch()
        vLayout.addWidget(modelLabel)
        vLayout.addWidget(modelSelect)
        vLayout.addSpacing(20)
        vLayout.addWidget(dataLabel)
        vLayout.addWidget(dataSelect)
        vLayout.addStretch()

        vLayout2 = QVBoxLayout()
        vLayout2.addSpacing(20)
        argTable = LineTableWidget()
        argTable.editItem
        argTable.setFixedWidth(500)
        argTable.setColumnCount(2)
        argTable.setRowCount(22)
        argTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        argTable.setHorizontalHeaderLabels(["参数名", "值"])
        set_table_scorllbar_style(argTable)
        argTable.delegate.setUseTransparentRows(True)
        for i, (key, val) in enumerate(yolo_defaults.items()):
            argTable.setItem(i, 0, QTableWidgetItem(key))
            argTable.setItem(i, 1, QTableWidgetItem(str(val)))
        vLayout2.addWidget(argTable)
        vLayout2.addSpacing(20)

        mainLayout.addSpacing(20)
        mainLayout.addLayout(vLayout)
        mainLayout.addSpacing(30)

        mainLayout.addLayout(vLayout2)
        mainLayout.addSpacing(20)

        return w

    def _createPreWidget(self) -> QWidget:
        w = QWidget()
        hLayout = QHBoxLayout(w)
        card1 = SimpleCardWidget(w)
        hLayout.addWidget(card1)

        vLayout = QVBoxLayout(card1)
        vLayout.addStretch()
        imageLayout = QHBoxLayout()

        imageDis = ImageLabel(card1)
        imageDis.setImage("ui/img/test.jpg")
        imageDis.setFixedSize(175, 175)
        imageDis.setCursor(Qt.CursorShape.PointingHandCursor)
        imageDis.clicked.connect(lambda: show_image_in_plt(imageDis.image))
        imageLayout.addStretch()
        imageLayout.addWidget(imageDis)
        imageLayout.addStretch()
        vLayout.addLayout(imageLayout)
        hLayout2 = QHBoxLayout()

        vLayout.addLayout(hLayout2)
        imageChoose = ComboBox(card1)
        imageChoose.setPlaceholderText("选择图片")
        deviceChoose = ComboBox(card1)
        deviceChoose.setPlaceholderText("选择设备")
        hLayout2.addWidget(imageChoose)
        hLayout2.addWidget(deviceChoose)

        sureLabel = BodyLabel(card1)
        sureLabel.setText("置信度:0.00")
        sureSlider = ToolTipSlider(Qt.Horizontal, card1)
        sureSlider.valueChanged.connect(
            lambda x: sureLabel.setText(f"置信度:{x*0.01:.2f}")
        )
        sureSlider.setRange(0, 100)
        preButton = RoundPushButton(card1)
        preButton.setFixedHeight(50)
        preButton.setCursor(Qt.CursorShape.PointingHandCursor)
        preButton.setText("推理")
        preAllButton = RoundPushButton(card1)
        preAllButton.setFixedHeight(50)
        preAllButton.setCursor(Qt.CursorShape.PointingHandCursor)
        preAllButton.setText("全部推理并输出")
        vLayout.addWidget(sureLabel)
        vLayout.addWidget(sureSlider)
        vLayout.addWidget(preButton)
        vLayout.addWidget(preAllButton)
        vLayout.addStretch()

        card2 = SimpleCardWidget(w)
        vLayout2 = QVBoxLayout(card2)

        imageSureLayout = QHBoxLayout()
        imageDis2 = ImageLabel(card2)
        imageDis2.setImage("ui/img/res.jpg")
        imageDis2.setFixedSize(175, 175)
        imageDis2.setCursor(Qt.CursorShape.PointingHandCursor)
        imageDis2.clicked.connect(lambda: show_image_in_plt(imageDis2.image))

        sureRangeSlider = RangeSlider(Qt.Orientation.Vertical, card2)
        sureRangeSlider.setRange(0, 100)
        sureRangeSlider.minValueChanged.connect(lambda x: print(f"最小值变化:{x}"))
        sureRangeSlider.maxValueChanged.connect(lambda x: print(f"最大值变化:{x}"))
        imageSureLayout.addStretch()
        imageSureLayout.addWidget(imageDis2)
        imageSureLayout.addStretch()
        imageSureLayout.addWidget(sureRangeSlider)

        
        vLayout2.addLayout(imageSureLayout)
        tableLayout = QHBoxLayout()
        labelTable = LineTableWidget(card2)
        labelTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        labelTable.setEditTriggers(labelTable.EditTrigger.NoEditTriggers)
        labelTable.setColumnCount(2)
        labelTable.setRowCount(20)
        labelTable.setHorizontalHeaderLabels(["标签名","数量"])
        set_table_scorllbar_style(labelTable)

        resTable = LineTableWidget(card2)
        resTable.setColumnCount(2)
        resTable.setRowCount(20)
        resTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        resTable.setEditTriggers(resTable.EditTrigger.NoEditTriggers)
        resTable.setHorizontalHeaderLabels(["序号", "置信度"])
        set_table_scorllbar_style(resTable)
        
        tableLayout.addWidget(labelTable)
        tableLayout.addWidget(resTable)
        
        vLayout2.addLayout(tableLayout)
        hLayout.addWidget(card2)

        return w
