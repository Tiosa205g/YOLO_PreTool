import torch
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

def get_available_devices():
    """返回所有可用设备列表 [cpu, cuda:0, cuda:1, ...]"""
    devices = ["cpu"]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(f"cuda:{i}")
    return devices

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
        w.mainLayout = QHBoxLayout(w)

        vLayout = QVBoxLayout()
        w.modelLabel = BodyLabel("模型文件:")
        w.modelSelect = DropSingleFileWidget()
        w.modelSelect.setFixedHeight(125)
        w.dataLabel = BodyLabel("数据集文件:")
        w.dataSelect = DropSingleFolderWidget()
        w.dataSelect.setFixedHeight(125)
        vLayout.addStretch()
        vLayout.addWidget(w.modelLabel)
        vLayout.addWidget(w.modelSelect)
        vLayout.addSpacing(20)
        vLayout.addWidget(w.dataLabel)
        vLayout.addWidget(w.dataSelect)
        vLayout.addStretch()

        vLayout2 = QVBoxLayout()
        vLayout2.addSpacing(20)
        w.argTable = LineTableWidget(w)
        w.argTable.setFixedWidth(500)
        w.argTable.setColumnCount(2)
        w.argTable.setRowCount(22)
        w.argTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        w.argTable.setHorizontalHeaderLabels(["参数名", "值"])
        for i, (key, val) in enumerate(yolo_defaults.items()):
            w.argTable.setItem(i, 0, QTableWidgetItem(key))
            w.argTable.setItem(i, 1, QTableWidgetItem(str(val)))

        vLayout2.addWidget(w.argTable)
        vLayout2.addSpacing(20)

        w.mainLayout.addSpacing(20)
        w.mainLayout.addLayout(vLayout)
        w.mainLayout.addSpacing(30)

        w.mainLayout.addLayout(vLayout2)
        w.mainLayout.addSpacing(20)

        return w

    def _createPreWidget(self) -> QWidget:
        w = QWidget()
        w.label = BodyLabel("PreWidget", w)
        return w
