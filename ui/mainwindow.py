import torch, os, math
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from pathlib import Path
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
from PySide6.QtCore import Qt, Signal, QThread
from qfluentwidgets_pro import (
    TopFluentWindow,
    QColor,
    FluentIcon,
    TopNavigationItemPosition,
    BodyLabel,
    DropSingleFileWidget,
    DropSingleFolderWidget,
    LineTableWidget,
    RoundPushButton,
    SimpleCardWidget,
    Slider,
    RangeSlider,
)
from qfluentwidgets_pro.components.widgets.acrylic_label import AcrylicLabel

exts = [
    ".avif",
    ".bmp",
    ".dng",
    ".heic",
    ".heif",
    ".jp2",
    ".jpeg",
    ".jpg",
    ".mpo",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
]


class Window(TopFluentWindow):
    model_changed = Signal(str)
    data_changed = Signal(str)
    model_path = ""
    data_path = ""
    yolo_args = {
        "iou": 0.7,
        "imgsz": 640,
        "rect": True,
        "half": False,
        "batch": 1,
        "max_det": 300,
        "visualize": False,
        "augment": False,
        "agnostic_nms": False,
        "retina_masks": False,
        "embed": None,
        "stream": False,
        "verbose": True,
        "compile": False,
        "end2end": None,
    }
    names = {}
    cls = {}  # {class_name:[box1,box2]}
    now_res = None
    now_name = ""
    now_boxs = []  # [box1,box2]
    files_path = []

    def __init__(self):
        super().__init__()
        self.setFixedSize(900, 500)
        self.setVisible(False)
        self.setupUI()

    def onBackgroundLoaded(self):
        """背景图加载完成后显示窗口并修复布局"""
        self.show()  # 显示窗口
        self.update()  # 强制刷新绘制
        self.repaint()  # 重绘界面
        self.adjustSize()  # 自适应大小（关键）
        self.resize(self.sizeHint())  # 强制恢复固定大小/正确尺寸

    def setupUI(self):
        self.setWindowTitle("YOLO目标检测图片推理工具")

        self.titleBar.maxBtn.hide()
        self.titleBar.setDoubleClickEnabled(False)  # 去除双击放大

        self.background = AcrylicLabel(20, QColor(105, 114, 168, 102), parent=self)
        self.background.setImage(get_resource_path("img/background.png"))
        self.background.blurThread.blurFinished.connect(self.onBackgroundLoaded)
        self.background.lower()  # 最下层

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

        self.stackedWidget.currentChanged.connect(self.widgetChanged)

    def _createImportWidget(self) -> QWidget:
        w = QWidget(self)
        mainLayout = QHBoxLayout(w)

        vLayout = QVBoxLayout()
        modelLabel = BodyLabel("模型文件:")
        modelLabel.setScaledContents(True)
        modelLabel.setFixedWidth(325)
        modelSelect = DropSingleFileWidget()
        modelSelect.setLabelText("请拖入模型文件")
        modelSelect.setFileFilter("yolo模型文件 (*.pt);;任意文件 (*.*)")
        self.model_changed.connect(modelLabel.setText)
        modelSelect.selectionChange.connect(self.modelChanged)
        modelSelect.draggedChange.connect(self.modelChanged)
        modelSelect.setFixedHeight(125)
        dataLabel = BodyLabel("数据集文件:")
        dataLabel.setScaledContents(True)
        dataLabel.setFixedWidth(325)
        dataSelect = DropSingleFolderWidget()
        dataSelect.setLabelText("请拖入数据文件夹")
        self.data_changed.connect(dataLabel.setText)
        dataSelect.selectionChange.connect(self.dataChanged)
        dataSelect.draggedChange.connect(self.dataChanged)
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
        self.argTable = LineTableWidget()
        # argTable.setFixedWidth(500)
        self.argTable.setColumnCount(2)
        self.argTable.setRowCount(len(self.yolo_args))
        self.argTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.argTable.setHorizontalHeaderLabels(["参数名", "值"])
        self.argTable.cellChanged.connect(self.argTableChanged)
        set_table_style(self.argTable)
        self.argTable.delegate.setUseTransparentRows(True)
        for i, (key, val) in enumerate(self.yolo_args.items()):
            self.argTable.setItem(i, 0, QTableWidgetItem(key))
            self.argTable.setItem(i, 1, QTableWidgetItem(str(val)))
        vLayout2.addWidget(self.argTable)
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

        self.imageDis = MyImageLabel(card1)
        self.imageDis.setStyleSheet("""
    background-color: rgba(128,128,128,50);
""")
        self.imageDis.setFixedSize(175, 175)
        self.imageDis.setCursor(Qt.CursorShape.PointingHandCursor)
        self.imageDis.clicked.connect(lambda: show_image_in_plt(self.imageDis))
        imageLayout.addStretch()
        imageLayout.addWidget(self.imageDis)
        imageLayout.addStretch()
        vLayout.addLayout(imageLayout)
        hLayout2 = QHBoxLayout()

        vLayout.addLayout(hLayout2)
        self.imageChoose = MyComboBox(card1)
        self.imageChoose.setPlaceholderText("选择图片")
        self.imageChoose.setMaxVisibleItems(4)
        self.imageChoose.currentIndexChanged.connect(self.imageChanged)
        # self.imageChoose.activated.connect(partial(print_widget_tree,self.imageChoose))
        self.deviceChoose = MyComboBox(card1)
        self.deviceChoose.setPlaceholderText("选择设备")
        self.deviceChoose.setMaxVisibleItems(4)
        hLayout2.addWidget(self.imageChoose)
        hLayout2.addWidget(self.deviceChoose)

        sureLabel = BodyLabel(card1)
        # sureLabel.setText("置信度:0.00")
        self.sureSlider = Slider(Qt.Horizontal, card1)
        self.sureSlider.valueChanged.connect(
            lambda x: sureLabel.setText(f"最低置信度:{x*0.01:.2f}")
        )
        self.sureSlider.setRange(0, 100)
        self.sureSlider.setValue(25)
        self.preButton = RoundPushButton(card1)
        self.preButton.setFixedHeight(50)
        self.preButton.setCursor(Qt.CursorShape.PointingHandCursor)
        self.preButton.setText("推理")
        self.preButton.clicked.connect(self.preButtonClicked)
        self.preAllButton = RoundPushButton(card1)
        self.preAllButton.setFixedHeight(50)
        self.preAllButton.setCursor(Qt.CursorShape.PointingHandCursor)
        self.preAllButton.setText("全部推理并输出")
        self.preAllButton.clicked.connect(self.preAllButtonClicked)
        vLayout.addWidget(sureLabel)
        vLayout.addWidget(self.sureSlider)
        vLayout.addWidget(self.preButton)
        vLayout.addWidget(self.preAllButton)
        vLayout.addStretch()

        card2 = SimpleCardWidget(w)
        vLayout2 = QVBoxLayout(card2)

        imageSureLayout = QHBoxLayout()
        self.imageDis2 = MyImageLabel(card2)
        self.imageDis2.setStyleSheet("""
    background-color: rgba(128,128,128,50);
""")
        self.imageDis2.setFixedSize(175, 175)
        self.imageDis2.setCursor(Qt.CursorShape.PointingHandCursor)
        self.imageDis2.clicked.connect(lambda: show_image_in_plt(self.imageDis2))

        self.sureRangeSlider = RangeSlider(Qt.Orientation.Vertical, card2)
        self.sureRangeSlider.setRange(0, 100)
        self.sureRangeSlider.rangeChanged.connect(self.sureRangeChanged)
        imageSureLayout.addStretch()
        imageSureLayout.addWidget(self.imageDis2)
        imageSureLayout.addStretch()
        imageSureLayout.addWidget(self.sureRangeSlider)

        vLayout2.addLayout(imageSureLayout)
        tableLayout = QHBoxLayout()
        self.labelTable = LineTableWidget(card2)
        self.labelTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.labelTable.setEditTriggers(self.labelTable.EditTrigger.NoEditTriggers)
        self.labelTable.setColumnCount(2)
        # self.labelTable.setRowCount(20)
        self.labelTable.setHorizontalHeaderLabels(["标签", "数量"])
    
        self.labelTable.cellClicked.connect(self.currentLabelChanged)
        self.labelTable.cellDoubleClicked.connect(self.labelDoubleClicked)
        set_table_style(self.labelTable)

        self.resTable = LineTableWidget(card2)
        self.resTable.setColumnCount(2)
        # self.resTable.setRowCount(20)
        self.resTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.resTable.setEditTriggers(self.resTable.EditTrigger.NoEditTriggers)
        self.resTable.setHorizontalHeaderLabels(["标签", "置信度"])
        self.resTable.cellClicked.connect(self.currentResChanged)
        self.resTable.cellDoubleClicked.connect(self.resDoubleClicked)
        set_table_style(self.resTable)

        tableLayout.addWidget(self.labelTable)
        tableLayout.addWidget(self.resTable)

        vLayout2.addLayout(tableLayout)
        hLayout.addWidget(card2)

        return w

    def modelChanged(self, path):
        if len(path) == 0:
            create_toast("注意⚠️：", "请确认是否正确选择", parent=self)
            return
        self.model_path = path[0]
        self.model_changed.emit(f"模型文件:{self.model_path}")
        create_toast("成功✅", f"载入模型文件{self.model_path}", parent=self)
        print(f"模型:{self.model_path}")

    def dataChanged(self, path):
        self.data_path = path
        if len(self.data_path) == 0 or self.data_path[0] == "":
            create_toast("注意⚠️：", "请确认是否正确选择", parent=self)
            # Toast.warning("注意⚠️：", "请确认是否正确选择", duration=1500, parent=self)
            return

        self.data_changed.emit(f"数据集文件:{self.data_path[0]}")
        create_toast("成功✅", f"载入数据文件夹{self.data_path[0]}", parent=self)
        print(f"数据集:{self.data_path[0]}")

    def argTableChanged(self, row, col):
        keyItem = self.argTable.item(row, 0)
        valItem = self.argTable.item(row, 1)
        if hasattr(keyItem, "text") and hasattr(valItem, "text"):
            key = keyItem.text()
            val = valItem.text()
            valList = {"true": True, "false": False, "none": None}
            val = (
                valList[val.lower()]
                if val.lower() in valList
                else (
                    val
                    if not isnum(val)
                    else float(val) if float(val) != int(float(val)) else int(val)
                )
            )
            self.yolo_args[key] = val
            print(f"{key} => {val}")

    def widgetChanged(self, index):

        widget = self.stackedWidget.widget(index)
        # print(f"widgetChanged:{self.data_path}")
        if (
            widget.objectName() == "preWidget"
            and len(self.data_path) != 0
            and self.data_path[0] != ""
        ):
            self.imageChoose.clear()
            self.deviceChoose.clear()
            self.files_path.clear()
            for file in Path(self.data_path[0]).glob("*"):
                if file.is_file() and file.suffix.lower() in exts:  # 筛选图片文件
                    imgPath = str(file.resolve(False))
                    self.files_path.append(imgPath)
                    self.imageChoose.addItem(
                        file.name, imgPath, imgPath
                    )  # 文件名 userdata:绝对路径 str

            self.deviceChoose.addItems(get_available_devices())

            print(f"设备列表{get_available_devices()}")

    def imageChanged(self, index):
        self.imageDis.setImage(self.imageChoose.currentData())  # str 路径

    def preButtonClicked(self):
        if self.model_path == "" or self.data_path == "":
            create_toast("注意⚠️：", "请确认是否选择模型以及数据集文件夹", parent=self)
            return
        # model = YOLO(self.model_path)
        # res = model.predict(
        #     self.imageChoose.currentData(),
        #     device=self.deviceChoose.currentText(),
        #     conf=float(self.sureSlider.value()) / 100,
        #     **self.yolo_args,
        # )

        def finish(res):
            speed = res[0].speed
            create_toast(
                "推理成功✅",
                f"预处理用时:{speed['preprocess']:.1f}ms, 推理用时:{speed['inference']:.1f}ms, 后处理用时:{speed['postprocess']:.1f}ms",
                parent=self,
            )
            self.now_res = res[0]
            img_bgr = res[0].plot()
            img_rgb = Image.fromarray(img_bgr[..., ::-1])  # bgr->rgb
            self.imageDis2.setImage(img_rgb.toqimage())

            self.labelTable.clearContents()
            self.labelTable.setCurrentCell(-1, -1)
            self.resTable.clearContents()
            self.resTable.setCurrentCell(-1, -1)
            self.resTable.setRowCount(0)
            self.sureRangeSlider.setRange(0, 100)
            # self.labelTable.setRowCount(len(res[0].names))
            # print(res[0].names,len(res[0].names))
            self.cls.clear()
            for box in res[0].boxes:
                self.names = res[0].names
                name = self.names[int(box.cls[0])]
                cpy: list = self.cls.get(name, [])  # 没有返回空
                cpy.append(box)
                self.cls[name] = cpy

            items = self.cls.items()
            self.labelTable.setRowCount(len(items))  # 在这里添加防止出现未识别到的label

            for i, (name, _) in enumerate(items):
                nameItem = QTableWidgetItem(name)
                numItem = QTableWidgetItem(str(len(self.cls[name])))

                self.labelTable.setItem(i, 0, nameItem)
                self.labelTable.setItem(i, 1, numItem)
            if len(items) > 0:
                self.currentLabelChanged(-1, -1)
            for w in (self.preAllButton, self.preButton, self.imageChoose, self.deviceChoose, self.sureSlider):
                w.setEnabled(True)

        t = ProgressToast("正在推理⏳:", "请耐心等待！", parent=self)

        args: dict = self.yolo_args
        args["device"] = self.deviceChoose.currentText()
        args["conf"] = float(self.sureSlider.value()) / 100
        for w in (self.preAllButton, self.preButton, self.imageChoose, self.deviceChoose, self.sureSlider):
            w.setEnabled(False)
        self.pt = PreThread(self.imageChoose.currentData(), self.model_path, args, True)
        self.pt.resultReady.connect(finish)
        self.pt.progress.connect(t.setProgress)
        self.pt.start()
        t.show()

    def preAllButtonClicked(self):
        def finish(total_time):
            create_toast(
                "推理成功✅",
                f"已保存至 runs/detect/preAll/custom 文件夹内\n总用时:{total_time:.1f}ms, 平均用时:{total_time/len(self.files_path):.1f}ms",
                parent=self,
                duration=2500,
            )
            for w in (self.preAllButton, self.preButton, self.imageChoose, self.deviceChoose, self.sureSlider):
                w.setEnabled(True)

        if self.model_path == "" or self.data_path == "":
            create_toast("注意⚠️：", "请确认是否选择模型以及数据集文件夹", parent=self)
            return
        p = ProgressToast("正在推理⏳:", "请耐心等待！", parent=self)

        args = self.yolo_args
        args["device"] = self.deviceChoose.currentText()
        args["conf"] = float(self.sureSlider.value()) / 100
        args["name"] = "custom"
        args["project"] = "preAll"
        args["exist_ok"] = True
        args["save"] = True
        for w in (self.preAllButton, self.preButton, self.imageChoose, self.deviceChoose, self.sureSlider):
            w.setEnabled(False)
        self.pt = PreThread(self.files_path, self.model_path, args)
        self.pt.timeResult.connect(finish)
        self.pt.progress.connect(p.setProgress)
        self.pt.start()

        p.show()

    def currentLabelChanged(self, row, col):
        def fill(data:list[Boxes]):
            self.resTable.clearContents()
            self.resTable.setRowCount(len(data))
            for i, box in enumerate(data):
                nameItem = QTableWidgetItem(self.names[int(box.cls[0])])
                confItem = QTableWidgetItem(str(int(float(box.conf[0]) * 100) / 100))

                self.resTable.setItem(i, 0, nameItem)
                self.resTable.setItem(i, 1, confItem)  # 截断小数点俩位之后的数
        item = self.labelTable.item(row, 0)
        # 筛选所有符合范围的box
        sure_boxs = {}
        max = self.sureRangeSlider.maxValue() / 100
        min = self.sureRangeSlider.minValue() / 100
        for label,boxs in self.cls.items():
            for box in boxs:
                conf = float(box.conf[0])
                if min <= conf <= max:
                    tmp:list = sure_boxs.get(label,[])
                    tmp.append(box)
                    sure_boxs[label] = tmp

        if hasattr(item, "text"): #说明选择了label
            label = item.text()
            boxs: list[Boxes] = self.cls[label]

            self.now_name = label
            self.now_boxs = sure_boxs.get(label,[])
        elif self.labelTable.rowCount() == 0:
            self.now_boxs.clear()
            self.resTable.clearContents()
            self.resTable.setRowCount(0)
        else: # 有label但是没有选中
            self.now_boxs.clear()
            for boxs in sure_boxs.values():
                self.now_boxs.extend(boxs)
        fill(self.now_boxs) #填充数据
        if self.now_res:
            img_bgr = drawCustomRes(self.now_res).plot(custom_boxs=self.now_boxs)
            img_rgb = Image.fromarray(img_bgr[..., ::-1])  # bgr->rgb
            self.imageDis2.setImage(img_rgb.toqimage())

    def sureRangeChanged(self, min, max):
        tmp_cls = {}
        iseq = False
        for name, boxs in self.cls.items():
            for box in boxs:
                if min / 100 <= float(box.conf[0]) <= (max + 1) / 100:
                    tmp_boxs: list = tmp_cls.get(name, [])
                    tmp_boxs.append(box)
                    tmp_cls[name] = tmp_boxs
        currentLabelId = self.labelTable.currentRow()
        currentLabelItem = self.labelTable.item(currentLabelId, 0)
        currentLabelName = (
            currentLabelItem.text() if hasattr(currentLabelItem, "text") else "#None#"
        )
        self.resTable.clearContents()  # 不管如何都要情况，否则再label变为空currentrow就成0不然就是tmp_cls没了，没办法触发清空
        self.labelTable.clearContents()
        self.labelTable.setRowCount(len(tmp_cls))
        for i, (name, boxs) in enumerate(tmp_cls.items()):
            nameItem = QTableWidgetItem(name)
            numItem = QTableWidgetItem(str(len(boxs)))
            self.labelTable.setItem(i, 0, nameItem)
            self.labelTable.setItem(i, 1, numItem)

            if currentLabelName == name:
                iseq = True
                self.labelTable.setCurrentCell(i, 0)
                self.currentLabelChanged(i, 0)  # 手动触发
                return
        if not iseq and len(tmp_cls) > 0:  
            self.labelTable.setCurrentCell(-1, -1)
        self.currentLabelChanged(-1, -1)
        # print(f"当前选中的label:{currentLabelId},name:{currentLabelName}")

    def currentResChanged(self, row, col):
        if self.resTable.currentRow() >= 0:
            img_bgr = drawCustomRes(self.now_res).plot(custom_boxs=[self.now_boxs[row]])
            img_rgb = Image.fromarray(img_bgr[..., ::-1])  # bgr->rgb
            self.imageDis2.setImage(img_rgb.toqimage())
        else:
            self.currentLabelChanged(self.labelTable.currentRow(),self.labelTable.currentColumn())
    def labelDoubleClicked(self,row,col):
        self.labelTable.setCurrentCell(-1,-1)
        self.resTable.setCurrentCell(-1,-1)
        self.currentLabelChanged(-1,-1)
    def resDoubleClicked(self,row,col):
        self.resTable.setCurrentCell(-1,-1)
        self.currentLabelChanged(-1,-1)
class PreThread(QThread):
    progress = Signal(int)
    resultReady = Signal(object)
    timeResult = Signal(float)

    def __init__(
        self,
        imgs: str | Path | int | Image.Image | list | tuple,
        model_path: str,
        args: dict = {"conf": 0.25},
        needRes=False,
    ):
        super().__init__()
        self.imgs = imgs
        self.args = args
        self.model_path = model_path
        self.needRes = needRes
        self.totalTime = 0
        self.model = YOLO(self.model_path)

    def run(self):
        res = []

        if isinstance(self.imgs, list):
            for i, img in enumerate(self.imgs):
                r = self.model.predict(
                    img,
                    **self.args,
                )
                self.totalTime += (
                    r[0].speed["preprocess"]
                    + r[0].speed["inference"]
                    + r[0].speed["postprocess"]
                )
                if not self.needRes:
                    del r
                    torch.cuda.empty_cache()
                else:
                    res.append(r[0])
                self.progress.emit(int((i + 1) / len(self.imgs) * 100))
        else:
            self.progress.emit(25)
            r = self.model.predict(
                self.imgs,
                **self.args,
            )
            self.progress.emit(80)
            self.totalTime += (
                r[0].speed["preprocess"]
                + r[0].speed["inference"]
                + r[0].speed["postprocess"]
            )
            if not self.needRes:
                del r
                torch.cuda.empty_cache()
            else:
                res.append(r[0])
            self.progress.emit(100)
        self.timeResult.emit(self.totalTime)
        if self.needRes:
            self.resultReady.emit(res)
