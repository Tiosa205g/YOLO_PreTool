from PySide6.QtWidgets import QVBoxLayout,QWidget
from PySide6.QtCore import Qt
from qfluentwidgets_pro import (TopFluentWindow,
                                QColor,
                                FluentIcon,
                                TopNavigationItemPosition,
                                BodyLabel,
                                )
from qfluentwidgets_pro.components.widgets.acrylic_label import AcrylicLabel

class Window(TopFluentWindow):
    def __init__(self):
        super().__init__()
        self.setFixedSize(900,500)
        self.setupUI()
    def setupUI(self):
        self.setWindowTitle("YOLO目标检测图片推理工具")
        
        self.titleBar.maxBtn.hide()
        self.titleBar.setDoubleClickEnabled(False) # 去除双击放大
        
        
        self.background = AcrylicLabel(20, QColor(105, 114, 168, 102),parent=self)
        self.background.setImage('ui/img/background.jpg')
        self.background.lower()
        
        self.importWidget = self._createImportWidget()
        self.importWidget.setObjectName("importWidget")
        self.addSubInterface(self.importWidget,
                            FluentIcon.DOWNLOAD,
                            "载入",
                            TopNavigationItemPosition.LEFT,
                            expanded=True
                            )
        
        self.preWidget = self._createPreWidget()
        self.preWidget.setObjectName("preWidget")
        self.addSubInterface(self.preWidget,
                             FluentIcon.PHOTO,
                             "推理",
                             TopNavigationItemPosition.LEFT,
                             expanded=True
                             )
    def _createImportWidget(self)->QWidget:
        w = QWidget(self)
        w.label = BodyLabel("ImportWidget",w)
        return w
    def _createPreWidget(self)->QWidget:
        w = QWidget()
        w.label = BodyLabel("PreWidget",w) 
        return w 
        


