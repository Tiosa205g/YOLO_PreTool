import torch
from matplotlib import pyplot as plt
from PIL import Image
from PySide6.QtWidgets import (
    QWidget,
    QApplication,
    QStyleOptionViewItem,
    QAbstractItemView,
    QStyledItemDelegate,
    QVBoxLayout,
)
from PySide6.QtGui import QImage, QImageReader, QMovie, QPixmap
from PySide6.QtCore import Qt, QModelIndex, Signal
from qfluentwidgets_pro import (
    QColor,
    LineTableWidget,
    TableWidget,
    LineEdit,
    TableItemDelegate,
    Toast,
    ToastPosition,
    ComboBox,
    SmoothScrollDelegate,
    ImageLabel,
    ProgressRing,
)
from qfluentwidgets_pro.components.widgets.combo_box import ComboBoxMenu


def get_available_devices():
    """返回所有可用设备列表 [cpu, cuda:0, cuda:1, ...]"""
    devices = ["cpu"]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(f"cuda:{i}")
    return devices


def set_scorll_bar_style(scroll: SmoothScrollDelegate):
    scroll.vScrollBar.setHandleColor("#EFECEC", "#EFECEC")
    scroll.vScrollBar.setGrooveColor(QColor(0, 0, 0, 0), QColor(0, 0, 0, 0))
    scroll.vScrollBar.setArrowColor("#EFECEC", "#EFECEC")


def set_table_style(table: TableWidget):
    table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
    set_scorll_bar_style(table.scrollDelagate)
    table.myd = MyTableItemDelegate(table)
    table.setItemDelegate(table.myd)


def show_image_in_plt(imglabel: ImageLabel):
    if not imglabel.pixmap() or imglabel.pixmap().isNull():
        return
    pil_img = Image.fromqimage(imglabel.image)

    plt.figure()
    plt.imshow(pil_img)
    plt.axis("off")
    plt.show()


def isnum(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def create_toast(
    title="",
    content="",
    duration=2000,
    iscloseable=True,
    position=ToastPosition.TOP_RIGHT,
    parent=None,
    toastColor=QColor(0, 0, 0, 0),
    backgroundColor=QColor(0, 0, 0, 0),
):
    t = Toast.new(
        title,
        content,
        duration,
        iscloseable,
        position,
        toastColor=toastColor,
        parent=parent,
        backgroundColor=backgroundColor,
    )
    t.setObjectName("toast")
    t.setStyleSheet("""
#toast{
    border: 1px solid #888888; 
    border-radius: 8px;
}
""")
    return t


def print_widget_tree(root: QWidget, indent: int = 0):
    """
    递归打印控件树（查看嵌套结构 + 类名/对象名）
    :param root: 根控件（主窗口/目标组件）
    :param indent: 缩进层级（展示嵌套）
    """
    if not root:
        return

    # 打印当前控件信息
    indent_str = "  " * indent
    class_name = root.metaObject().className()  # 控件类名（QSS 用）
    obj_name = root.objectName()  # 控件对象名（QSS 用）
    visible = "✅ 可见" if root.isVisible() else "❌ 隐藏"
    print(f"{indent_str}└── [{class_name}]  objectName: '{obj_name}'  {visible}")

    # 递归遍历所有子控件
    for child in root.children():
        if isinstance(child, QWidget):  # 只遍历可视化控件
            print_widget_tree(child, indent + 1)


def add_style_sheet(widget: QWidget, styleSheet: str):
    qss = widget.styleSheet() + "\n" + styleSheet
    widget.setStyleSheet(qss)


class MyTableItemDelegate(TableItemDelegate):
    def __init__(self, parent):
        super().__init__(parent)

    def createEditor(
        self, parent: QWidget, option: QStyleOptionViewItem, index: QModelIndex
    ) -> QWidget:
        lineEdit = LineEdit(parent)
        lineEdit.setProperty("transparent", True)
        lineEdit.setStyle(QApplication.style())
        lineEdit.setStyleSheet("""
LineEdit, TextEdit, PlainTextEdit, TextBrowser {
    background-color: transparent; /* 核心：纯透明背景 */
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-bottom: 1px solid rgba(255, 255, 255, 0.5442);
    border-radius: 5px;
    padding: 0px 10px;
    color: white;
    selection-background-color: --ThemeColorLight1;
    selection-color: black;
}

/* 多行文本框内边距 */
TextEdit, PlainTextEdit, TextBrowser  {
    padding: 2px 3px 2px 8px;
}


LineEdit:hover, TextEdit:hover, PlainTextEdit:hover, TextBrowser:hover {
    background: rgba(255, 255, 255, 0.0837);
}

/* 聚焦状态 - 全部改为透明背景 */
LineEdit:focus[transparent=true] {
    background-color: transparent; /* 纯透明 */
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

LineEdit[transparent=false]:focus {
    background-color: transparent; /* 纯透明 */
}

TextEdit:focus, PlainTextEdit:focus, TextBrowser:focus {
    border-bottom: 1px solid --ThemeColorLight1;
    background-color: transparent; /* 纯透明 */
}

/* 禁用状态 - 背景透明 */
LineEdit:disabled, TextEdit:disabled, PlainTextEdit:disabled, TextBrowser:disabled {
    color: rgba(255, 255, 255, 92);
    background-color: transparent; /* 纯透明 */
    border: 1px solid rgba(255, 255, 255, 0.0698);
}

/* 输入框按钮 */
#lineEditButton {
    background-color: transparent;
    border-radius: 4px;
    margin: 0;
}

#lineEditButton:hover {
    background-color: rgba(255, 255, 255, 9);
}

#lineEditButton:pressed {
    background-color: rgba(255, 255, 255, 6);
}

/* 密码框组件 - 背景完全透明 */
PinBoxLineEdit {
    background-color: transparent; /* 纯透明背景 */
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-bottom: 1px solid rgba(255, 255, 255, 0.5442);
    border-radius: 5px;
    padding: 0px;
    color: white;
}

PinBoxLineEdit:hover {
    background: rgba(255, 255, 255, 0.0837);
}

PinBoxLineEdit:focus {
    background-color: transparent; /* 纯透明 */
    border: 2px solid --ThemeColorPrimary;
}""")
        lineEdit.setClearButtonEnabled(True)
        lineEdit.textChanged.connect(self.textChanged)

        # print("创建编辑框")
        return lineEdit

    def textChanged(self, content):
        table: LineTableWidget = self.parent()
        selection = table.selectedItems()
        if len(selection) == 1:
            table.takeItem(selection[0].row(), selection[0].column())
        # print(f"编辑内容:{content}")


class MyComboBox(ComboBox):
    """优化菜单删除机制，手动删除更新菜单"""

    def __init__(self, parent=None):
        super().__init__(parent)

    def clear(self):
        self.dropMenu = None
        return super().clear()

    def _createComboMenu(self):
        menu = ComboBoxMenu(self)
        # print(menu.styleSheet())
        menu.setStyleSheet("""MenuActionListWidget {
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 9px;
    background-color: rgba(80, 80, 80, 0.5);
    outline: none;
}

MenuActionListWidget::item {
    padding-left: 10px;
    padding-right: 10px;
    margin-left: 6px;
    margin-right: 6px;
    border-radius: 5px;

    border: none;
    border-left: none;   

    color: white;
}

MenuActionListWidget::item:hover {
    background-color: rgba(255, 255, 255, 0.08);
    border-left: none;
}

MenuActionListWidget::item:selected {
    background-color: rgba(255, 255, 255, 0.08);
    border-left: none;  
}""")
        set_scorll_bar_style(menu.view.scrollDelegate)
        return menu

    def _showComboMenu(self):
        super()._showComboMenu()
        # print_widget_tree(self)


class MyImageLabel(ImageLabel):
    """去除自动更改组件大小"""

    def __init__(self, parent=None):
        super().__init__(parent)

    def setImage(self, image=None):
        self.image = image or QImage()

        if isinstance(image, str):
            reader = QImageReader(image)
            if reader.supportsAnimation():
                self.setMovie(QMovie(image))
            else:
                self.image = reader.read()
        elif isinstance(image, QPixmap):
            self.image = image.toImage()

        # self.setFixedSize(self.image.size())
        self.update()


class ProgressToast(Toast):
    finished = Signal()

    def __init__(
        self,
        title="",
        content="",
        orient=Qt.Horizontal,
        toastColor=QColor(0, 0, 0, 0),
        parent=None,
        position=ToastPosition.TOP_RIGHT,
        isClosable=False,
        duration=-1,
        backgroundColor=QColor(0, 0, 0, 0),
        useThemeColor=False,
    ):
        super().__init__(
            title,
            content,
            duration,
            isClosable,
            position,
            orient,
            toastColor,
            parent,
            backgroundColor,
            useThemeColor,
        )
        self.setObjectName("toast")
        self.setStyleSheet("""
        #toast{
            border: 1px solid #888888; 
            border-radius: 8px;
        }
        """)
        self.progressRing = ProgressRing(self)
        self.progressRing.setTextVisible(True)
        self.progressRing.setFixedSize(50,50)
        self.addWidget(self.progressRing, alignment=Qt.AlignmentFlag.AlignLeft)
        self.force_text_vertical()
        layout = self.hBoxLayout

        text = self.textLayout
        widget = self.widgetLayout
        # 调换顺序
        layout.removeItem(text)
        layout.removeItem(widget)

        layout.insertLayout(0, widget)
        layout.insertLayout(1, text)
        

    def setProgress(self, val: int):
        if val >= 100:
            self.finished.emit()
            self.close()
            return
        self.progressRing.setValue(val)
    def force_text_vertical(self):
        old = self.textLayout
        items = []

        while old.count():
            item = old.takeAt(0)
            items.append(item)

        new = QVBoxLayout()
        new.setSpacing(old.spacing())
        new.setContentsMargins(old.contentsMargins())
        new.setAlignment(old.alignment())

        for item in items:
            if item.widget():
                new.addWidget(item.widget())
            elif item.layout():
                new.addLayout(item.layout())
            else:
                new.addItem(item)

        parent_layout = self.hBoxLayout

        parent_layout.removeItem(old)
        parent_layout.insertLayout(0, new)

        self.textLayout = new
