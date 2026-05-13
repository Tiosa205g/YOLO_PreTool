import torch
from matplotlib import pyplot as plt
from PIL import Image
from PySide6.QtWidgets import (
    QWidget,
    QApplication,
    QStyleOptionViewItem,
    QAbstractItemView
)
from PySide6.QtCore import Qt,QModelIndex
from qfluentwidgets_pro import (
    QColor,
    LineTableWidget,
    TableWidget,
    LineEdit,
    TableItemDelegate,
)
def get_available_devices():
    """返回所有可用设备列表 [cpu, cuda:0, cuda:1, ...]"""
    devices = ["cpu"]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(f"cuda:{i}")
    return devices
def set_table_scorllbar_style(table:TableWidget):
    table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
    table.scrollDelagate.vScrollBar.setHandleColor("#EFECEC","#EFECEC")
    table.scrollDelagate.vScrollBar.setGrooveColor(QColor(0, 0, 0, 0),QColor(0, 0, 0, 0))
    table.scrollDelagate.vScrollBar.setArrowColor("#EFECEC","#EFECEC")
    table.myd = MyTableItemDelegate(table)
    table.setItemDelegate(table.myd)
def show_image_in_plt(q_image):
    pil_img = Image.fromqimage(q_image)

    plt.figure()
    plt.imshow(pil_img)
    plt.axis("off")
    plt.show()
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
    def textChanged(self,content):
        table:LineTableWidget = self.parent()
        selection = table.selectedItems()
        if len(selection) == 1:
            table.takeItem(selection[0].row(),selection[0].column())
        # print(f"编辑内容:{content}")