import torch,sys
import numpy as np
from pathlib import Path
from copy import deepcopy
from matplotlib import pyplot as plt
from ultralytics.utils.plotting import Annotator
from ultralytics.engine.results import Results,LetterBox,colors,Boxes
from PIL import Image
from PySide6.QtWidgets import (
    QWidget,
    QApplication,
    QStyleOptionViewItem,
    QAbstractItemView,
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

def get_resource_path(relative_path: str) -> str:
    if getattr(sys, 'frozen', False):
        base = Path(sys.executable).parent
    else:
        base = Path.cwd()
    return str(base / relative_path)
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

class drawCustomRes(Results):
    def __init__(self,res:Results):
        # 继承原有的res
        self.__dict__ = res.__dict__
    def plot(
        self,
        conf: bool = True,
        line_width: float | None = None,
        font_size: float | None = None,
        font: str = "Arial.ttf",
        pil: bool = False,
        img: np.ndarray | None = None,
        im_gpu: torch.Tensor | None = None,
        kpt_radius: int = 5,
        kpt_line: bool = True,
        labels: bool = True,
        boxes: bool = True,
        masks: bool = True,
        probs: bool = True,
        show: bool = False,
        save: bool = False,
        filename: str | None = None,
        color_mode: str = "class",
        txt_color: tuple[int, int, int] = (255, 255, 255),
        custom_boxs: list[Boxes]|None = None
    ) -> np.ndarray:
        """作图"""
        assert color_mode in {"instance", "class"}, f"Expected color_mode='instance' or 'class', not {color_mode}."
        if img is None and isinstance(self.orig_img, torch.Tensor):
            img = (self.orig_img[0].detach().permute(1, 2, 0).contiguous() * 255).byte().cpu().numpy()

        names = self.names
        is_obb = self.obb is not None
        pred_boxes, show_boxes = custom_boxs if isinstance(custom_boxs,list) else self.obb if is_obb else self.boxes, boxes
        pred_masks, show_masks = self.masks, masks
        pred_probs, show_probs = self.probs, probs
        annotator = Annotator(
            deepcopy(self.orig_img if img is None else img),
            line_width,
            font_size,
            font,
            pil or (pred_probs is not None and show_probs),  # Classify tasks default to pil=True
            example=names,
        )

        # Plot Segment results
        if pred_masks and show_masks:
            if im_gpu is None:
                img = LetterBox(pred_masks.shape[1:])(image=annotator.result())
                im_gpu = (
                    torch.as_tensor(img, dtype=torch.float16, device=pred_masks.data.device)
                    .permute(2, 0, 1)
                    .flip(0)
                    .contiguous()
                    / 255
                )
            idx = (
                pred_boxes.id
                if pred_boxes.is_track and color_mode == "instance"
                else pred_boxes.cls
                if pred_boxes and color_mode == "class"
                else reversed(range(len(pred_masks)))
            )
            annotator.masks(pred_masks.data, colors=[colors(x, True) for x in idx], im_gpu=im_gpu)

        # Plot Detect results
        if pred_boxes is not None and show_boxes:
            for i, d in enumerate(reversed(pred_boxes)):
                c, d_conf, id = int(d.cls), float(d.conf) if conf else None, int(d.id.item()) if d.is_track else None
                name = ("" if id is None else f"id:{id} ") + names[c]
                label = (f"{name} {d_conf:.2f}" if conf else name) if labels else (f"{d_conf:.2f}" if conf else None)
                box = d.xyxyxyxy.squeeze() if is_obb else d.xyxy.squeeze()
                annotator.box_label(
                    box,
                    label,
                    color=colors(
                        c
                        if color_mode == "class"
                        else id
                        if id is not None
                        else i
                        if color_mode == "instance"
                        else None,
                        True,
                    ),
                )

        # Plot Classify results
        if pred_probs is not None and show_probs:
            text = "\n".join(f"{names[j] if names else j} {pred_probs.data[j]:.2f}" for j in pred_probs.top5)
            x = round(self.orig_shape[0] * 0.03)
            annotator.text([x, x], text, txt_color=txt_color, box_color=(64, 64, 64, 128))  # RGBA box

        # Plot Pose results
        if self.keypoints is not None:
            for i, k in enumerate(reversed(self.keypoints.data)):
                annotator.kpts(
                    k,
                    self.orig_shape,
                    radius=kpt_radius,
                    kpt_line=kpt_line,
                    kpt_color=colors(i, True) if color_mode == "instance" else None,
                )

        # Show results
        if show:
            annotator.show(self.path)

        # Save results
        if save:
            annotator.save(filename or f"results_{Path(self.path).name}")

        return annotator.result(pil)