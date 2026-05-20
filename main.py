import sys

from ui.mainwindow import Window
from qfluentwidgets_pro import (
    setTheme,
    setThemeColor,
    Theme,
    FluentTranslator,
    QColor,
)
from PySide6.QtWidgets import QApplication

app = QApplication([])
translator = FluentTranslator()
app.installTranslator(translator)
setTheme(Theme.DARK)
setThemeColor(QColor.fromRgb(200, 162, 200))
w = Window()
# w.show()
sys.exit(app.exec())
