import sys

from ui.mainwindow import Window
from qfluentwidgets_pro import (
    setTheme,
    Theme,
    FluentTranslator,
)

from PySide6.QtWidgets import QApplication

app = QApplication([])
translator = FluentTranslator()
app.installTranslator(translator)
setTheme(Theme.DARK)
w = Window()
w.show()
sys.exit(app.exec())
