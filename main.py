import sys

from ui.mainwindow import Window
from qfluentwidgets_pro import setTheme,Theme

from PySide6.QtWidgets import QApplication


app = QApplication([])
setTheme(Theme.DARK)
w = Window()
w.show()
sys.exit(app.exec())