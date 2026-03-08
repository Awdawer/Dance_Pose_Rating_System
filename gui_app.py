import sys
import os
from PyQt5 import QtWidgets, QtCore
import PyQt5 as _PyQt5

# Ensure Qt platform plugin path is discoverable on Windows
_qt_plugins_root = os.path.join(os.path.dirname(_PyQt5.__file__), "Qt5", "plugins")
_qt_platforms = os.path.join(_qt_plugins_root, "platforms")
if os.name == "nt" and os.path.isdir(_qt_platforms):
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", _qt_platforms)

# Add src to python path if needed (though usually current dir is in path)
sys.path.append(os.path.dirname(__file__))

from src.ui.main_window import MainWindow

def main():
    # Also add library path at runtime for safety
    QtCore.QCoreApplication.addLibraryPath(_qt_plugins_root)
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
