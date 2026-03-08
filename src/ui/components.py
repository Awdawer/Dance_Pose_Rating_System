from PyQt5 import QtWidgets, QtGui, QtCore

class VideoPanel(QtWidgets.QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.setScaledContents(True)
        self.setMinimumSize(480, 270)
        self.setStyleSheet("background: #111; border-radius: 6px;")

class ScoreChartWidget(QtWidgets.QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMinimumHeight(150)
        self.setStyleSheet("background: #222; border-radius: 6px;")
        self.scores = []
        self.max_points = 300 # Approx 10 seconds at 30fps, or scroll? 
        # Let's auto-scale X, but keep a reasonable buffer.
        
    def add_score(self, score):
        self.scores.append(score)
        self.update()
        
    def reset(self):
        self.scores = []
        self.update()
        
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        # Background
        painter.fillRect(0, 0, w, h, QtGui.QColor("#222"))
        
        # Grid lines (0, 50, 80, 100)
        padding = 20
        graph_h = h - 2 * padding
        
        def val_to_y(v):
            # v is 0..100
            return int(h - padding - (v / 100.0) * graph_h)
            
        pen_grid = QtGui.QPen(QtGui.QColor("#444"))
        pen_grid.setStyle(QtCore.Qt.DashLine)
        
        # Draw grid
        for val in [0, 50, 80, 100]:
            y = val_to_y(val)
            if val == 80:
                pen_grid.setColor(QtGui.QColor("#10B981")) # Green for pass line
            else:
                pen_grid.setColor(QtGui.QColor("#444"))
            painter.setPen(pen_grid)
            painter.drawLine(0, y, w, y)
            
            # Text label
            painter.setPen(QtGui.QColor("#888"))
            painter.drawText(5, y - 2, f"{val}%")

        if len(self.scores) < 2:
            return

        # Draw Line
        path = QtGui.QPainterPath()
        
        # Determine X scale
        # If points < w, 1 pixel per point? Or stretch to fill?
        # Let's stretch to fill, it looks better for "Session Progress"
        n = len(self.scores)
        step_x = w / max(n - 1, 1)
        
        start_y = val_to_y(self.scores[0])
        path.moveTo(0, start_y)
        
        for i in range(1, n):
            x = i * step_x
            y = val_to_y(self.scores[i])
            path.lineTo(x, y)
            
        # Stroke
        pen_line = QtGui.QPen(QtGui.QColor("#3B82F6")) # Blue
        pen_line.setWidth(2)
        painter.setPen(pen_line)
        painter.drawPath(path)
        
        # Fill gradient area under line
        path.lineTo(w, h)
        path.lineTo(0, h)
        path.closeSubpath()
        
        grad = QtGui.QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0, QtGui.QColor(59, 130, 246, 100)) # Blue semi-transparent
        grad.setColorAt(1, QtGui.QColor(59, 130, 246, 0))
        painter.fillPath(path, grad)
