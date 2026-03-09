from PyQt5 import QtWidgets, QtGui, QtCore

class VideoPanel(QtWidgets.QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.setScaledContents(False) # Disable auto-stretch to keep aspect ratio
        self.setAlignment(QtCore.Qt.AlignCenter) # Center the image
        self.setMinimumSize(480, 270)
        self.setStyleSheet("background: #000; border-radius: 6px;") # Black background for letterbox
        
    def heightForWidth(self, width):
        return int(width * 9 / 16)
        
    def sizeHint(self):
        # Provide a hint that respects 16:9
        return QtCore.QSize(640, 360)

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
        
        # Background with subtle gradient
        bg_grad = QtGui.QLinearGradient(0, 0, 0, h)
        bg_grad.setColorAt(0, QtGui.QColor("#1A1A1A"))
        bg_grad.setColorAt(1, QtGui.QColor("#2A2A2A"))
        painter.fillRect(0, 0, w, h, bg_grad)
        
        # Grid lines (0, 50, 80, 100)
        padding = 30
        graph_h = h - 2 * padding
        
        def val_to_y(v):
            return int(h - padding - (v / 100.0) * graph_h)
            
        pen_grid = QtGui.QPen(QtGui.QColor("#333"))
        pen_grid.setStyle(QtCore.Qt.SolidLine)
        pen_grid.setWidth(1)
        
        # Draw grid
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)
        
        for val in [0, 50, 80, 100]:
            y = val_to_y(val)
            if val == 80:
                # Pass line - Neon Green
                pen_grid.setColor(QtGui.QColor("#00FF9D"))
                pen_grid.setStyle(QtCore.Qt.DashLine)
                pen_grid.setWidth(2)
            else:
                pen_grid.setColor(QtGui.QColor("#444"))
                pen_grid.setStyle(QtCore.Qt.SolidLine)
                pen_grid.setWidth(1)
                
            painter.setPen(pen_grid)
            painter.drawLine(0, y, w, y)
            
            # Text label
            painter.setPen(QtGui.QColor("#AAA"))
            painter.drawText(5, y - 5, f"{val}%")

        if len(self.scores) < 2:
            return

        # Draw Line
        path = QtGui.QPainterPath()
        
        n = len(self.scores)
        step_x = w / max(n - 1, 1)
        
        start_y = val_to_y(self.scores[0])
        path.moveTo(0, start_y)
        
        for i in range(1, n):
            x = i * step_x
            y = val_to_y(self.scores[i])
            # Bezier curve for smooth look? Or straight for accuracy?
            # Let's keep straight for performance but add glow
            path.lineTo(x, y)
            
        # Stroke - Neon Blue/Cyan Gradient
        grad_pen = QtGui.QLinearGradient(0, 0, w, 0)
        grad_pen.setColorAt(0, QtGui.QColor("#00C6FF")) # Cyan
        grad_pen.setColorAt(1, QtGui.QColor("#0072FF")) # Blue
        
        pen_line = QtGui.QPen(QtGui.QBrush(grad_pen), 3)
        pen_line.setJoinStyle(QtCore.Qt.RoundJoin)
        pen_line.setCapStyle(QtCore.Qt.RoundCap)
        painter.setPen(pen_line)
        painter.drawPath(path)
        
        # Fill gradient area under line
        fill_path = QtGui.QPainterPath(path)
        fill_path.lineTo(w, h)
        fill_path.lineTo(0, h)
        fill_path.closeSubpath()
        
        grad_fill = QtGui.QLinearGradient(0, 0, 0, h)
        grad_fill.setColorAt(0, QtGui.QColor(0, 198, 255, 80)) # Semi-transparent Cyan
        grad_fill.setColorAt(1, QtGui.QColor(0, 114, 255, 10))  # Fading out
        painter.fillPath(fill_path, grad_fill)
        
        # Highlight current point
        last_x = (n - 1) * step_x
        last_y = val_to_y(self.scores[-1])
        painter.setBrush(QtGui.QColor("#FFFFFF"))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawEllipse(QtCore.QPointF(last_x, last_y), 5, 5)
        # Glow for point
        painter.setBrush(QtGui.QColor(0, 198, 255, 100))
        painter.drawEllipse(QtCore.QPointF(last_x, last_y), 10, 10)
