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
        
    def clear(self):
        """清空面板，移除显示的内容"""
        self.setPixmap(QtGui.QPixmap())  # 设置空图片
        self.update()

class ScoreChartWidget(QtWidgets.QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMinimumHeight(150)
        self.setStyleSheet("background: #222; border-radius: 6px;")
        self.scores = []  # 实时评分历史
        self.dtw_scores = []  # DTW评分历史
        self.max_points = 300  # Approx 10 seconds at 30fps
        
    def add_score(self, score):
        """兼容旧接口，只添加实时评分"""
        self.scores.append(score)
        self.dtw_scores.append(score)  # DTW评分相同
        self._trim_data()
        self.update()
    
    def add_scores(self, real_time_score, dtw_score):
        """同时添加两个评分"""
        self.scores.append(real_time_score)
        self.dtw_scores.append(dtw_score)
        self._trim_data()
        self.update()
    
    def _trim_data(self):
        """保持数据长度在限制内"""
        if len(self.scores) > self.max_points:
            self.scores = self.scores[-self.max_points:]
        if len(self.dtw_scores) > self.max_points:
            self.dtw_scores = self.dtw_scores[-self.max_points:]
        
    def reset(self):
        self.scores = []
        self.dtw_scores = []
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

        # 绘制两条线：实时评分（绿色）和DTW评分（蓝色）
        self._draw_line(painter, w, h, val_to_y, self.scores, 
                       QtGui.QColor("#10B981"), QtGui.QColor("#059669"),  # 绿色系
                       draw_fill=True, label="Real-time")
        
        if len(self.dtw_scores) >= 2:
            self._draw_line(painter, w, h, val_to_y, self.dtw_scores,
                           QtGui.QColor("#60A5FA"), QtGui.QColor("#3B82F6"),  # 蓝色系
                           draw_fill=False, label="DTW")
    
    def _draw_line(self, painter, w, h, val_to_y, scores, color_start, color_end, draw_fill=True, label=""):
        """绘制单条折线"""
        if len(scores) < 2:
            return
        
        n = len(scores)
        step_x = w / max(n - 1, 1)
        
        # 绘制折线路径
        path = QtGui.QPainterPath()
        start_y = val_to_y(scores[0])
        path.moveTo(0, start_y)
        
        for i in range(1, n):
            x = i * step_x
            y = val_to_y(scores[i])
            path.lineTo(x, y)
        
        # 线条样式
        pen_line = QtGui.QPen(color_start, 3)
        pen_line.setJoinStyle(QtCore.Qt.RoundJoin)
        pen_line.setCapStyle(QtCore.Qt.RoundCap)
        painter.setPen(pen_line)
        painter.drawPath(path)
        
        # 填充区域（仅实时评分）
        if draw_fill:
            fill_path = QtGui.QPainterPath(path)
            fill_path.lineTo(w, h)
            fill_path.lineTo(0, h)
            fill_path.closeSubpath()
            
            grad_fill = QtGui.QLinearGradient(0, 0, 0, h)
            grad_fill.setColorAt(0, QtGui.QColor(color_start.red(), color_start.green(), color_start.blue(), 60))
            grad_fill.setColorAt(1, QtGui.QColor(color_start.red(), color_start.green(), color_start.blue(), 5))
            painter.fillPath(fill_path, grad_fill)
        
        # 标签（显示在折线末端）
        if label:
            last_x = (n - 1) * step_x
            last_y = val_to_y(scores[-1])
            painter.setPen(color_start)
            font = painter.font()
            font.setPointSize(10)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(int(last_x) + 10, int(last_y), label)
