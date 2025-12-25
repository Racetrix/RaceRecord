import math
import numpy as np
from PyQt6.QtGui import QPainter, QColor, QFont, QPen, QBrush, QPainterPath, QPolygonF, QLinearGradient
from PyQt6.QtCore import Qt, QPointF, QRectF

# 定义样式常量
STYLE_DIGITAL = 0
STYLE_NEEDLE = 1
STYLE_LINEAR  = 2  # 新增 RS 风格

GAUGE_NAMES = {
    STYLE_DIGITAL: "🔮 科技圆环 (Digital)",
    STYLE_NEEDLE:  "🏎️ 物理指针 (Needle)",
    STYLE_LINEAR:  "📏 RS 线性风格 (Audi RS)"
}

def get_speed_color(speed, g_min, g_max):
    # 简单的 HSV 颜色辅助函数 (从 race_render 搬过来的)
    if g_max <= g_min: g_max = g_min + 1
    val = np.clip((speed - g_min) / (g_max - g_min), 0, 1)
    hue = int((1.0 - val) * 240)
    return QColor.fromHsv(hue, 255, 255)

class BaseGauge:
    def __init__(self):
        self.font_val = QFont("Arial", 60, QFont.Weight.Black)
        self.font_unit = QFont("Arial", 18, QFont.Weight.Bold)
        self.font_ticks = QFont("Arial", 14, QFont.Weight.Bold)

    def render(self, painter, x, y, speed, max_speed, scale, tick_width_scale):
        raise NotImplementedError

# === 样式 0: 科技圆环 (原版) ===
class DigitalGauge(BaseGauge):
    def render(self, painter, x, y, speed, max_speed, scale, tick_width_scale):
        painter.save()
        painter.translate(x, y)
        painter.scale(scale, scale)
        
        radius = 200
        # 底色环
        painter.setPen(QPen(QColor(40, 40, 40, 200), 25, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(QRectF(-radius, -radius, radius*2, radius*2), 225*16, -270*16)
        
        # 进度环
        ratio = min(speed / max_speed, 1.0)
        c = get_speed_color(speed, 0, max_speed) # 这里的 g_min 简写为 0
        painter.setPen(QPen(c, 25, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(QRectF(-radius, -radius, radius*2, radius*2), 225*16, int(-270*ratio*16))
        
        # 文字
        self.font_val.setPixelSize(120)
        painter.setFont(self.font_val); painter.setPen(Qt.GlobalColor.white)
        painter.drawText(QRectF(-radius, -80, radius*2, 120), Qt.AlignmentFlag.AlignCenter, f"{int(speed)}")
        
        self.font_unit.setPixelSize(30)
        painter.setFont(self.font_unit); painter.setPen(QColor(180, 180, 180))
        painter.drawText(QRectF(-radius, 60, radius*2, 40), Qt.AlignmentFlag.AlignCenter, "KM/H")
        
        painter.restore()

# === 样式 1: 物理指针 (原版) ===
class NeedleGauge(BaseGauge):
    def render(self, painter, x, y, speed, max_speed, scale, tick_width_scale):
        painter.save()
        painter.translate(x, y)
        painter.scale(scale, scale)
        
        radius = 200
        painter.setFont(self.font_ticks)
        
        # 计算步长 (略微简化逻辑)
        main_step = 20
        if max_speed <= 120: main_step = 10
        if max_speed > 260: main_step = 30
        
        sub_step = main_step / 5.0
        
        # 1. 绘制红区
        red_start = max_speed * 0.8
        start_a = 225 - (red_start / max_speed) * 270
        span_a = -270 * 0.2
        painter.setPen(QPen(QColor(220, 0, 0, 80), 12, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
        r_red = radius - 15
        painter.drawArc(QRectF(-r_red, -r_red, r_red*2, r_red*2), int(start_a*16), int(span_a*16))
        
        # 2. 绘制刻度
        curr = 0
        while curr <= max_speed + 0.1:
            angle_rad = math.radians(225 - (curr / max_speed) * 270)
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
            
            is_label = abs(curr % main_step) < 0.1
            
            if is_label:
                tick_len = 28
                painter.setPen(QPen(Qt.GlobalColor.white, 3.5 * tick_width_scale))
                # 绘制数字
                tx = cos_a * (radius - 65)
                ty = -sin_a * (radius - 65)
                painter.drawText(QRectF(tx-35, ty-15, 70, 30), Qt.AlignmentFlag.AlignCenter, str(int(curr)))
            else:
                tick_len = 12
                painter.setPen(QPen(QColor(180, 180, 180), 1.5 * tick_width_scale))
            
            p1 = QPointF(cos_a*radius, -sin_a*radius)
            p2 = QPointF(cos_a*(radius-tick_len), -sin_a*(radius-tick_len))
            painter.drawLine(p2, p1)
            curr += sub_step

        # 3. 绘制指针
        painter.save()
        curr_angle = 225 - (min(speed, max_speed) / max_speed) * 270
        painter.rotate(-curr_angle + 90)
        c = get_speed_color(speed, 0, max_speed)
        painter.setBrush(c); painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPolygon(QPolygonF([QPointF(-9, 0), QPointF(0, -radius+5), QPointF(9, 0)]))
        painter.restore()
        
        # 4. 中心圆 & 数字
        painter.setBrush(QColor(30, 30, 30)); painter.drawEllipse(QPointF(0,0), 20, 20)
        
        self.font_val.setPixelSize(70)
        painter.setFont(self.font_val); painter.setPen(Qt.GlobalColor.white)
        painter.drawText(QRectF(-150, 60, 300, 80), Qt.AlignmentFlag.AlignCenter, f"{int(speed)}")
        
        self.font_unit.setPixelSize(20)
        painter.setFont(self.font_unit); painter.setPen(QColor(180, 180, 180))
        painter.drawText(QRectF(-50, 130, 100, 30), Qt.AlignmentFlag.AlignCenter, "KM/H")

        painter.restore()

# === 样式 2: RS 线性风格 (New!) ===
class LinearGauge(BaseGauge):
    def render(self, painter, x, y, speed, max_speed, scale, tick_width_scale):
        """
        仿奥迪 RS 风格:
        左侧大数字 + 上方折线进度条
        """
        painter.save()
        painter.translate(x, y)
        painter.scale(scale, scale)
        
        # 布局尺寸
        w = 500
        h = 250
        # 整体居中偏移
        painter.translate(-w/3, 0) 

        # 1. 绘制大数字 (速度)
        self.font_val.setFamily("Eurostile") # 如果没有会回退到 Arial
        self.font_val.setPixelSize(140)
        self.font_val.setItalic(True)
        painter.setFont(self.font_val)
        
        # 速度颜色随快慢变化
        c_speed = get_speed_color(speed, 0, max_speed)
        painter.setPen(Qt.GlobalColor.white) # 数字保持白色更易读，或者用 c_speed
        
        # 绘制主数字
        str_speed = f"{int(speed)}"
        metrics = painter.fontMetrics()
        # 数字对齐中心
        painter.drawText(QRectF(0, 0, 300, 150), Qt.AlignmentFlag.AlignCenter, str_speed)
        
        # 绘制 "km/h"
        self.font_unit.setPixelSize(24)
        self.font_unit.setItalic(True)
        painter.setFont(self.font_unit)
        painter.setPen(QColor(180, 180, 180))
        painter.drawText(QRectF(0, 110, 300, 40), Qt.AlignmentFlag.AlignCenter, "KM/H")

        # 2. 绘制线性进度条 (折线形状)
        # 路径形状： /----------\
        # 坐标定义 (相对于数字上方)
        bar_x = -50
        bar_y = -20
        bar_w = 400
        bar_h = 20 # 条的粗细
        
        # 定义折线路径点
        # p1(左下) -> p2(左上拐点) -> p3(右上拐点) -> p4(右下)
        p1 = QPointF(bar_x, bar_y + 60)
        p2 = QPointF(bar_x + 60, bar_y)
        p3 = QPointF(bar_x + bar_w, bar_y)
        
        # 总路径长度 (近似计算用于进度)
        len_seg1 = 85 # 斜线长度
        len_seg2 = bar_w - 60 # 直线长度
        total_len = len_seg1 + len_seg2
        
        # 绘制底槽 (深灰色)
        path_bg = QPainterPath()
        path_bg.moveTo(p1)
        path_bg.lineTo(p2)
        path_bg.lineTo(p3)
        
        pen_bg = QPen(QColor(40, 40, 40), bar_h, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap)
        # 斜接连接点
        pen_bg.setJoinStyle(Qt.PenJoinStyle.MiterJoin) 
        painter.setPen(pen_bg)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path_bg)
        
        # 3. 绘制进度 (彩色)
        ratio = min(speed / max_speed, 1.0)
        current_len = total_len * ratio
        
        path_progress = QPainterPath()
        path_progress.moveTo(p1)
        
        if current_len <= len_seg1:
            # 还在斜坡阶段
            # 简单的线性插值计算终点
            t = current_len / len_seg1
            curr_x = p1.x() + (p2.x() - p1.x()) * t
            curr_y = p1.y() + (p2.y() - p1.y()) * t
            path_progress.lineTo(QPointF(curr_x, curr_y))
        else:
            # 已经过了拐点，在直线上
            path_progress.lineTo(p2)
            remain = current_len - len_seg1
            curr_x = p2.x() + remain
            path_progress.lineTo(QPointF(curr_x, p2.y()))
            
        # 进度条颜色：使用渐变
        grad = QLinearGradient(p1, p3)
        grad.setColorAt(0.0, QColor(0, 200, 255)) # 蓝
        grad.setColorAt(1.0, QColor(255, 0, 50))  # 红
        
        pen_prog = QPen(QBrush(grad), bar_h, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap)
        pen_prog.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        painter.setPen(pen_prog)
        painter.drawPath(path_progress)
        
        # 4. 绘制刻度数字 (可选，RS风格通常只标红区或者极值)
        # 这里我们在末尾标一个 Max Speed
        painter.setFont(QFont("Arial", 16, QFont.Weight.Bold, True))
        painter.setPen(QColor(150, 150, 150))
        painter.drawText(int(p3.x() + 10), int(p3.y() + 15), f"{int(max_speed)}")

        painter.restore()