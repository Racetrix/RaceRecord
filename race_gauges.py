import math
import numpy as np
from PyQt6.QtGui import QPainter, QColor, QFont, QPen, QBrush, QPainterPath, QPolygonF, QLinearGradient
from PyQt6.QtCore import Qt, QPointF, QRectF

# 定义样式常量
STYLE_DIGITAL = 0
STYLE_NEEDLE = 1
STYLE_LINEAR  = 2 

GAUGE_NAMES = {
    STYLE_DIGITAL: "🔮 科技圆环 (Digital)",
    STYLE_NEEDLE:  "🏎️ 物理指针 (Needle)",
    STYLE_LINEAR:  "📏 RS 线性风格 (Audi RS)"
}

# 默认参数配置库
DEFAULT_CONFIGS = {
    STYLE_DIGITAL: {
        'scale': 1.0, 'x': 0, 'y': 0, 
        'ring_width': 25.0  # 圆环粗细
    },
    STYLE_NEEDLE: {
        'scale': 1.0, 'x': 0, 'y': 0, 
        'tick_width': 2.0,    # 刻度粗细
        'pointer_width': 1.0  # 指针粗细 (新增)
    },
    STYLE_LINEAR: {
        'scale': 1.0, 'x': 0, 'y': 0,
        'bar_height': 20.0,   # 进度条高度
        'tick_density': 10    # 刻度密度
    }
}

def get_speed_color(speed, g_min, g_max):
    if g_max <= g_min: g_max = g_min + 1
    val = np.clip((speed - g_min) / (g_max - g_min), 0, 1)
    hue = int((1.0 - val) * 240)
    return QColor.fromHsv(hue, 255, 255)

class BaseGauge:
    def __init__(self):
        self.font_val = QFont("Arial", 60, QFont.Weight.Black)
        self.font_unit = QFont("Arial", 18, QFont.Weight.Bold)
        self.font_ticks = QFont("Arial", 14, QFont.Weight.Bold)

    # 接口更新：接收 config 字典
    def render(self, painter, x, y, speed, max_speed, config):
        raise NotImplementedError

# === 样式 0: 科技圆环 ===
class DigitalGauge(BaseGauge):
    def render(self, painter, x, y, speed, max_speed, config):
        scale = config.get('scale', 1.0)
        ring_w = config.get('ring_width', 25.0)
        
        painter.save()
        painter.translate(x, y)
        painter.scale(scale, scale)
        
        radius = 200
        # 底色
        painter.setPen(QPen(QColor(40, 40, 40, 200), ring_w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(QRectF(-radius, -radius, radius*2, radius*2), 225*16, -270*16)
        
        # 进度
        ratio = min(speed / max_speed, 1.0)
        c = get_speed_color(speed, 0, max_speed)
        painter.setPen(QPen(c, ring_w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(QRectF(-radius, -radius, radius*2, radius*2), 225*16, int(-270*ratio*16))
        
        # 文字
        self.font_val.setPixelSize(120)
        painter.setFont(self.font_val); painter.setPen(Qt.GlobalColor.white)
        painter.drawText(QRectF(-radius, -80, radius*2, 120), Qt.AlignmentFlag.AlignCenter, f"{int(speed)}")
        
        self.font_unit.setPixelSize(30)
        painter.setFont(self.font_unit); painter.setPen(QColor(180, 180, 180))
        painter.drawText(QRectF(-radius, 60, radius*2, 40), Qt.AlignmentFlag.AlignCenter, "KM/H")
        
        painter.restore()

# === 样式 1: 物理指针 ===
class NeedleGauge(BaseGauge):
    def render(self, painter, x, y, speed, max_speed, config):
        scale = config.get('scale', 1.0)
        t_width = config.get('tick_width', 2.0)
        p_width = config.get('pointer_width', 1.0) # 🔥 找回了指针粗细
        
        painter.save()
        painter.translate(x, y)
        painter.scale(scale, scale)
        
        radius = 200
        painter.setFont(self.font_ticks)
        
        main_step = 20
        if max_speed <= 120: main_step = 10
        if max_speed > 260: main_step = 30
        sub_step = main_step / 5.0
        
        # 红区
        red_start = max_speed * 0.8
        start_a = 225 - (red_start / max_speed) * 270
        span_a = -270 * 0.2
        painter.setPen(QPen(QColor(220, 0, 0, 80), 12, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
        r_red = radius - 15
        painter.drawArc(QRectF(-r_red, -r_red, r_red*2, r_red*2), int(start_a*16), int(span_a*16))
        
        # 刻度
        curr = 0
        while curr <= max_speed + 0.1:
            angle_rad = math.radians(225 - (curr / max_speed) * 270)
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
            is_label = abs(curr % main_step) < 0.1
            
            if is_label:
                tick_len = 28
                painter.setPen(QPen(Qt.GlobalColor.white, 3.5 * t_width))
                tx = cos_a * (radius - 65); ty = -sin_a * (radius - 65)
                painter.drawText(QRectF(tx-35, ty-15, 70, 30), Qt.AlignmentFlag.AlignCenter, str(int(curr)))
            else:
                tick_len = 12
                painter.setPen(QPen(QColor(180, 180, 180), 1.5 * t_width))
            
            p1 = QPointF(cos_a*radius, -sin_a*radius)
            p2 = QPointF(cos_a*(radius-tick_len), -sin_a*(radius-tick_len))
            painter.drawLine(p2, p1)
            curr += sub_step

        # 指针 (应用 p_width)
        painter.save()
        curr_angle = 225 - (min(speed, max_speed) / max_speed) * 270
        painter.rotate(-curr_angle + 90)
        c = get_speed_color(speed, 0, max_speed)
        painter.setBrush(c); painter.setPen(Qt.PenStyle.NoPen)
        
        # 根据 p_width 调整指针胖瘦
        w_base = 9 * p_width
        painter.drawPolygon(QPolygonF([QPointF(-w_base, 0), QPointF(0, -radius+5), QPointF(w_base, 0)]))
        painter.restore()
        
        # 中心圆
        painter.setBrush(QColor(30, 30, 30)); painter.drawEllipse(QPointF(0,0), 20, 20)
        
        self.font_val.setPixelSize(70)
        painter.setFont(self.font_val); painter.setPen(Qt.GlobalColor.white)
        painter.drawText(QRectF(-150, 60, 300, 80), Qt.AlignmentFlag.AlignCenter, f"{int(speed)}")
        
        self.font_unit.setPixelSize(20)
        painter.setFont(self.font_unit); painter.setPen(QColor(180, 180, 180))
        painter.drawText(QRectF(-50, 130, 100, 30), Qt.AlignmentFlag.AlignCenter, "KM/H")

        painter.restore()

# === 样式 2: RS 线性风格 (升级版) ===
class LinearGauge(BaseGauge):
    def render(self, painter, x, y, speed, max_speed, config):
        scale = config.get('scale', 1.0)
        bar_h = config.get('bar_height', 20.0)
        tick_density = int(config.get('tick_density', 10)) # 分成几大格
        
        painter.save()
        painter.translate(x, y)
        painter.scale(scale, scale)
        painter.translate(-160, 0) # 视觉修正

        # 1. 文字
        self.font_val.setFamily("Eurostile") 
        self.font_val.setPixelSize(140)
        self.font_val.setItalic(True)
        painter.setFont(self.font_val)
        painter.setPen(Qt.GlobalColor.white)
        painter.drawText(QRectF(0, 0, 300, 150), Qt.AlignmentFlag.AlignCenter, f"{int(speed)}")
        
        self.font_unit.setPixelSize(24)
        self.font_unit.setItalic(True)
        painter.setFont(self.font_unit)
        painter.setPen(QColor(180, 180, 180))
        painter.drawText(QRectF(0, 110, 300, 40), Qt.AlignmentFlag.AlignCenter, "KM/H")

        # 2. 定义路径
        bar_x, bar_y, bar_w = -50, -20, 400
        p1 = QPointF(bar_x, bar_y + 60)
        p2 = QPointF(bar_x + 60, bar_y)
        p3 = QPointF(bar_x + bar_w, bar_y)
        
        len_seg1 = math.sqrt((p2.x()-p1.x())**2 + (p2.y()-p1.y())**2) # 斜线长
        len_seg2 = p3.x() - p2.x() # 直线长
        total_len = len_seg1 + len_seg2
        
        # 3. 绘制底槽
        path_bg = QPainterPath(); path_bg.moveTo(p1); path_bg.lineTo(p2); path_bg.lineTo(p3)
        pen_bg = QPen(QColor(40, 40, 40), bar_h, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap)
        pen_bg.setJoinStyle(Qt.PenJoinStyle.MiterJoin) 
        painter.setPen(pen_bg); painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path_bg)
        
        # 4. 🔥 绘制刻度 (Ticks)
        # 我们沿着路径走，每隔一段距离画一条垂直于路径的短线
        if tick_density < 2: tick_density = 2
        step_len = total_len / tick_density
        
        painter.setPen(QPen(QColor(20, 20, 20), 2)) # 刻度颜色(深黑，在进度条上做出镂空感)
        
        # 向量辅助函数
        def get_point_at(dist):
            if dist <= len_seg1:
                t = dist / len_seg1
                return QPointF(p1.x() + (p2.x()-p1.x())*t, p1.y() + (p2.y()-p1.y())*t), True # True=斜坡
            else:
                remain = dist - len_seg1
                return QPointF(p2.x() + remain, p2.y()), False # False=直线

        # 斜坡的法向量 (用于画刻度方向)
        dx1, dy1 = p2.x()-p1.x(), p2.y()-p1.y()
        norm1 = math.sqrt(dx1**2 + dy1**2)
        ux1, uy1 = -dy1/norm1, dx1/norm1 # 旋转90度
        
        # 直线的法向量
        ux2, uy2 = 0, -1
        
        tick_half_h = bar_h / 2 + 2 # 刻度长度略大于条宽
        
        for i in range(1, tick_density):
            d = i * step_len
            pt, is_slope = get_point_at(d)
            
            ux, uy = (ux1, uy1) if is_slope else (ux2, uy2)
            
            t_start = QPointF(pt.x() - ux*tick_half_h, pt.y() - uy*tick_half_h)
            t_end   = QPointF(pt.x() + ux*tick_half_h, pt.y() + uy*tick_half_h)
            painter.drawLine(t_start, t_end)

        # 5. 绘制进度
        ratio = min(speed / max_speed, 1.0)
        current_len = total_len * ratio
        path_progress = QPainterPath(); path_progress.moveTo(p1)
        
        if current_len > 0.1: # 只有大于0才画
            final_pt, _ = get_point_at(current_len)
            if current_len <= len_seg1:
                path_progress.lineTo(final_pt)
            else:
                path_progress.lineTo(p2)
                path_progress.lineTo(final_pt)
                
            grad = QLinearGradient(p1, p3)
            grad.setColorAt(0.0, QColor(0, 200, 255))
            grad.setColorAt(1.0, QColor(255, 0, 50))
            
            # 使用 CompositionMode 实现“遮罩刻度”效果太复杂，
            # 简单做法是先画进度条，但刻度会被盖住。
            # 为了让刻度显现，我们应该把刻度画在进度条 *上面*，或者用半透明。
            # 这里我采用：底槽 -> 进度 -> 重新画一遍刻度 (这次用白色/黑色混搭)
            
            pen_prog = QPen(QBrush(grad), bar_h, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap)
            pen_prog.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
            painter.setPen(pen_prog)
            painter.drawPath(path_progress)
            
            # 重绘刻度 (让刻度浮在进度条上)
            painter.setPen(QPen(QColor(0, 0, 0, 150), 3)) # 半透明黑线
            for i in range(1, tick_density):
                d = i * step_len
                if d > current_len: break # 还没到的进度不用重绘
                pt, is_slope = get_point_at(d)
                ux, uy = (ux1, uy1) if is_slope else (ux2, uy2)
                t_start = QPointF(pt.x() - ux*(bar_h/2), pt.y() - uy*(bar_h/2))
                t_end   = QPointF(pt.x() + ux*(bar_h/2), pt.y() + uy*(bar_h/2))
                painter.drawLine(t_start, t_end)

        painter.restore()