import math
from PyQt6.QtGui import QPainter, QColor, QFont, QPen, QBrush, QPainterPath, QPolygonF, QLinearGradient, QRadialGradient
from PyQt6.QtCore import Qt, QPointF, QRectF

class TelemetryVisualizer:
    def __init__(self):
        self.font_label = QFont("Arial", 12, QFont.Weight.Bold)
        self.font_val = QFont("Consolas", 14, QFont.Weight.Bold)

    def render(self, painter, x, y, state, config):
        raise NotImplementedError

# === G-Force Ball (G值球) ===
class GBall(TelemetryVisualizer):
    def render(self, painter, x, y, state, config):
        scale = config.get('scale', 1.0)
        max_g = config.get('max_g', 1.5) # G值量程
        invert_lon = config.get('invert_lon', False)
        invert_lat = config.get('invert_lat', False)
        swap_axes = config.get('swap_axes', False)

        painter.save()
        painter.translate(x, y)
        painter.scale(scale, scale)
        
        radius = 80
        ball_radius = 12

        # 1. 获取 G 值
        raw_lon = state['lon_g'] * (-1 if invert_lon else 1)
        raw_lat = state['lat_g'] * (-1 if invert_lat else 1)

        if swap_axes:
            g_x_val = raw_lon
            g_y_val = raw_lat
        else:
            g_x_val = raw_lat
            g_y_val = raw_lon 

        # 2. 背景
        bg_grad = QRadialGradient(0, 0, radius)
        bg_grad.setColorAt(0.0, QColor(40, 40, 40, 150))
        bg_grad.setColorAt(1.0, QColor(20, 20, 20, 150))
        painter.setBrush(QBrush(bg_grad))
        painter.setPen(QPen(QColor(80, 80, 80), 2))
        painter.drawEllipse(QPointF(0,0), radius, radius)
        
        # 3. 准星 (使用整数坐标)
        painter.setPen(QPen(QColor(100, 100, 100, 100), 1))
        painter.drawLine(0, -radius, 0, radius)
        painter.drawLine(-radius, 0, radius, 0)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        
        ref_g = 1.0 if max_g > 1.0 else 0.5
        ref_radius = (ref_g / max_g) * radius
        painter.drawEllipse(QPointF(0,0), ref_radius, ref_radius)
        
        # 4. 小球位置
        pos_x = (g_x_val / max_g) * radius
        pos_y = -(g_y_val / max_g) * radius 

        dist = math.sqrt(pos_x**2 + pos_y**2)
        if dist + ball_radius > radius:
            ratio = (radius - ball_radius) / dist
            pos_x *= ratio
            pos_y *= ratio

        # 5. 绘制小球
        ball_grad = QRadialGradient(pos_x - 3, pos_y - 3, ball_radius)
        ball_grad.setColorAt(0.0, QColor(255, 50, 50))
        ball_grad.setColorAt(1.0, QColor(180, 0, 0))
        painter.setBrush(QBrush(ball_grad))
        painter.setPen(QPen(QColor(255, 100, 100), 1))
        painter.drawEllipse(QPointF(pos_x, pos_y), ball_radius, ball_radius)
        
        # 6. 数值
        painter.setFont(self.font_label)
        painter.setPen(QColor(180, 180, 180))
        painter.drawText(QRectF(-radius, -radius-25, radius*2, 20), Qt.AlignmentFlag.AlignCenter, "G-FORCE")
        
        total_g = math.sqrt(raw_lon**2 + raw_lat**2)
        painter.setFont(self.font_val)
        painter.setPen(Qt.GlobalColor.white)
        if total_g > max_g * 0.8:
            painter.setPen(QColor(255, 50, 50))
        painter.drawText(QRectF(-radius, radius+5, radius*2, 25), Qt.AlignmentFlag.AlignCenter, f"{total_g:.2f}G")

        painter.restore()

# === Attitude Indicator (姿态仪) ===
class AttitudeIndicator(TelemetryVisualizer):
    def render(self, painter, x, y, state, config):
        scale = config.get('scale', 1.0)
        max_pitch = config.get('max_pitch', 30.0)
        invert_roll = config.get('invert_roll', False)
        invert_pitch = config.get('invert_pitch', False)

        roll_deg = state['roll'] * (-1 if invert_roll else 1)
        pitch_deg = state['pitch'] * (-1 if invert_pitch else 1)
        heading = state['heading']

        size = 160 

        painter.save()
        painter.translate(x, y)
        painter.scale(scale, scale)

        # 裁剪
        path = QPainterPath()
        path.addEllipse(QPointF(0,0), size/2, size/2)
        painter.setClipPath(path)

        # --- 地平线 ---
        painter.save()
        painter.rotate(roll_deg)
        
        pixels_per_deg = (size / 2) / max_pitch
        pitch_offset = pitch_deg * pixels_per_deg
        painter.translate(0, pitch_offset)
        
        # 天空
        sky_grad = QLinearGradient(0, -size, 0, 0)
        sky_grad.setColorAt(0.0, QColor(0, 100, 200))
        sky_grad.setColorAt(1.0, QColor(100, 180, 255))
        painter.fillRect(QRectF(-size, -size*2, size*2, size*2), QBrush(sky_grad))
        
        # 地面
        gnd_grad = QLinearGradient(0, 0, 0, size)
        gnd_grad.setColorAt(0.0, QColor(120, 80, 40))
        gnd_grad.setColorAt(1.0, QColor(80, 50, 20))
        painter.fillRect(QRectF(-size, 0, size*2, size*2), QBrush(gnd_grad))
        
        # 地平线
        painter.setPen(QPen(Qt.GlobalColor.white, 2))
        painter.drawLine(-size, 0, size, 0)
        
        # 俯仰刻度
        painter.setPen(QPen(Qt.GlobalColor.white, 1))
        for p in range(10, int(max_pitch)+1, 10):
            y_sky = -p * pixels_per_deg
            # 🔥 修复：强制转为 int，防止 drawLine 报错
            painter.drawLine(-20, int(y_sky), 20, int(y_sky))
            
            y_gnd = p * pixels_per_deg
            painter.drawLine(-10, int(y_gnd), 10, int(y_gnd))

        painter.restore()

        # --- 静态元素 ---
        painter.setClipping(False) 
        painter.setPen(QPen(QColor(60, 60, 60), 4))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(0,0), size/2, size/2)
        
        # 飞机/车辆符号
        painter.setPen(QPen(QColor(255, 200, 0), 3))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawLine(-40, 0, -10, 0)
        painter.drawLine(10, 0, 40, 0)
        painter.drawEllipse(QPointF(0,0), 3, 3)
        painter.drawLine(0, 0, 0, 15)

        # --- 顶部航向标 ---
        painter.save()
        painter.setPen(QPen(Qt.GlobalColor.white, 1))
        painter.setFont(QFont("Arial", 8))
        
        header_h = 25
        painter.setBrush(QColor(20, 20, 20, 200))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(QRectF(-size/2, -size/2 - 10, size, header_h))

        visible_range = 30
        start_h = int(heading - visible_range)
        end_h = int(heading + visible_range)
        pixels_per_h_deg = (size * 0.8) / (visible_range * 2)

        painter.setPen(QPen(Qt.GlobalColor.white, 1))
        for h_val in range(start_h, end_h + 1):
            norm_h = h_val % 360
            angle_diff = h_val - heading
            x_pos = angle_diff * pixels_per_h_deg
            
            if abs(x_pos) < size/2 * 0.8:
                is_major = (norm_h % 15 == 0)
                tick_h = 8 if is_major else 5
                
                # 🔥 修复：强制转为 int
                ix = int(x_pos)
                iy_top = int(-size/2)
                painter.drawLine(ix, iy_top, ix, iy_top + tick_h)
                
                if is_major:
                    label = str(norm_h)
                    if norm_h == 0: label = "N"
                    elif norm_h == 90: label = "E"
                    elif norm_h == 180: label = "S"
                    elif norm_h == 270: label = "W"
                    painter.drawText(QRectF(x_pos-15, -size/2 - 20, 30, 20), Qt.AlignmentFlag.AlignCenter, label)
        
        # 红色指针
        painter.setPen(QPen(QColor(255, 200, 0), 2))
        # 🔥 修复：使用整数坐标
        top_y = int(-size/2)
        painter.drawLine(0, top_y + 10, 0, top_y + 20)
        painter.drawPolygon(QPolygonF([QPointF(0, top_y+10), QPointF(-4, top_y), QPointF(4, top_y)]))

        painter.restore()

        # 标题
        painter.setFont(self.font_label)
        painter.setPen(QColor(180, 180, 180))
        painter.drawText(QRectF(-size/2, size/2 + 5, size, 20), Qt.AlignmentFlag.AlignCenter, "ATTITUDE")
        
        painter.restore()