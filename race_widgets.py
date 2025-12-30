import numpy as np
from PyQt6.QtGui import QPainter, QPen, QColor, QFont, QBrush, QPainterPath, QTransform
from PyQt6.QtCore import Qt, QPointF, QRectF
import qtawesome as qta 

def get_speed_color(speed):
    val = np.clip(speed / 200.0, 0, 1)
    hue = int((1.0 - val) * 240) 
    return QColor.fromHsv(hue, 220, 255)

class RaceWidgets:
    def __init__(self):
        self.font_big = QFont("Arial", 40, QFont.Weight.Bold)
        self.font_small = QFont("Consolas", 14, QFont.Weight.Bold)
        try:
            self.icon_sat = qta.icon('fa5s.satellite', color='white').pixmap(32, 32)
        except: self.icon_sat = None

    # === 1. 地图 (解决卡顿版) ===
    def draw_map(self, painter, state, dm, x, y, size):
        painter.save()
        
        # 裁剪区域
        path = QPainterPath()
        path.addRect(QRectF(x, y, size, size))
        painter.setClipPath(path)
        
        cx, cy = x + size/2, y + size/2
        
        # 动态地图变换逻辑 (QTransform 矩阵变换，保证丝滑)
        zoom = 4.0 
        
        transform = QTransform()
        transform.translate(cx, cy)            # 移到组件中心
        transform.rotate(-state['heading'] + 90) # 旋转
        transform.scale(zoom, zoom)            # 缩放
        transform.scale(1, -1)                 # 翻转Y
        transform.translate(-state['mx'], -state['my']) # 移到车的位置
        
        painter.setTransform(transform, combine=True)
        
        # 绘制轨迹
        points = [QPointF(mx, my) for mx, my in zip(dm.meter_x, dm.meter_y)]
        
        # 性能优化：简单的点抽稀或全部绘制 (Qt处理几千个点没问题)
        painter.setPen(QPen(QColor(255, 255, 255, 120), 10, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawPolyline(points)

        # 绘制高亮轨迹 (基于速度)
        step = 5
        visible_dist = size / zoom * 1.5
        
        for i in range(0, len(points)-step, step):
            # 简单的视锥剔除
            if abs(points[i].x() - state['mx']) > visible_dist: continue
            if abs(points[i].y() - state['my']) > visible_dist: continue
            
            spd = dm.cache.get('Speed_kmh', np.zeros(len(points)))[i]
            painter.setPen(QPen(get_speed_color(spd), 10, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(points[i], points[i+step])
            
        # 绘制车 (在原点)
        painter.setPen(QPen(Qt.GlobalColor.black, 2))
        painter.setBrush(Qt.GlobalColor.white)
        painter.drawEllipse(QPointF(state['mx'], state['my']), 8/zoom, 8/zoom)

        painter.restore()

    # === 2. 速度表 ===
    def draw_speed(self, painter, state, x, y, size):
        painter.save()
        painter.translate(x + size/2, y + size/2)
        scale = size / 200.0
        painter.scale(scale, scale)
        
        radius = 80
        speed = state['speed']
        
        painter.setPen(QPen(QColor(0,0,0,150), 15))
        painter.drawEllipse(QPointF(0,0), radius, radius)
        
        ratio = np.clip(speed / 200.0, 0, 1)
        angle = int(-270 * ratio * 16)
        painter.setPen(QPen(get_speed_color(speed), 15, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(QRectF(-radius, -radius, radius*2, radius*2), 225*16, angle)
        
        painter.setFont(self.font_big); painter.setPen(Qt.GlobalColor.white)
        painter.drawText(QRectF(-radius, -40, radius*2, 80), Qt.AlignmentFlag.AlignCenter, f"{int(speed)}")
        painter.setFont(self.font_small); painter.setPen(QColor(200,200,200))
        painter.drawText(QRectF(-radius, 40, radius*2, 30), Qt.AlignmentFlag.AlignCenter, "KM/H")
        
        painter.restore()

    # === 3. G值球 ===
    def draw_g_force(self, painter, state, x, y, size):
        painter.save()
        painter.translate(x + size/2, y + size/2)
        scale = size / 150.0 
        painter.scale(scale, scale)
        
        radius = 60
        max_g = 1.5 
        
        painter.setBrush(QColor(0,0,0,180)); painter.setPen(QPen(Qt.GlobalColor.white, 2))
        painter.drawEllipse(QPointF(0,0), radius, radius)
        
        painter.setBrush(Qt.BrushStyle.NoBrush); painter.setPen(QPen(QColor(255,255,255,80), 1, Qt.PenStyle.DashLine))
        painter.drawEllipse(QPointF(0,0), radius*0.66, radius*0.66)
        
        # 修正拼写
        gx = (state['lat_g'] / max_g) * radius
        gy = -(state['lon_g'] / max_g) * radius 
        
        dist = np.sqrt(gx**2 + gy**2)
        if dist > radius: gx = gx/dist*radius; gy = gy/dist*radius
        
        painter.setBrush(QColor(255, 50, 50)); painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(gx, gy), 8, 8)
        
        painter.setPen(Qt.GlobalColor.white); painter.setFont(QFont("Arial", 8))
        painter.drawText(QRectF(-radius, radius+5, radius*2, 20), Qt.AlignmentFlag.AlignCenter, f"{state['lat_g']:.1f}G / {state['lon_g']:.1f}G")
        
        painter.restore()

    # === 4. 姿态仪 ===
    def draw_attitude(self, painter, state, x, y, size):
        painter.save()
        painter.translate(x + size/2, y + size/2)
        scale = size / 150.0
        painter.scale(scale, scale)
        
        radius = 60
        path = QPainterPath(); path.addEllipse(QPointF(0,0), radius, radius); painter.setClipPath(path)
        
        roll = state['roll']
        pitch = state['pitch']
        
        painter.rotate(-roll)
        offset = pitch * 3
        
        painter.fillRect(QRectF(-100, -100+offset, 200, 100), QColor(0, 150, 255)) 
        painter.fillRect(QRectF(-100, 0+offset, 200, 100), QColor(150, 100, 50))  
        painter.setPen(QPen(Qt.GlobalColor.white, 2)); painter.drawLine(-100, int(offset), 100, int(offset))
        
        painter.rotate(roll)
        painter.setPen(QPen(Qt.GlobalColor.yellow, 3))
        painter.drawLine(-20, 0, -10, 0); painter.drawLine(10, 0, 20, 0); painter.drawPoint(0,0)
        
        painter.restore()
        
    # === 5. 信息栏 ===
    def draw_info(self, painter, state, x, y):
        painter.save()
        painter.translate(x, y)
        
        painter.setFont(self.font_small); painter.setPen(Qt.GlobalColor.white)
        painter.setBrush(QColor(0,0,0,150)); painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(0, 0, 160, 80)
        
        painter.setPen(Qt.GlobalColor.white)
        current_y = 25
        if self.icon_sat: 
            painter.drawPixmap(5, 5, 20, 20, self.icon_sat)
            painter.drawText(30, current_y, f"SAT: {state['sats']:.0f}")
        else:
            painter.drawText(10, current_y, f"SAT: {state['sats']:.0f}")
            
        current_y += 30
        painter.drawText(10, current_y, f"ALT: {state['alt']:.0f} m")
        
        painter.restore()