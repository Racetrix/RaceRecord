import numpy as np
from PyQt6.QtGui import QPainter, QPen, QColor, QFont, QBrush, QPainterPath, QTransform, QPolygonF, QImage
from PyQt6.QtCore import Qt, QPointF, QRectF
import qtawesome as qta 

# ================= 常量定义 =================
RESOLUTION = (1920, 1080)
MODE_PATH = 0 
MODE_GAUGE = 1
STYLE_DIGITAL = 0 
STYLE_NEEDLE = 1 

# 地图视角
MAP_STATIC_NORTH = 0  # 北向 (静态全局)
MAP_DYNAMIC_HEAD = 1  # 车头向 (动态跟随)

COLOR_SPEED = 0 
COLOR_WHITE = 1  
COLOR_RED = 2    
COLOR_CYAN = 3   
# ==========================================

def qimage_to_numpy(qimg):
    qimg = qimg.convertToFormat(QImage.Format.Format_RGBA8888)
    width = qimg.width(); height = qimg.height()
    ptr = qimg.bits(); ptr.setsize(qimg.sizeInBytes())
    arr = np.frombuffer(ptr, np.uint8)
    bytes_per_line = qimg.bytesPerLine()
    arr = arr.reshape((height, bytes_per_line))
    arr = arr[:, :width * 4]
    arr = arr.reshape((height, width, 4))
    return np.ascontiguousarray(arr.copy())

def get_speed_color(speed, g_min, g_max):
    if g_max <= g_min: g_max = g_min + 1
    val = np.clip((speed - g_min) / (g_max - g_min), 0, 1)
    hue = int((1.0 - val) * 240) 
    return QColor.fromHsv(hue, 255, 255)

class Renderer:
    def __init__(self, data_manager):
        self.dm = data_manager 
        self.track_width = 15     
        self.car_size = 30        
        self.map_style = MAP_STATIC_NORTH
        self.gauge_style = STYLE_DIGITAL 
        self.max_speed = 200      
        self.show_gauge = True    
        self.show_extra = False    
        self.show_time = True
        self.show_sats = True
        self.show_alt = False     
        
        # 布局参数
        self.gauge_scale = 1.0      
        self.gauge_offset_x = 0     
        self.gauge_offset_y = 0     
        self.tick_width_scale = 1.0 
        
        self.path_color_mode = COLOR_SPEED
        self.grad_min = 0.0
        self.grad_max = 160.0
        
        # 动态缩放开关
        self.enable_dynamic_zoom = True

        self.font_big = QFont("Arial", 90, QFont.Weight.Bold)
        self.font_mid = QFont("Arial", 22, QFont.Weight.Bold)
        self.font_small = QFont("Consolas", 26, QFont.Weight.Bold)
        self.font_ticks = QFont("Arial", 14, QFont.Weight.Bold)
        self.font_needle_val = QFont("Arial", 55, QFont.Weight.Black) 
        
        try:
            self.icon_sat = qta.icon('fa5s.satellite', color='white').pixmap(64, 64)
            self.icon_alt = qta.icon('fa5s.mountain', color='white').pixmap(64, 64)
            self.icon_clock = qta.icon('fa5s.stopwatch', color='white').pixmap(64, 64)
        except:
            self.icon_sat = None 

    def render(self, painter, w, h, current_time, transparent_bg, render_mode):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        
        if transparent_bg:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
            painter.fillRect(0, 0, w, h, Qt.GlobalColor.transparent)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        else:
            painter.fillRect(0, 0, w, h, Qt.GlobalColor.black)

        if self.dm.df_proc is None: return
        state = self.dm.get_state_at_time(current_time)
        if not state: return

        if render_mode == MODE_PATH:
            if self.map_style == MAP_STATIC_NORTH:
                self.draw_static_map(painter, w, h, state, transparent_bg)
            else:
                self.draw_dynamic_map(painter, w, h, state, current_time, transparent_bg)
        elif render_mode == MODE_GAUGE:
            self.draw_gauge_manual(painter, w, h, state, current_time)

    def get_solid_color(self):
        if self.path_color_mode == COLOR_WHITE: return Qt.GlobalColor.white
        if self.path_color_mode == COLOR_RED: return QColor(255, 50, 50)
        if self.path_color_mode == COLOR_CYAN: return QColor(0, 255, 255)
        return Qt.GlobalColor.white

    def draw_path_lines(self, painter, points):
        if self.path_color_mode == COLOR_SPEED:
            step = 2
            for i in range(0, len(points)-step, step):
                spd = self.dm.cached_speeds[i]
                c = get_speed_color(spd, self.grad_min, self.grad_max)
                painter.setPen(QPen(c, self.track_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                painter.drawLine(points[i], points[i+step])
        else:
            c = self.get_solid_color()
            painter.setPen(QPen(c, self.track_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawPolyline(points)

    def draw_static_map(self, painter, w, h, s, transparent_bg):
        margin = 60
        avail_w, avail_h = w - margin*2, h - margin*2
        if avail_w * self.dm.aspect_ratio > avail_h:
            draw_h = avail_h; draw_w = avail_h / self.dm.aspect_ratio
        else:
            draw_w = avail_w; draw_h = avail_w * self.dm.aspect_ratio
        offset_x, offset_y = (w - draw_w) / 2, (h - draw_h) / 2
        def to_screen(nx, ny): return QPointF(offset_x + nx * draw_w, h - (offset_y + ny * draw_h))

        points = [to_screen(x, y) for x, y in zip(self.dm.norm_x, self.dm.norm_y)]
        
        if not transparent_bg and self.path_color_mode == COLOR_SPEED:
            painter.setPen(QPen(QColor(40, 40, 40), self.track_width + 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawPolyline(points)
        
        self.draw_path_lines(painter, points)
            
        car_pos = to_screen(s['nx'], s['ny'])
        painter.setPen(QPen(Qt.GlobalColor.black, 3))
        painter.setBrush(QBrush(Qt.GlobalColor.white))
        painter.drawEllipse(car_pos, self.car_size, self.car_size)

    def draw_dynamic_map(self, painter, w, h, s, current_time, transparent_bg):
        base_scale = 3.0 
        
        if self.enable_dynamic_zoom:
            speed_ratio = np.clip(s['speed'] / 200.0, 0, 1) 
            zoom = base_scale * (1.8 - np.sqrt(speed_ratio) * 1.2)
        else:
            speed_factor = max(0, s['speed']) / float(self.max_speed)
            zoom = base_scale * (1.0 - speed_factor * 0.6)

        # 旋转逻辑 (回到简单的车头向)
        rotation_angle = -s['heading'] + 90

        transform = QTransform()
        transform.translate(w/2, h/2)
        transform.rotate(rotation_angle)
        transform.scale(zoom, zoom)
        transform.scale(1, -1) 
        transform.translate(-s['mx'], -s['my'])
        
        painter.setTransform(transform)
        points = [QPointF(x, y) for x, y in zip(self.dm.meter_x, self.dm.meter_y)]
        
        if not transparent_bg and self.path_color_mode == COLOR_SPEED:
            painter.setPen(QPen(QColor(40, 40, 40), self.track_width/zoom + 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawPolyline(points)
            
        real_width = self.track_width / zoom
        
        if self.path_color_mode == COLOR_SPEED:
            step = 2
            for i in range(0, len(points)-step, step):
                spd = self.dm.cached_speeds[i]
                c = get_speed_color(spd, self.grad_min, self.grad_max)
                painter.setPen(QPen(c, real_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                painter.drawLine(points[i], points[i+step])
        else:
            c = self.get_solid_color()
            painter.setPen(QPen(c, real_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawPolyline(points)
            
        real_car_size = self.car_size / zoom
        painter.setPen(QPen(Qt.GlobalColor.black, 2/zoom))
        painter.setBrush(QBrush(Qt.GlobalColor.white))
        painter.drawEllipse(QPointF(s['mx'], s['my']), real_car_size, real_car_size)
            
        painter.resetTransform()

    def draw_gauge_manual(self, painter, w, h, s, t):
        base_cx = w / 2; base_cy = h / 2
        target_cx = base_cx + self.gauge_offset_x
        target_cy = base_cy + self.gauge_offset_y
        
        painter.save()
        painter.translate(target_cx, target_cy)
        painter.scale(self.gauge_scale, self.gauge_scale)
        
        if self.show_gauge:
            if self.gauge_style == STYLE_DIGITAL: self.draw_digital_gauge(painter, 0, 0, s['speed'])
            else: self.draw_needle_gauge(painter, 0, 0, s['speed'])
        painter.restore()

        if self.show_extra:
            start_x = w - 350
            start_y = h - 300
            self.draw_extra_info(painter, start_x, start_y, t, s['sats'], s['alt'])

    def draw_digital_gauge(self, painter, cx, cy, speed):
        radius = 200
        painter.setPen(QPen(QColor(40, 40, 40, 200), 25, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(QRectF(-radius, -radius, radius*2, radius*2), 225*16, -270*16)
        
        ratio = min(speed / self.max_speed, 1.0)
        painter.setPen(QPen(get_speed_color(speed, 0, self.max_speed), 25, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(QRectF(-radius, -radius, radius*2, radius*2), 225*16, int(-270*ratio*16))
        
        painter.setFont(self.font_big); painter.setPen(Qt.GlobalColor.white)
        painter.drawText(QRectF(-radius, -80, radius*2, 120), Qt.AlignmentFlag.AlignCenter, f"{int(speed)}")
        painter.setFont(self.font_mid); painter.setPen(QColor(180, 180, 180))
        painter.drawText(QRectF(-radius, 60, radius*2, 40), Qt.AlignmentFlag.AlignCenter, "KM/H")

    def draw_needle_gauge(self, painter, cx, cy, speed):
        radius = 200
        painter.setFont(self.font_ticks)
        max_val = int(self.max_speed)
        
        if max_val <= 60: main_step = 5
        elif max_val <= 120: main_step = 10
        elif max_val <= 260: main_step = 20
        elif max_val <= 360: main_step = 30
        else: main_step = 50
            
        if main_step == 30: sub_divisions = 3
        else: sub_divisions = 5
            
        sub_step = main_step / sub_divisions

        w_label = 3.5 * self.tick_width_scale
        w_mid = 2.5 * self.tick_width_scale
        w_small = 1.5 * self.tick_width_scale

        redline_start_val = max_val * 0.8
        redline_start_angle = 225 - (redline_start_val / max_val) * 270
        redline_span_angle = -270 * 0.2
        painter.setPen(QPen(QColor(220, 0, 0, 80), 12, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
        r_red = radius - 15
        painter.drawArc(QRectF(-r_red, -r_red, r_red*2, r_red*2), int(redline_start_angle*16), int(redline_span_angle*16))

        current_val = 0
        while current_val <= max_val + 0.01:
            angle_rad = np.radians(225 - (current_val / max_val) * 270)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            
            is_label = abs(current_val % main_step) < 0.01
            is_mid = abs(current_val % (main_step / 2)) < 0.01 and not is_label
            if is_mid and sub_divisions % 2 != 0: is_mid = False 

            if is_label:
                tick_len = 28
                painter.setPen(QPen(Qt.GlobalColor.white, w_label))
                p_text = QPointF(cos_a*(radius-65), -sin_a*(radius-65))
                painter.drawText(QRectF(p_text.x()-35, p_text.y()-15, 70, 30), Qt.AlignmentFlag.AlignCenter, str(int(current_val)))
            elif is_mid:
                tick_len = 20
                painter.setPen(QPen(QColor(200, 200, 200), w_mid))
            else:
                tick_len = 12
                painter.setPen(QPen(QColor(220, 220, 220), w_small))
            
            p_outer = QPointF(cos_a*(radius), -sin_a*(radius))
            p_inner = QPointF(cos_a*(radius-tick_len), -sin_a*(radius-tick_len))
            painter.drawLine(p_inner, p_outer)
            current_val += sub_step

        painter.save()
        current_angle = 225 - (min(speed, self.max_speed) / self.max_speed) * 270
        painter.rotate(-current_angle + 90)
        ptr_color = get_speed_color(speed, self.grad_min, self.grad_max)
        painter.setBrush(ptr_color); painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPolygon(QPolygonF([QPointF(-9, 0), QPointF(0, -radius+5), QPointF(9, 0)]))
        painter.restore()
        
        painter.setBrush(QColor(60, 60, 60)); painter.setPen(QPen(QColor(30, 30, 30), 2))
        painter.drawEllipse(QPointF(0,0), 20, 20)
        painter.setBrush(QColor(30, 30, 30)); painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(0,0), 12, 12)
        
        painter.setFont(self.font_needle_val); painter.setPen(Qt.GlobalColor.white)
        painter.drawText(QRectF(-150, 60, 300, 80), Qt.AlignmentFlag.AlignCenter, f"{int(speed)}")
        painter.setFont(self.font_ticks); painter.setPen(QColor(180, 180, 180))
        painter.drawText(QRectF(-50, 130, 100, 30), Qt.AlignmentFlag.AlignCenter, "KM/H")

    def draw_extra_info(self, painter, start_x, start_y, t, sats, alt):
        if not self.icon_clock: return
        line_h = 70; current_y = start_y
        def draw_row(icon, text):
            painter.drawPixmap(int(start_x), int(current_y), icon)
            path = QPainterPath(); path.addText(start_x + 80, current_y + 45, self.font_small, text)
            painter.setPen(QPen(Qt.GlobalColor.black, 5)); painter.setBrush(Qt.BrushStyle.NoBrush); painter.drawPath(path)
            painter.setPen(Qt.PenStyle.NoPen); painter.setBrush(Qt.GlobalColor.white); painter.drawPath(path)
            return line_h
        
        if self.show_time:
            mins, secs = int(t // 60), int(t % 60); mils = int((t * 100) % 100)
            current_y += draw_row(self.icon_clock, f"{mins:02d}:{secs:02d}.{mils:02d}")
        if self.show_sats: current_y += draw_row(self.icon_sat, f"{int(sats)} SATS")
        if self.show_alt: current_y += draw_row(self.icon_alt, f"{int(alt)} M")