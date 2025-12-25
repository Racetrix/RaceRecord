import numpy as np
from PyQt6.QtGui import QPainter, QPen, QColor, QFont, QBrush, QPainterPath, QTransform, QPolygonF, QImage
from PyQt6.QtCore import Qt, QPointF, QRectF
import qtawesome as qta 

# 🔥 引入新模块
from race_gauges import DigitalGauge, NeedleGauge, LinearGauge, STYLE_DIGITAL, STYLE_NEEDLE, STYLE_LINEAR

# ================= 常量定义 =================
RESOLUTION = (1920, 1080)
MODE_PATH = 0 
MODE_GAUGE = 1

# 地图视角
MAP_STATIC_NORTH = 0  # 北向
MAP_DYNAMIC_HEAD = 1  # 车头向

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
        
        self.gauge_style = STYLE_LINEAR # 默认用新样式
        self.max_speed = 200      
        self.show_gauge = True    
        self.show_extra = False    
        self.show_time = True
        self.show_sats = True
        self.show_alt = False     
        
        self.gauge_scale = 1.0      
        self.gauge_offset_x = 0     
        self.gauge_offset_y = 0     
        self.tick_width_scale = 1.0 
        
        self.path_color_mode = COLOR_SPEED
        self.grad_min = 0.0
        self.grad_max = 160.0
        
        self.enable_dynamic_zoom = True

        # 🔥 初始化仪表实例
        self.gauges = {
            STYLE_DIGITAL: DigitalGauge(),
            STYLE_NEEDLE: NeedleGauge(),
            STYLE_LINEAR: LinearGauge()
        }

        self.font_small = QFont("Consolas", 26, QFont.Weight.Bold)
        
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

    # 🔥🔥 修改：通过代理类绘制仪表 🔥🔥
    def draw_gauge_manual(self, painter, w, h, s, t):
        if self.show_gauge and self.gauge_style in self.gauges:
            # 1. 绘制速度表
            base_cx = w / 2; base_cy = h / 2
            target_cx = base_cx + self.gauge_offset_x
            target_cy = base_cy + self.gauge_offset_y
            
            gauge_instance = self.gauges[self.gauge_style]
            gauge_instance.render(painter, target_cx, target_cy, s['speed'], self.max_speed, self.gauge_scale, self.tick_width_scale)

        if self.show_extra:
            start_x = w - 350
            start_y = h - 300
            self.draw_extra_info(painter, start_x, start_y, t, s['sats'], s['alt'])

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