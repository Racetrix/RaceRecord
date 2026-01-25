import numpy as np
from PyQt6.QtGui import QPainter, QPen, QColor, QFont, QBrush, QPainterPath, QTransform, QPolygonF, QImage, QRadialGradient
from PyQt6.QtCore import Qt, QPointF, QRectF
import math

# 渲染模式枚举
MODE_MAP = 0     
MODE_SPEED = 1   
MODE_GFORCE = 2  
MODE_ATTITUDE = 3
MODE_STUDIO = 4  # 演播室模式

# 样式枚举
MAP_STATIC = 0
MAP_DYNAMIC = 1
STYLE_DIGITAL = 0
STYLE_NEEDLE = 1
STYLE_LINEAR = 2
COLOR_SPEED = 0
COLOR_WHITE = 1

def qimage_to_numpy(qimg):
    qimg = qimg.convertToFormat(QImage.Format.Format_RGBA8888)
    width = qimg.width(); height = qimg.height()
    ptr = qimg.bits(); ptr.setsize(qimg.sizeInBytes())
    arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
    return np.ascontiguousarray(arr.copy())

def get_speed_color(speed, g_min, g_max):
    if g_max <= g_min: g_max = g_min + 1
    val = np.clip((speed - g_min) / (g_max - g_min), 0, 1)
    hue = int((1.0 - val) * 240) 
    return QColor.fromHsv(hue, 220, 255)

class Renderer:
    def __init__(self, data_manager):
        self.dm = data_manager 
        
        # === 演播室布局配置 ===
        self.layout = {
            'map':      {'show': True,  'x': 0.82, 'y': 0.25, 'scale': 0.8},
            'speed':    {'show': True,  'x': 0.50, 'y': 0.82, 'scale': 0.6},
            'gforce':   {'show': True,  'x': 0.15, 'y': 0.82, 'scale': 0.8},
            'attitude': {'show': False, 'x': 0.85, 'y': 0.82, 'scale': 0.6}
        }

        # === 基础参数 ===
        self.map_type = MAP_STATIC
        self.map_color_mode = COLOR_SPEED
        self.track_width = 15     
        self.car_size = 30
        self.map_zoom_factor = 1.0 
        self.grad_min = 0; self.grad_max = 160
        
        # === 2. 速度表参数 ===
        # 🔥 修复2：默认改为 STYLE_DIGITAL (0)，与 UI 默认选项对齐
        self.gauge_style = STYLE_DIGITAL 
        self.max_speed = 260 
        self.gauge_scale = 1.0
        
        # === 3. G值/姿态参数 ===
        self.g_scale = 0.5
        self.max_g = 1.5 
        self.g_invert_x = False 
        self.g_invert_y = False 
        
        self.att_scale = 1.0
        self.att_invert_roll = False
        self.att_invert_pitch = False

        # === 字体 ===
        self.font_big = QFont("Arial", 75, QFont.Weight.Black) 
        self.font_mid = QFont("Arial", 20, QFont.Weight.Bold)
        self.font_small = QFont("Arial", 14, QFont.Weight.Bold)
        self.font_ticks = QFont("Arial", 16, QFont.Weight.Bold) 

    def render(self, painter, w, h, current_time, transparent_bg, target_mode, bg_image=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        
        # 1. 绘制背景
        if bg_image and not bg_image.isNull():
            img_w = bg_image.width(); img_h = bg_image.height()
            scale = min(w / img_w, h / img_h)
            new_w = img_w * scale; new_h = img_h * scale
            offset_x = (w - new_w) / 2; offset_y = (h - new_h) / 2
            
            if not transparent_bg:
                painter.fillRect(0, 0, w, h, Qt.GlobalColor.black)
            painter.drawImage(QRectF(offset_x, offset_y, new_w, new_h), bg_image)
            
        elif not transparent_bg:
            painter.fillRect(0, 0, w, h, Qt.GlobalColor.black)
        else:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
            painter.fillRect(0, 0, w, h, Qt.GlobalColor.transparent)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

        if self.dm.df_proc is None: return
        state = self.dm.get_state_at_time(current_time)
        if not state: return

        cx, cy = w / 2, h / 2
        min_dim = max(100, min(w, h)) 

        # 3. 分发渲染逻辑
        if target_mode == MODE_STUDIO:
            self.render_component(painter, w, h, 'map', state, min_dim)
            self.render_component(painter, w, h, 'speed', state, min_dim)
            self.render_component(painter, w, h, 'gforce', state, min_dim)
            self.render_component(painter, w, h, 'attitude', state, min_dim)
            
        else:
            if target_mode == MODE_MAP:
                self.render_map(painter, w, h, state, current_time, transparent_bg)
            elif target_mode == MODE_SPEED:
                self.render_single_gauge(painter, cx, cy, min_dim, self.gauge_scale, lambda: 
                    self.draw_digital_gauge(painter, state['speed']) if self.gauge_style == STYLE_DIGITAL 
                    else self.draw_custom_needle_gauge(painter, state['speed']))
            elif target_mode == MODE_GFORCE:
                self.render_single_gauge(painter, cx, cy, min_dim, self.g_scale, lambda: self.draw_custom_gforce(painter, state))
            elif target_mode == MODE_ATTITUDE:
                self.render_single_gauge(painter, cx, cy, min_dim, self.att_scale, lambda: self.draw_attitude(painter, state))

    # 辅助：渲染单个组件
    def render_component(self, painter, w, h, key, state, min_dim):
        cfg = self.layout[key]
        if not cfg['show']: return
        
        painter.save()
        tx = cfg['x'] * w; ty = cfg['y'] * h
        painter.translate(tx, ty)
        
        base_factor = 1.2
        s = cfg['scale'] * (min_dim / 1000.0) * base_factor
        painter.scale(s, s)
        
        if key == 'map':
            # 🔥 修复1：去掉强制 MAP_DYNAMIC，现在听从 self.map_type 设置
            # 虚拟画布大小设为 400x400，让静态地图也能在这个框里居中显示
            self.render_map(painter, 400, 400, state, 0, True, is_widget=True)
            
        elif key == 'speed':
            if self.gauge_style == STYLE_DIGITAL: self.draw_digital_gauge(painter, state['speed'])
            elif self.gauge_style == STYLE_NEEDLE: self.draw_custom_needle_gauge(painter, state['speed'])
            # elif self.gauge_style == STYLE_LINEAR: self.draw_linear_gauge(painter, state['speed'])
        elif key == 'gforce':
            self.draw_custom_gforce(painter, state)
        elif key == 'attitude':
            self.draw_attitude(painter, state)
            
        painter.restore()

    def render_single_gauge(self, painter, cx, cy, min_dim, user_scale, draw_func):
        base_size = 250 
        scale = (min_dim / (base_size * 2.1)) * max(0.1, user_scale)
        painter.save(); painter.translate(cx, cy); painter.scale(scale, scale)
        draw_func(); painter.restore()

    # ================= 绘图函数 =================
    
    def render_map(self, painter, w, h, s, t, transparent_bg, is_widget=False):
        painter.save()
        if is_widget: cx, cy = 0, 0
        else: cx, cy = w / 2, h / 2
        
        # 获取用户设置的车标半径
        car_radius = self.car_size / 2
        
        if self.map_type == MAP_DYNAMIC:
            # === 动态模式 ===
            radius = 180 if is_widget else min(w, h) / 2 - 5
            
            # 1. 圆形遮罩
            path = QPainterPath(); path.addEllipse(QPointF(cx, cy), radius, radius)
            painter.setClipPath(path)
            
            # 2. 半透明黑底
            painter.setBrush(QColor(0, 0, 0, 100)); painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(cx, cy), radius, radius)
            
            # 3. 绘制赛道 (变换坐标系)
            painter.save()
            base_zoom = 3.0 * self.map_zoom_factor 
            speed_ratio = np.clip(s['speed'] / 200.0, 0, 1)
            zoom = base_zoom * (1.8 - np.sqrt(speed_ratio) * 1.0)
            rotation = -s['heading'] + 90
            tr = QTransform(); tr.translate(cx, cy); tr.rotate(rotation); tr.scale(zoom, zoom); tr.scale(1, -1); tr.translate(-s['mx'], -s['my'])
            painter.setTransform(tr, combine=True)
            
            points = [QPointF(x, y) for x, y in zip(self.dm.meter_x, self.dm.meter_y)]
            self.draw_track_lines(painter, points)
            
            # 4. 绘制车标 (保持之前的逻辑)
            real_radius = car_radius / max(0.01, zoom)
            pen_width = 2 / max(0.01, zoom)
            painter.setPen(QPen(Qt.GlobalColor.black, pen_width)) 
            painter.setBrush(QBrush(Qt.GlobalColor.white))      
            painter.drawEllipse(QPointF(s['mx'], s['my']), real_radius, real_radius)
            painter.restore()

            # 🔥 5. 绘制外圈装饰环 (重点修改：加粗)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            # 将线宽从 3 改为 8，确保缩放后依然清晰锐利
            painter.setPen(QPen(Qt.GlobalColor.white, 8)) 
            painter.drawEllipse(QPointF(cx, cy), radius, radius)
            
        else:
            # === 静态模式 (逻辑保持不变) ===
            if not is_widget: painter.setClipRect(0, 0, w, h)
            if is_widget:
                draw_area_w = 400; draw_area_h = 400
                painter.translate(-draw_area_w/2, -draw_area_h/2)
                w, h = draw_area_w, draw_area_h
                
            margin = 20
            avail_w = w - margin*2; avail_h = h - margin*2
            if avail_w * self.dm.aspect_ratio > avail_h: draw_h = avail_h; draw_w = avail_h / self.dm.aspect_ratio
            else: draw_w = avail_w; draw_h = avail_w * self.dm.aspect_ratio
            offset_x = (w - draw_w) / 2; offset_y = (h - draw_h) / 2
            def to_screen(nx, ny): return QPointF(offset_x + nx * draw_w, h - (offset_y + ny * draw_h))

            points = [to_screen(x, y) for x, y in zip(self.dm.norm_x, self.dm.norm_y)]
            self.draw_track_lines(painter, points)
            
            car_pos = to_screen(s['nx'], s['ny'])
            painter.setPen(QPen(Qt.GlobalColor.black, 2))
            painter.setBrush(QBrush(Qt.GlobalColor.white))
            painter.drawEllipse(car_pos, car_radius, car_radius)
            
        painter.restore()

    def draw_track_lines(self, painter, points):
        # 确保这里使用的是 self.track_width
        if self.map_color_mode == COLOR_SPEED:
            step = 2
            for i in range(0, len(points)-step, step):
                if i % 50 == 0: pass 
                spd = self.dm.cache['Speed_kmh'][i] if 'Speed_kmh' in self.dm.cache else 0
                c = get_speed_color(spd, self.grad_min, self.grad_max)
                # 🔥 确认：使用 self.track_width
                painter.setPen(QPen(c, self.track_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                painter.drawLine(points[i], points[i+step])
        else:
            c = Qt.GlobalColor.white
            # 🔥 确认：使用 self.track_width
            painter.setPen(QPen(c, self.track_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawPolyline(points)

    def draw_digital_gauge(self, painter, speed):
        r_base = 220; r_ticks_outer = 215; r_prog = 175; r_pointer_base = 242
        max_val = max(80, int(self.max_speed))
        dynamic_color = get_speed_color(speed, self.grad_min, self.grad_max)
        start_angle = 225; total_angle = 270
        current_ratio = min(speed / max_val, 1.0)
        span_angle = -total_angle * current_ratio
        
        painter.setPen(QPen(QColor(30, 30, 30), 15, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(QRectF(-r_prog, -r_prog, r_prog*2, r_prog*2), int(start_angle * 16), int(-total_angle * 16))
        
        if abs(span_angle) > 0.1:
            glow = QColor(dynamic_color); glow.setAlpha(50)
            painter.setPen(QPen(glow, 25, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawArc(QRectF(-r_prog, -r_prog, r_prog*2, r_prog*2), int(start_angle * 16), int(span_angle * 16))
            painter.setPen(QPen(dynamic_color, 15, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawArc(QRectF(-r_prog, -r_prog, r_prog*2, r_prog*2), int(start_angle * 16), int(span_angle * 16))

        total_ticks = 70
        painter.save(); painter.setFont(self.font_ticks)
        for i in range(total_ticks + 1):
            ratio = i / total_ticks
            angle_deg = start_angle - (ratio * total_angle)
            angle_rad = math.radians(angle_deg)
            cos_a = math.cos(angle_rad); sin_a = -math.sin(angle_rad)
            is_major = (i % 5 == 0)
            if is_major: tick_len = 18; width = 3; color = QColor(220, 220, 220) 
            else: tick_len = 10; width = 1; color = QColor(100, 100, 100)
            p_out = QPointF(r_ticks_outer * cos_a, r_ticks_outer * sin_a)
            p_in = QPointF((r_ticks_outer - tick_len) * cos_a, (r_ticks_outer - tick_len) * sin_a)
            painter.setPen(QPen(color, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap)); painter.drawLine(p_in, p_out)
        painter.restore()

        current_angle_deg = start_angle + span_angle
        painter.save(); painter.rotate(-current_angle_deg); painter.translate(r_pointer_base, 0)
        tri_len = 25; tri_w = 12
        pointer_poly = QPolygonF([QPointF(-tri_len, 0), QPointF(0, -tri_w), QPointF(0, tri_w)])
        painter.setBrush(QBrush(dynamic_color))
        sharp_pen = QPen(Qt.GlobalColor.white, 2); sharp_pen.setStyle(Qt.PenStyle.SolidLine); sharp_pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        painter.setPen(sharp_pen); painter.drawPolygon(pointer_poly); painter.restore()

        painter.setFont(QFont("Arial", 85, QFont.Weight.Bold)); painter.setPen(Qt.GlobalColor.white)
        painter.drawText(QRectF(-150, -60, 300, 100), Qt.AlignmentFlag.AlignCenter, f"{int(speed)}")
        painter.setFont(QFont("Arial", 18, QFont.Weight.Normal)); painter.setPen(QColor(180, 180, 180))
        painter.drawText(QRectF(-150, 40, 300, 40), Qt.AlignmentFlag.AlignCenter, "km/h")
        logo_font = QFont("Brush Script MT", 28, QFont.Weight.Bold); logo_font.setItalic(True)
        painter.setFont(logo_font); painter.setPen(QColor(220, 220, 220))
        painter.drawText(QRectF(-150, 100, 300, 60), Qt.AlignmentFlag.AlignCenter, "Racetrix")

    def draw_custom_needle_gauge(self, painter, speed):
        radius = 220
        max_val = max(80, int(self.max_speed))
        dynamic_color = get_speed_color(speed, self.grad_min, self.grad_max)
        
        # 1. 背景：透明
        
        # 2. 刻度步长
        if max_val <= 60: step_val = 5
        elif max_val <= 160: step_val = 10
        elif max_val <= 300: step_val = 20
        else: step_val = 30
        
        start_angle = 225; total_angle = 270
        steps = int(max_val / step_val)
        sub_divisions = 5 
        total_sub_ticks = steps * sub_divisions
        
        painter.save()
        bold_font = self.font_ticks; bold_font.setWeight(QFont.Weight.Black); painter.setFont(bold_font)
        
        for i in range(total_sub_ticks + 1):
            val = i * (step_val / sub_divisions)
            if val > max_val: break
            
            ratio = val / max_val; angle_rad = math.radians(225 - (ratio * 270))
            cos_a = math.cos(angle_rad); sin_a = -math.sin(angle_rad)
            is_major = (i % sub_divisions == 0)
            
            if is_major: 
                r_out = radius; r_in = radius - 35; width = 6; color = Qt.GlobalColor.white
            else: 
                r_out = radius - 2; r_in = radius - 18; width = 3; color = QColor(200, 200, 200) 
            
            p_out = QPointF(r_out * cos_a, r_out * sin_a); p_in = QPointF(r_in * cos_a, r_in * sin_a)
            painter.setPen(QPen(color, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)); painter.drawLine(p_in, p_out)
            
            if is_major:
                r_text = radius - 70; pt_text = QPointF(r_text * cos_a, r_text * sin_a)
                text_rect = QRectF(pt_text.x() - 35, pt_text.y() - 35, 70, 70)
                painter.setPen(QPen(Qt.GlobalColor.white, 2)) 
                painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, str(int(val)))
        painter.restore()
        
        # 3. 红区 (修复报错：使用 Qt.PenCapStyle.FlatCap)
        # 逻辑：寻找大约在 82% 处的那个大刻度作为起点
        raw_start_val = max_val * 0.82 
        # 吸附算法：向下取整到最近的 step_val
        red_start_val = int(raw_start_val / step_val) * step_val
        
        if red_start_val >= max_val: red_start_val = max_val - step_val
        
        red_start_ratio = red_start_val / max_val
        red_angle_start = 225 - (red_start_ratio * 270)
        red_angle_span = -((1.0 - red_start_ratio) * 270)
        
        # 🔥 修改这里：Qt.PenCapStyle.ButtCap -> Qt.PenCapStyle.FlatCap
        # FlatCap 就是平头端点，效果和 ButtCap 一样，能保证切口整齐
        painter.setPen(QPen(QColor(220, 0, 0, 200), 16, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
        r_red = radius - 15 
        painter.drawArc(QRectF(-r_red, -r_red, r_red*2, r_red*2), int(red_angle_start * 16), int(red_angle_span * 16))

        # 4. 指针
        current_ratio = min(speed / max_val, 1.05)
        painter.save(); painter.rotate(-(225 - (current_ratio * 270)))
        needle_len = radius - 25; root_w = 11; tail_len = 25 
        
        glow_color = QColor(dynamic_color); glow_color.setAlpha(40)
        painter.setBrush(QBrush(glow_color)); painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPolygon(QPolygonF([QPointF(0, -(root_w+8)), QPointF(needle_len+5, 0), QPointF(0, (root_w+8)), QPointF(-(tail_len+5), 0)]))
        
        glow_color.setAlpha(100); painter.setBrush(QBrush(glow_color))
        painter.drawPolygon(QPolygonF([QPointF(0, -(root_w+3)), QPointF(needle_len+2, 0), QPointF(0, (root_w+3)), QPointF(-(tail_len+2), 0)]))
        painter.setBrush(QBrush(dynamic_color))
        painter.drawPolygon(QPolygonF([QPointF(0, -root_w), QPointF(needle_len, 0), QPointF(0, root_w), QPointF(-tail_len, 0)]))
        painter.restore()
        
        # 5. 中心盖帽
        painter.setBrush(Qt.BrushStyle.NoBrush); painter.setPen(QPen(Qt.GlobalColor.white, 4)) 
        painter.drawEllipse(QPointF(0,0), 22, 22) 
        painter.setBrush(QColor(10, 10, 10)); painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(0,0), 18, 18)
        
        # 6. 底部数字
        painter.setFont(QFont("Arial", 80, QFont.Weight.Black)); painter.setPen(Qt.GlobalColor.white)
        painter.drawText(QRectF(-150, 75, 300, 90), Qt.AlignmentFlag.AlignCenter, f"{int(speed)}")
        painter.setFont(self.font_mid); painter.setPen(QColor(200, 200, 200))
        painter.drawText(QRectF(-150, 155, 300, 40), Qt.AlignmentFlag.AlignCenter, "KM/H")

    def draw_custom_gforce(self, painter, s):
        radius = 120; max_g = self.max_g 
        raw_gx = s['lat_g']; raw_gy = s['lon_g'] 
        gx = -raw_gx if self.g_invert_x else raw_gx
        gy = -raw_gy if self.g_invert_y else raw_gy 
        
        painter.setBrush(QColor(0,0,0,100)); painter.setPen(QPen(Qt.GlobalColor.white, 5)) 
        painter.drawEllipse(QPointF(0,0), radius, radius)
        
        painter.setBrush(Qt.BrushStyle.NoBrush); painter.setPen(QPen(QColor(255,255,255,100), 1, Qt.PenStyle.DashLine))
        painter.drawEllipse(QPointF(0,0), radius*0.33, radius*0.33); painter.drawEllipse(QPointF(0,0), radius*0.66, radius*0.66)
        painter.setPen(QPen(QColor(255,255,255,50), 1, Qt.PenStyle.SolidLine))
        painter.drawLine(-radius, 0, radius, 0); painter.drawLine(0, -radius, 0, radius)
        
        px = (gx / max_g) * radius; py = -(gy / max_g) * radius 
        if np.sqrt(px**2+py**2) > radius: f=radius/np.sqrt(px**2+py**2); px*=f; py*=f
        
        grad = QRadialGradient(px, py, 15); grad.setColorAt(0, QColor(255, 100, 100)); grad.setColorAt(1, QColor(200, 0, 0))
        painter.setPen(Qt.PenStyle.NoPen); painter.setBrush(QBrush(grad)); painter.drawEllipse(QPointF(px, py), 12, 12)
        
        painter.setFont(self.font_small); painter.setPen(QColor(180, 180, 180))
        painter.drawText(QRectF(-100, -radius - 40, 200, 30), Qt.AlignmentFlag.AlignCenter, "G-FORCE")
        current_g = np.sqrt(gx**2 + gy**2)
        painter.setFont(self.font_mid); painter.setPen(Qt.GlobalColor.white)
        painter.drawText(QRectF(-100, radius + 10, 200, 40), Qt.AlignmentFlag.AlignCenter, f"{current_g:.2f}G")

    def draw_attitude(self, painter, s):
        radius = 120
        roll = s['roll'] * (-1 if self.att_invert_roll else 1)
        pitch = s['pitch'] * (-1 if self.att_invert_pitch else 1)
        path = QPainterPath(); path.addEllipse(QPointF(0,0), radius, radius)
        painter.setClipPath(path)
        painter.rotate(-roll); offset = pitch * 4 
        painter.fillRect(QRectF(-300, -300+offset, 600, 300), QColor(0, 140, 255))
        painter.fillRect(QRectF(-300, 0+offset, 600, 300), QColor(140, 100, 50))
        painter.setPen(QPen(Qt.GlobalColor.white, 3)); painter.drawLine(-300, int(offset), 300, int(offset))
        painter.rotate(roll)
        painter.setPen(QPen(Qt.GlobalColor.yellow, 5)); painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawLine(-40, 0, -15, 0); painter.drawLine(15, 0, 40, 0)
        painter.drawPoint(0,0); painter.drawLine(0, 0, 0, 10)