import numpy as np
from PyQt6.QtGui import QPainter, QPen, QColor, QFont, QBrush, QPainterPath, QTransform, QPolygonF, QImage, QRadialGradient
from PyQt6.QtCore import Qt, QPointF, QRectF
import math
import qtawesome as qta 

# 渲染模式枚举
MODE_MAP = 0     
MODE_SPEED = 1   
MODE_GFORCE = 2  
MODE_ATTITUDE = 3

# 样式枚举
MAP_STATIC = 0
MAP_DYNAMIC = 1
STYLE_DIGITAL = 0
STYLE_NEEDLE = 1
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
        
        # === 1. 地图参数 ===
        self.map_type = MAP_STATIC
        self.map_color_mode = COLOR_SPEED
        self.track_width = 15     
        self.car_size = 30
        self.map_zoom_factor = 1.0 
        self.grad_min = 0; self.grad_max = 160
        
        # === 2. 速度表参数 ===
        self.gauge_style = STYLE_NEEDLE 
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

    def render(self, painter, w, h, current_time, transparent_bg, target_mode):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        
        if transparent_bg:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
            painter.fillRect(0, 0, w, h, Qt.GlobalColor.transparent)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        else:
            painter.fillRect(0, 0, w, h, Qt.GlobalColor.black)

        if self.dm.df_proc is None:
            painter.setPen(QColor(100, 100, 100)); painter.setFont(QFont("Arial", 20))
            painter.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, "等待加载数据...")
            return

        state = self.dm.get_state_at_time(current_time)
        if not state: return

        cx, cy = w / 2, h / 2
        min_dim = max(100, min(w, h)) 

        if target_mode == MODE_MAP:
            self.render_map(painter, w, h, state, current_time, transparent_bg)
            
        elif target_mode == MODE_SPEED:
            base_r = 250
            scale = (min_dim / (base_r * 2.1)) * max(0.1, self.gauge_scale)
            painter.save(); painter.translate(cx, cy); painter.scale(scale, scale)
            if self.gauge_style == STYLE_DIGITAL: self.draw_digital_gauge(painter, state['speed'])
            else: self.draw_custom_needle_gauge(painter, state['speed'])
            painter.restore()
            
        elif target_mode == MODE_GFORCE:
            base_size = 300 
            scale = (min_dim / base_size) * max(0.1, self.g_scale)
            painter.save(); painter.translate(cx, cy); painter.scale(scale, scale)
            self.draw_custom_gforce(painter, state)
            painter.restore()
            
        elif target_mode == MODE_ATTITUDE:
            base_size = 300
            scale = (min_dim / base_size) * max(0.1, self.att_scale)
            painter.save(); painter.translate(cx, cy); painter.scale(scale, scale)
            self.draw_attitude(painter, state)
            painter.restore()

    # ================= 1. 地图渲染 (支持圆形遮罩) =================
    def render_map(self, painter, w, h, s, t, transparent_bg):
        painter.save()
        
        if self.map_type == MAP_DYNAMIC:
            # 🔥 修改：动态地图使用圆形遮罩 (Radar Style)
            cx, cy = w / 2, h / 2
            radius = min(w, h) / 2 - 5 # 留一点边距
            
            # 1. 设定圆形剪裁路径
            path = QPainterPath()
            path.addEllipse(QPointF(cx, cy), radius, radius)
            painter.setClipPath(path)
            
            # 2. 画背景 (半透明黑) 和 粗边框 (白)
            # 让它看起来像个仪表，而不是悬空的切片
            painter.setBrush(QColor(0, 0, 0, 120)) 
            painter.setPen(QPen(Qt.GlobalColor.white, 5)) # 粗边框
            painter.drawEllipse(QPointF(cx, cy), radius, radius)
            
        else:
            # 静态地图：保持矩形全屏
            painter.setClipRect(0, 0, w, h)
        
        if self.map_type == MAP_STATIC:
            margin = 20
            avail_w, avail_h = w - margin*2, h - margin*2
            if avail_w * self.dm.aspect_ratio > avail_h:
                draw_h = avail_h; draw_w = avail_h / self.dm.aspect_ratio
            else:
                draw_w = avail_w; draw_h = avail_w * self.dm.aspect_ratio
            
            offset_x = (w - draw_w) / 2; offset_y = (h - draw_h) / 2
            def to_screen(nx, ny): return QPointF(offset_x + nx * draw_w, h - (offset_y + ny * draw_h))

            points = [to_screen(x, y) for x, y in zip(self.dm.norm_x, self.dm.norm_y)]
            self.draw_track_lines(painter, points)
            
            car_pos = to_screen(s['nx'], s['ny'])
            painter.setPen(QPen(Qt.GlobalColor.black, 3)); painter.setBrush(QBrush(Qt.GlobalColor.white))
            painter.drawEllipse(car_pos, self.car_size, self.car_size)
            
        else:
            # Dynamic Logic
            base_zoom = 3.0 * self.map_zoom_factor 
            speed_ratio = np.clip(s['speed'] / 200.0, 0, 1)
            zoom = base_zoom * (1.8 - np.sqrt(speed_ratio) * 1.0)
            
            rotation = -s['heading'] + 90
            tr = QTransform()
            tr.translate(w/2, h/2); tr.rotate(rotation)
            tr.scale(zoom, zoom); tr.scale(1, -1)
            tr.translate(-s['mx'], -s['my'])
            
            painter.setTransform(tr, combine=True)
            points = [QPointF(x, y) for x, y in zip(self.dm.meter_x, self.dm.meter_y)]
            self.draw_track_lines(painter, points)
            
            real_sz = self.car_size / max(0.01, zoom)
            painter.setPen(QPen(Qt.GlobalColor.black, 2/max(0.01, zoom)))
            painter.setBrush(QBrush(Qt.GlobalColor.white))
            painter.drawEllipse(QPointF(s['mx'], s['my']), real_sz, real_sz)
            
        painter.restore()

    def draw_track_lines(self, painter, points):
        if self.map_color_mode == COLOR_SPEED:
            step = 2
            for i in range(0, len(points)-step, step):
                if i % 50 == 0: pass 
                spd = self.dm.cache['Speed_kmh'][i] if 'Speed_kmh' in self.dm.cache else 0
                c = get_speed_color(spd, self.grad_min, self.grad_max)
                painter.setPen(QPen(c, self.track_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                painter.drawLine(points[i], points[i+step])
        else:
            c = Qt.GlobalColor.white
            painter.setPen(QPen(c, self.track_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawPolyline(points)

    # ================= 2. 速度表渲染 =================
    def draw_custom_needle_gauge(self, painter, speed):
        radius = 220
        max_val = max(80, int(self.max_speed))
        dynamic_color = get_speed_color(speed, self.grad_min, self.grad_max)
        
        painter.setBrush(QColor(0, 0, 0, 0)); painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(0,0), radius, radius)

        candidates = [5, 10, 20, 25, 30]
        step_val = 10
        for cand in candidates:
            count = max_val / cand
            if 10 <= count <= 20: step_val = cand
            elif count < 10 and cand == 5: step_val = 5
        if max_val > 200 and step_val == 10: step_val = 20 
        
        start_angle = 225; total_angle = 270
        steps = int(max_val / step_val)
        sub_divisions = 5 
        total_sub_ticks = steps * sub_divisions
        
        painter.save(); painter.setFont(self.font_ticks)
        for i in range(total_sub_ticks + 1):
            val = i * (step_val / sub_divisions)
            if val > max_val: break
            ratio = val / max_val
            angle_rad = math.radians(225 - (ratio * 270))
            cos_a = math.cos(angle_rad); sin_a = -math.sin(angle_rad)
            is_major = (i % sub_divisions == 0)
            
            if is_major:
                r_out = radius; r_in = radius - 35; width = 5; color = Qt.GlobalColor.white
            else:
                r_out = radius - 2; r_in = radius - 18; width = 2; color = QColor(180, 180, 180) 
                
            p_out = QPointF(r_out * cos_a, r_out * sin_a)
            p_in = QPointF(r_in * cos_a, r_in * sin_a)
            painter.setPen(QPen(color, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(p_in, p_out)
            
            if is_major:
                r_text = radius - 70
                pt_text = QPointF(r_text * cos_a, r_text * sin_a)
                text_rect = QRectF(pt_text.x() - 30, pt_text.y() - 30, 60, 60)
                painter.setPen(QPen(Qt.GlobalColor.white, 1))
                painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, str(int(val)))
        painter.restore()
        
        redline_angle_start = (225 - (0.85 * 270))
        painter.setPen(QPen(QColor(220, 0, 0, 200), 12, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
        r_red = radius - 15 
        painter.drawArc(QRectF(-r_red, -r_red, r_red*2, r_red*2), int(redline_angle_start * 16), int(-(270 * 0.15) * 16))

        current_ratio = min(speed / max_val, 1.05)
        painter.save(); painter.rotate(-(225 - (current_ratio * 270)))
        
        needle_len = radius - 25; root_w = 9; tail_len = 25
        glow_color = QColor(dynamic_color); glow_color.setAlpha(40)
        painter.setBrush(QBrush(glow_color)); painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPolygon(QPolygonF([QPointF(0, -(root_w+8)), QPointF(needle_len+5, 0), QPointF(0, (root_w+8)), QPointF(-(tail_len+5), 0)]))
        glow_color.setAlpha(100); painter.setBrush(QBrush(glow_color))
        painter.drawPolygon(QPolygonF([QPointF(0, -(root_w+3)), QPointF(needle_len+2, 0), QPointF(0, (root_w+3)), QPointF(-(tail_len+2), 0)]))
        painter.setBrush(QBrush(dynamic_color))
        painter.drawPolygon(QPolygonF([QPointF(0, -root_w), QPointF(needle_len, 0), QPointF(0, root_w), QPointF(-tail_len, 0)]))
        painter.restore()
        
        painter.setBrush(Qt.BrushStyle.NoBrush); painter.setPen(QPen(Qt.GlobalColor.white, 3))
        painter.drawEllipse(QPointF(0,0), 20, 20)
        painter.setBrush(QColor(10, 10, 10)); painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(0,0), 18, 18)
        
        painter.setFont(self.font_big); painter.setPen(Qt.GlobalColor.white)
        painter.drawText(QRectF(-150, 75, 300, 90), Qt.AlignmentFlag.AlignCenter, f"{int(speed)}")
        painter.setFont(self.font_mid); painter.setPen(QColor(200, 200, 200))
        painter.drawText(QRectF(-150, 155, 300, 40), Qt.AlignmentFlag.AlignCenter, "KM/H")

    # ================= 2.1 数字圆环表 (紧贴刻度版) =================
    def draw_digital_gauge(self, painter, speed):
        # --- 尺寸定义 ---
        r_base = 220
        r_ticks_outer = 215   # 刻度线最外端 (向外推，填补空隙)
        r_prog = 175          # 进度条半径 (内侧)
        r_pointer_base = 242  # 指针底座位置 (更外侧)
        
        max_val = max(80, int(self.max_speed))
        dynamic_color = get_speed_color(speed, self.grad_min, self.grad_max)
        
        # 1. 绘制进度条 (最内层)
        start_angle = 225
        total_angle = 270
        current_ratio = min(speed / max_val, 1.0)
        span_angle = -total_angle * current_ratio
        
        # 底槽
        painter.setPen(QPen(QColor(30, 30, 30), 15, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(QRectF(-r_prog, -r_prog, r_prog*2, r_prog*2), int(start_angle * 16), int(-total_angle * 16))
        
        # 动态进度 (光晕+实体)
        if abs(span_angle) > 0.1:
            glow = QColor(dynamic_color); glow.setAlpha(50)
            painter.setPen(QPen(glow, 25, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawArc(QRectF(-r_prog, -r_prog, r_prog*2, r_prog*2), int(start_angle * 16), int(span_angle * 16))
            
            painter.setPen(QPen(dynamic_color, 15, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawArc(QRectF(-r_prog, -r_prog, r_prog*2, r_prog*2), int(start_angle * 16), int(span_angle * 16))

        # 2. 绘制刻度 (中间层)
        total_ticks = 70
        painter.save()
        painter.setFont(self.font_ticks)
        
        for i in range(total_ticks + 1):
            ratio = i / total_ticks
            angle_deg = start_angle - (ratio * total_angle)
            angle_rad = math.radians(angle_deg)
            cos_a = math.cos(angle_rad); sin_a = -math.sin(angle_rad)
            
            is_major = (i % 5 == 0)
            if is_major:
                tick_len = 18; width = 3; color = QColor(220, 220, 220) # 加长刻度
            else:
                tick_len = 10; width = 1; color = QColor(100, 100, 100)
            
            # 刻度由 r_ticks_outer 向内画
            p_out = QPointF(r_ticks_outer * cos_a, r_ticks_outer * sin_a)
            p_in = QPointF((r_ticks_outer - tick_len) * cos_a, (r_ticks_outer - tick_len) * sin_a)
            
            painter.setPen(QPen(color, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
            painter.drawLine(p_in, p_out)
            
        painter.restore()

        # 3. 绘制指针 (最外层，向内指，紧贴刻度)
        current_angle_deg = start_angle + span_angle
        
        painter.save()
        painter.rotate(-current_angle_deg) 
        
        # 移到最外圈底座位置
        painter.translate(r_pointer_base, 0)
        
        # 定义三角形 (向左/向圆心方向指)
        # 尖端计算：242 - 25 = 217 (刻度在215，留2px缝隙，视觉上紧贴)
        tri_len = 25 
        tri_w = 12
        
        # 坐标系：(0,0)是外侧底座，X轴负方向是圆心
        pointer_poly = QPolygonF([
            QPointF(-tri_len, 0),    # 尖端 (向内伸出 25px)
            QPointF(0, -tri_w),      # 上底角
            QPointF(0, tri_w)        # 下底角
        ])
        
        painter.setBrush(QBrush(dynamic_color))
        
        # 尖锐描边
        sharp_pen = QPen(Qt.GlobalColor.white, 2)
        sharp_pen.setStyle(Qt.PenStyle.SolidLine)
        sharp_pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        painter.setPen(sharp_pen)
        
        painter.drawPolygon(pointer_poly)
        painter.restore()

        # 4. 中心文字
        painter.setFont(QFont("Arial", 85, QFont.Weight.Bold))
        painter.setPen(Qt.GlobalColor.white)
        painter.drawText(QRectF(-150, -60, 300, 100), Qt.AlignmentFlag.AlignCenter, f"{int(speed)}")
        
        painter.setFont(QFont("Arial", 18, QFont.Weight.Normal))
        painter.setPen(QColor(180, 180, 180))
        painter.drawText(QRectF(-150, 40, 300, 40), Qt.AlignmentFlag.AlignCenter, "km/h")
        
        logo_font = QFont("Brush Script MT", 28, QFont.Weight.Bold)
        logo_font.setItalic(True)
        painter.setFont(logo_font)
        painter.setPen(QColor(220, 220, 220))
        painter.drawText(QRectF(-150, 100, 300, 60), Qt.AlignmentFlag.AlignCenter, "Racetrix")

    # ================= 3. G值球渲染 (修复边框过细) =================
    def draw_custom_gforce(self, painter, s):
        radius = 120
        max_g = self.max_g 
        
        raw_gx = s['lat_g']; raw_gy = s['lon_g'] 
        gx = -raw_gx if self.g_invert_x else raw_gx
        gy = -raw_gy if self.g_invert_y else raw_gy 
        
        # 1. 背景 + 粗边框
        painter.setBrush(QColor(0,0,0,100)) # 半透明黑底
        
        # 🔥 修改：边框由 1px 改为 5px 白色实线
        painter.setPen(QPen(Qt.GlobalColor.white, 5)) 
        painter.drawEllipse(QPointF(0,0), radius, radius)
        
        # 内部虚线 (保持细一点)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(255,255,255,100), 1, Qt.PenStyle.DashLine))
        painter.drawEllipse(QPointF(0,0), radius*0.33, radius*0.33)
        painter.drawEllipse(QPointF(0,0), radius*0.66, radius*0.66)
        
        # 十字准星
        painter.setPen(QPen(QColor(255,255,255,50), 1, Qt.PenStyle.SolidLine))
        painter.drawLine(-radius, 0, radius, 0)
        painter.drawLine(0, -radius, 0, radius)
        
        px = (gx / max_g) * radius
        py = -(gy / max_g) * radius 
        if np.sqrt(px**2+py**2) > radius: f=radius/np.sqrt(px**2+py**2); px*=f; py*=f
        
        grad = QRadialGradient(px, py, 15)
        grad.setColorAt(0, QColor(255, 100, 100)); grad.setColorAt(1, QColor(200, 0, 0))
        painter.setPen(Qt.PenStyle.NoPen); painter.setBrush(QBrush(grad))
        painter.drawEllipse(QPointF(px, py), 12, 12)
        
        painter.setFont(self.font_small); painter.setPen(QColor(180, 180, 180))
        painter.drawText(QRectF(-100, -radius - 40, 200, 30), Qt.AlignmentFlag.AlignCenter, "G-FORCE")
        
        current_g = np.sqrt(gx**2 + gy**2)
        painter.setFont(self.font_mid); painter.setPen(Qt.GlobalColor.white)
        painter.drawText(QRectF(-100, radius + 10, 200, 40), Qt.AlignmentFlag.AlignCenter, f"{current_g:.2f}G")

    # ================= 4. 姿态仪渲染 =================
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