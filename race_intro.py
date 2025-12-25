import random
import numpy as np
from PyQt6.QtGui import QPainter, QColor, QFont, QPen, QLinearGradient, QBrush, QPainterPath
from PyQt6.QtCore import Qt, QPointF, QRectF

# 定义样式常量
INTRO_CLASSIC = 0
INTRO_GLITCH = 1
INTRO_SCAN = 2
INTRO_TYPEWRITER = 3

INTRO_NAMES = {
    INTRO_CLASSIC: "Classic Fade (经典渐隐)",
    INTRO_GLITCH: "Cyber Glitch (赛博故障)",
    INTRO_SCAN: "Speed Scan (极速扫描)",
    INTRO_TYPEWRITER: "Typewriter (打字机)"
}

class IntroRenderer:
    def __init__(self):
        self.duration = 2.5 # 统一片头时长 2.5秒
        self.logo_text = "Racetrix"
        self.base_font_size = 140

    def render(self, painter, w, h, t, style, font_size=140):
        """
        统一渲染入口
        t: 当前片头时间 (0.0 ~ self.duration)
        """
        self.logo_text = "Racetrix"
        
        painter.save()
        
        # 统一背景 (透明或黑底由外部painter决定，这里只画内容)
        # 这里我们假设是在透明/黑色层上绘制
        
        if style == INTRO_CLASSIC:
            self._draw_classic(painter, w, h, t, font_size)
        elif style == INTRO_GLITCH:
            self._draw_glitch(painter, w, h, t, font_size)
        elif style == INTRO_SCAN:
            self._draw_scan(painter, w, h, t, font_size)
        elif style == INTRO_TYPEWRITER:
            self._draw_typewriter(painter, w, h, t, font_size)
            
        painter.restore()

    def _get_font(self, size, italic=True):
        return QFont("Arial", size, QFont.Weight.Bold, italic)

    def _draw_text_centered(self, painter, w, h, text, font):
        painter.setFont(font)
        metrics = painter.fontMetrics()
        tw = metrics.horizontalAdvance(text)
        th = metrics.height()
        x = (w - tw) / 2
        y = (h + th) / 2 - metrics.descent()
        return x, y, tw, th

    # === 样式 1: 经典渐隐 ===
    def _draw_classic(self, painter, w, h, t, size):
        # 0-1.5s 静止，1.5-2.5s 放大渐隐
        static_dur = 1.5
        fade_dur = 1.0
        
        opacity = 1.0
        scale = 1.0
        
        if t > static_dur:
            prog = (t - static_dur) / fade_dur
            opacity = max(0, 1.0 - prog)
            scale = 1.0 + prog * 0.5
            
        painter.translate(w/2, h/2)
        painter.scale(scale, scale)
        painter.translate(-w/2, -h/2)
        painter.setOpacity(opacity)
        
        font = self._get_font(size)
        x, y, tw, th = self._draw_text_centered(painter, w, h, self.logo_text, font)
        
        # 绿色渐变
        gradient = QLinearGradient(x, y - th, x, y)
        gradient.setColorAt(0.0, QColor(0, 230, 118))
        gradient.setColorAt(1.0, QColor(27, 94, 32))
        
        path = QPainterPath()
        path.addText(x, y, font, self.logo_text)
        
        painter.setBrush(QBrush(gradient))
        painter.setPen(QPen(QColor(0, 255, 127), 2))
        painter.drawPath(path)

    # === 样式 2: 赛博故障 ===
    def _draw_glitch(self, painter, w, h, t, size):
        # 整个过程都在抖动，最后0.5秒消失
        opacity = 1.0 if t < 2.0 else max(0, 1.0 - (t-2.0)/0.5)
        painter.setOpacity(opacity)
        
        font = self._get_font(size)
        base_x, base_y, tw, th = self._draw_text_centered(painter, w, h, self.logo_text, font)
        
        # 随机抖动参数
        # 利用时间t生成伪随机，保证同一帧渲染结果一致(不闪瞎眼)
        random.seed(int(t * 20)) 
        
        # 画两层错位 (红/青色分离)
        offset_r_x = random.randint(-5, 5)
        offset_c_x = random.randint(-5, 5)
        
        # 红色层
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 0, 0, 180))
        path_r = QPainterPath()
        path_r.addText(base_x + offset_r_x, base_y, font, self.logo_text)
        painter.drawPath(path_r)
        
        # 青色层
        painter.setBrush(QColor(0, 255, 255, 180))
        path_c = QPainterPath()
        path_c.addText(base_x + offset_c_x, base_y, font, self.logo_text)
        painter.drawPath(path_c)
        
        # 白色主层
        painter.setBrush(QColor(255, 255, 255))
        path_main = QPainterPath()
        path_main.addText(base_x, base_y, font, self.logo_text)
        painter.drawPath(path_main)
        
        # 随机横条遮挡 (Glitch Blocks)
        painter.setBrush(QColor(20, 20, 20)) # 深色遮挡
        for _ in range(3):
            bar_h = random.randint(5, 20)
            bar_y = random.randint(int(base_y - th), int(base_y))
            painter.drawRect(int(base_x - 20), bar_y, int(tw + 40), bar_h)

    # === 样式 3: 极速扫描 ===
    def _draw_scan(self, painter, w, h, t, size):
        # 0-1.5s 扫描出现，1.5-2.5s 保持并消失
        scan_dur = 1.5
        font = self._get_font(size)
        x, y, tw, th = self._draw_text_centered(painter, w, h, self.logo_text, font)
        
        # 创建裁剪区域
        path = QPainterPath()
        path.addText(x, y, font, self.logo_text)
        
        painter.setClipPath(path)
        
        # 绘制扫描进度
        if t < scan_dur:
            progress = t / scan_dur
            scan_x = x + tw * progress
            
            # 已扫描部分 (实心绿)
            painter.fillRect(int(x), int(y-th*1.2), int(tw * progress), int(th*1.5), QColor(0, 255, 127))
            
            # 扫描线 (高亮白)
            painter.fillRect(int(scan_x), int(y-th*1.2), 5, int(th*1.5), QColor(255, 255, 255))
        else:
            # 扫描完成，显示全绿，最后渐隐
            op = 1.0 if t < 2.0 else max(0, 1.0 - (t-2.0)/0.5)
            painter.setOpacity(op)
            painter.fillRect(int(x), int(y-th*1.2), int(tw), int(th*1.5), QColor(0, 255, 127))

    # === 样式 4: 打字机 ===
    def _draw_typewriter(self, painter, w, h, t, size):
        total_chars = len(self.logo_text)
        type_dur = 1.5 # 1.5秒打完
        char_interval = type_dur / total_chars
        
        # 计算当前显示几个字
        current_char_count = min(total_chars, int(t / char_interval) + 1)
        if t > 2.0: # 最后0.5秒渐隐
            painter.setOpacity(max(0, 1.0 - (t-2.0)/0.5))
            
        display_text = self.logo_text[:current_char_count]
        
        font = self._get_font(size, italic=False) # 打字机通常不用斜体
        font.setFamily("Consolas") # 用等宽字体更有感觉
        
        x, y, tw, th = self._draw_text_centered(painter, w, h, self.logo_text, font) # 用全长算居中
        
        # 绘制文字
        painter.setPen(QColor(0, 255, 0))
        painter.drawText(int(x), int(y), display_text)
        
        # 光标闪烁
        if t < 2.0 and int(t * 10) % 2 == 0:
            # 计算当前文字宽度
            metrics = painter.fontMetrics()
            curr_w = metrics.horizontalAdvance(display_text)
            painter.fillRect(int(x + curr_w + 5), int(y - th + metrics.descent()), 15, int(th), QColor(0, 255, 0))