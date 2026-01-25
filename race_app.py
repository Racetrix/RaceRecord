import sys, os
from unittest.mock import patch
if getattr(sys, 'frozen', False):
    with patch('importlib.metadata.version') as mock_version:
        mock_version.return_value = '2.37.2'  # 你的imageio版本
        import imageio
else:
    import imageio

import cv2
import numpy as np


from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QComboBox, QGroupBox, 
    QMessageBox, QProgressBar, QRadioButton, QButtonGroup, 
    QTabWidget, QCheckBox, QSlider, QScrollArea, QSpinBox, QDoubleSpinBox
)
from PyQt6.QtGui import QPainter, QPen, QColor, QFont, QImage
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal

from race_data import DataManager
from race_render import (
    Renderer, qimage_to_numpy, 
    MODE_MAP, MODE_SPEED, MODE_GFORCE, MODE_ATTITUDE, MODE_STUDIO,
    MAP_STATIC, MAP_DYNAMIC, STYLE_DIGITAL, STYLE_NEEDLE, COLOR_SPEED, COLOR_WHITE
)

STYLESHEET = """
QMainWindow { background-color: #181818; }
QWidget { font-family: 'Microsoft YaHei', sans-serif; font-size: 12px; color: #EEE; }
QTabWidget::pane { border: 1px solid #444; background: #202020; }
QTabBar::tab { background: #333; color: #AAA; padding: 6px 12px; border-top-left-radius: 4px; border-top-right-radius: 4px; margin-right: 2px; }
QTabBar::tab:selected { background: #4EC9B0; color: #000; font-weight: bold; }
QGroupBox { border: 1px solid #444; border-radius: 4px; margin-top: 10px; padding-top: 10px; font-weight: bold; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #4EC9B0; top: 0px; }
QPushButton { background: #333; border: 1px solid #555; padding: 5px; border-radius: 4px; min-height: 20px;}
QPushButton:hover { background: #444; border-color: #4EC9B0; }
QLabel { color: #BBB; }
QCheckBox { spacing: 5px; }
QSlider::groove:horizontal { border: 1px solid #333; height: 6px; background: #222; margin: 2px 0; }
QSlider::handle:horizontal { background: #4EC9B0; border: 1px solid #4EC9B0; width: 12px; height: 12px; margin: -4px 0; border-radius: 6px; }
"""
class VideoPlayerThread(QThread):
    frame_ready = pyqtSignal(QImage)
    
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.running = True
        self.target_time = 0.0
        
    def run(self):
        # 尝试使用 MSMF (Windows 硬件加速后端)
        cap = cv2.VideoCapture(self.path, cv2.CAP_MSMF)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.path)
            
        current_pos = 0.0
        
        while self.running:
            # 计算时间差：目标时间 - 当前视频位置
            diff = self.target_time - current_pos
            
            # 情况A: 偏差太大 (>1秒) -> 直接跳转 (Seek)
            if abs(diff) > 1.0:
                cap.set(cv2.CAP_PROP_POS_MSEC, self.target_time * 1000)
                current_pos = self.target_time
                ret, frame = cap.read()
                if ret: self.process_and_emit(frame)
                
            # 情况B: 稍微落后 (0 ~ 1秒) -> 连续读取 (追赶)
            elif diff > 0.01: 
                ret, frame = cap.read()
                if ret:
                    self.process_and_emit(frame)
                    # 尝试更新当前位置
                    pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                    if pos_msec > 0:
                        current_pos = pos_msec / 1000.0
                else:
                    # 读不到帧(结尾)，短暂休息
                    self.msleep(50)
                    
            # 情况C: 视频跑得太快 (diff <= 0) -> 等待主界面时间追上来
            else:
                self.msleep(10) 

        cap.release()

    def process_and_emit(self, frame):
        try:
            h, w = frame.shape[:2]
            # 强制 540P 预览，保证流畅
            preview_h = 540
            if h > preview_h:
                scale = preview_h / h
                new_w = int(w * scale)
                # 使用最快的缩放算法 INTER_NEAREST
                frame = cv2.resize(frame, (new_w, preview_h), interpolation=cv2.INTER_NEAREST)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            qimg = QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
            self.frame_ready.emit(qimg)
        except Exception:
            pass

    # === 🔥 兼容性修复区域 🔥 ===
    
    # 1. 加回 seek 方法 (sync_to 的别名)
    def seek(self, t):
        self.target_time = t

    def sync_to(self, t):
        self.target_time = t

    # 2. 加回 set_playing 方法 (空函数)
    # 新版逻辑是根据时间自动追赶，不需要显式的 playing 状态，
    # 但为了防止旧版主程序报错，这里留一个空壳。
    def set_playing(self, playing):
        pass

    def stop(self):
        self.running = False
        self.wait()
    
class RecorderWorker(QThread):
    progress = pyqtSignal(int); finished = pyqtSignal(str)
    
    def __init__(self, renderer, path, transparent, mode, fps, w, h, start_t, end_t, cap_path=None, supersample=False):
        super().__init__()
        self.renderer = renderer
        self.path = path
        self.trans = transparent
        self.mode = mode
        self.fps = fps
        self.w = w
        self.h = h
        self.start_t = start_t
        self.end_t = end_t
        self.cap_path = cap_path 
        self.supersample = supersample # 🔥 新增：超采样开关
        self.running = True

    def run(self):
        duration = self.end_t - self.start_t
        if duration <= 0:
            self.finished.emit("时间范围无效")
            return
        
        total_frames = int(duration * self.fps)
        dt = 1.0 / self.fps
        t = self.start_t
        
        # 决定渲染倍率
        scale_factor = 2.0 if self.supersample else 1.0
        
        # 最终输出尺寸 (1080P)
        out_w, out_h = self.w - (self.w % 2), self.h - (self.h % 2)
        
        # 实际渲染尺寸 (如果是超采样，这里就是 4K)
        render_w = int(out_w * scale_factor)
        render_h = int(out_h * scale_factor)
        
        cap = None
        if self.cap_path:
            cap = cv2.VideoCapture(self.cap_path)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)

        try:
            if self.trans:
                writer = imageio.get_writer(self.path, fps=self.fps, codec='png', pixelformat='rgba', format='FFMPEG')
            else:
                writer = imageio.get_writer(
                    self.path, fps=self.fps, codec='libx264', pixelformat='yuv420p', 
                    quality=None, macro_block_size=None, ffmpeg_params=['-crf', '18', '-preset', 'slow'] 
                )
            
            # 创建画布：如果是超采样，这里创建的是 4K 画布
            img = QImage(render_w, render_h, QImage.Format.Format_ARGB32_Premultiplied)
            
            for i in range(total_frames):
                if not self.running: break
                
                bg_frame = None
                if cap and cap.isOpened():
                    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        ih, iw, ch = frame.shape
                        bg_frame = QImage(frame.data, iw, ih, ch * iw, QImage.Format.Format_RGB888).copy()

                p = QPainter(img)
                # 开启最高质量抗锯齿
                p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
                p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

                # 🔥 魔法在这里：如果是超采样，我们把整个坐标系放大 2 倍
                # 这样所有矢量绘图操作都会在 4K 网格上进行，精度提高 4 倍
                if scale_factor != 1.0:
                    p.scale(scale_factor, scale_factor)

                # 清理背景 (注意：p.scale 之后，fillRect 只需要填原始 w,h 就会填满被放大的画布)
                if self.trans:
                    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
                    p.fillRect(0, 0, out_w, out_h, Qt.GlobalColor.transparent)
                    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
                else:
                    p.fillRect(0, 0, out_w, out_h, Qt.GlobalColor.black)
                
                # 渲染：传入的是原始尺寸 out_w, out_h
                # 但因为上面调用了 p.scale(2.0, 2.0)，所有绘制指令会被自动放大
                self.renderer.render(p, out_w, out_h, t, self.trans, self.mode, bg_image=bg_frame)
                p.end()
                
                # 输出处理
                if self.supersample:
                    # 🔥 缩小回 1080P：使用平滑缩放算法 (SmoothTransformation)
                    # 这一步就是把 4K 压回 1080P，实现超级抗锯齿
                    final_img = img.scaled(out_w, out_h, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    rgba = qimage_to_numpy(final_img)
                else:
                    rgba = qimage_to_numpy(img)

                if self.trans: writer.append_data(rgba)
                else: writer.append_data(rgba[:, :, :3])
                
                t += dt
                if i % 10 == 0: self.progress.emit(int(i / total_frames * 100))
                
            if cap: cap.release()
            writer.close()
            self.finished.emit(f"导出成功！\n保存在: {self.path}")
            
        except Exception as e:
            if cap: cap.release()
            import traceback; traceback.print_exc()
            self.finished.emit(f"导出失败: {str(e)}")

    def stop(self): self.running = False

class RaceCanvas(QWidget):
    def __init__(self, renderer):
        super().__init__()
        self.renderer=renderer; self.t=0.0; self.mode=MODE_SPEED; self.rw=1080; self.rh=1080
        self.bg_frame = None 
        self.setStyleSheet("background: #000; border: 1px solid #444;")
        
    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        scale = min(self.width()/self.rw, self.height()/self.rh)
        dx = (self.width()-self.rw*scale)/2; dy = (self.height()-self.rh*scale)/2
        p.translate(dx, dy); p.scale(scale, scale)
        p.setClipRect(0, 0, self.rw, self.rh)
        self.renderer.render(p, self.rw, self.rh, self.t, False, self.mode, bg_image=self.bg_frame)
        p.setPen(QPen(QColor(0,100,255), 2)); p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(0,0,self.rw, self.rh)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Race Overlay - Pro Studio")
        self.resize(1300, 850)
        self.setStyleSheet(STYLESHEET)
        
        self.dm = DataManager(); self.renderer = Renderer(self.dm)
        self.timer = QTimer(); self.timer.timeout.connect(self.update_play)
        self.start_time = 0.0; self.end_time = 0.0
        self.g_smooth_val = 0.5 
        
        self.cap = None; self.cap_path = None
        self.video_duration = 0
        self.current_video_pos = -1.0
        
        # 🔥 新增：默认帧率变量
        self.video_fps = 30.0 
        
        main = QWidget(); self.setCentralWidget(main)
        layout = QHBoxLayout(main)
        
        self.canvas = RaceCanvas(self.renderer)
        layout.addWidget(self.canvas, 1)
        
        right = QWidget(); right.setFixedWidth(450)
        r_layout = QVBoxLayout(right)
        
        # 1. 数据源
        gb_data = QGroupBox("1. 数据源")
        ld = QVBoxLayout(); hb = QHBoxLayout()
        btn_load = QPushButton("📂 打开 CSV"); btn_load.clicked.connect(self.load_csv)
        self.lbl_info = QLabel("未加载")
        hb.addWidget(btn_load); hb.addWidget(self.lbl_info); ld.addLayout(hb)
        
        hb2 = QHBoxLayout()
        btn_video = QPushButton("🎬 加载背景视频"); btn_video.clicked.connect(self.load_video)
        self.lbl_video = QLabel("无视频")
        hb2.addWidget(btn_video); hb2.addWidget(self.lbl_video); ld.addLayout(hb2)
        
        self.chk_1hz = QCheckBox("1Hz 优化 (高斯平滑)"); self.chk_1hz.stateChanged.connect(lambda: self.reprocess_data())
        ld.addWidget(self.chk_1hz)
        gb_data.setLayout(ld); r_layout.addWidget(gb_data)
        
        # 2. 设置 Tabs
        self.tabs = QTabWidget(); self.tabs.currentChanged.connect(self.on_tab_change)
        
        # Studio Tab
        t_studio = QWidget(); 
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll_content = QWidget(); l_stu = QVBoxLayout(scroll_content)
        l_stu.addWidget(QLabel("🎥 演播室组件配置"))
        components = [('speed', '🚀 速度表'), ('map', '🗺️ 地图'), ('gforce', '🔴 G值球'), ('attitude', '✈️ 姿态仪')]
        for key, name in components:
            gb = QGroupBox(name); gl = QVBoxLayout()
            chk = QCheckBox("显示"); chk.setChecked(self.renderer.layout[key]['show'])
            chk.stateChanged.connect(lambda v, k=key: self.update_studio_layout(k, 'show', v))
            gl.addWidget(chk)
            gl.addLayout(self.create_slider("水平位置 X", 0, 100, int(self.renderer.layout[key]['x']*100), lambda v, k=key: self.update_studio_layout(k, 'x', v/100.0)))
            gl.addLayout(self.create_slider("垂直位置 Y", 0, 100, int(self.renderer.layout[key]['y']*100), lambda v, k=key: self.update_studio_layout(k, 'y', v/100.0)))
            gl.addLayout(self.create_slider("缩放 Scale", 10, 200, int(self.renderer.layout[key]['scale']*100), lambda v, k=key: self.update_studio_layout(k, 'scale', v/100.0)))
            gb.setLayout(gl); l_stu.addWidget(gb)
        scroll_content.setLayout(l_stu); scroll.setWidget(scroll_content)
        t_studio_layout = QVBoxLayout(); t_studio_layout.addWidget(scroll); t_studio.setLayout(t_studio_layout)
        self.tabs.addTab(t_studio, "🎥 演播室")
        
        # Other Tabs
        t_map = QWidget(); lm = QVBoxLayout(t_map)
        self.add_combo(lm, "模式", ["静态北向", "动态车头"], self.set_map_mode)
        self.add_combo(lm, "颜色", ["速度渐变", "纯白"], self.set_map_color)
        self.add_step(lm, "缩放", 0.1, 5.0, 1.0, 0.1, lambda v: setattr(self.renderer, 'map_zoom_factor', v))
        self.add_step(lm, "赛道宽度", 1, 100, 15, 1, lambda v: setattr(self.renderer, 'track_width', int(v)))
        self.add_step(lm, "车标大小", 5, 200, 30, 2, lambda v: setattr(self.renderer, 'car_size', int(v)))
        
        h_grad = QHBoxLayout(); h_grad.addWidget(QLabel("渐变范围:")); sb_gmin = QSpinBox(); sb_gmin.setRange(0, 400); sb_gmin.setValue(0); sb_gmax = QSpinBox(); sb_gmax.setRange(10, 500); sb_gmax.setValue(160)
        sb_gmin.valueChanged.connect(lambda v: setattr(self.renderer, 'grad_min', v) or self.canvas.update()); sb_gmax.valueChanged.connect(lambda v: setattr(self.renderer, 'grad_max', v) or self.canvas.update())
        h_grad.addWidget(sb_gmin); h_grad.addWidget(QLabel("-")); h_grad.addWidget(sb_gmax); lm.addLayout(h_grad)
        btn_snap = QPushButton("📸 导出透明地图"); btn_snap.clicked.connect(self.snapshot_map); lm.addWidget(btn_snap); lm.addStretch(); self.tabs.addTab(t_map, "🗺️ 地图")
        
        t_spd = QWidget(); ls = QVBoxLayout(t_spd)
        self.add_combo(ls, "样式", ["数字圆环", "物理指针"], self.set_gauge_style)
        self.add_step(ls, "表底速度", 80, 400, 260, 10, lambda v: setattr(self.renderer, 'max_speed', v))
        ls.addStretch(); self.tabs.addTab(t_spd, "🚀 速度")
        
        t_g = QWidget(); lg = QVBoxLayout(t_g)
        self.add_check(lg, "反转横向 X", lambda v: setattr(self.renderer, 'g_invert_x', v))
        self.add_check(lg, "反转纵向 Y", lambda v: setattr(self.renderer, 'g_invert_y', v))
        self.add_step(lg, "表底量程", 0.5, 5.0, 1.0, 0.1, lambda v: setattr(self.renderer, 'max_g', v))
        self.add_step(lg, "平滑系数", 0.05, 3.0, 0.5, 0.05, self.update_g_smooth)
        lg.addStretch(); self.tabs.addTab(t_g, "🔴 G值")
        
        t_att = QWidget(); la = QVBoxLayout(t_att)
        self.add_check(la, "反转翻滚", lambda v: setattr(self.renderer, 'att_invert_roll', v))
        self.add_check(la, "反转俯仰", lambda v: setattr(self.renderer, 'att_invert_pitch', v))
        la.addStretch(); self.tabs.addTab(t_att, "✈️ 姿态")
        
        r_layout.addWidget(self.tabs)
        
        # 3. 剪辑
        gb_exp = QGroupBox("3. 剪辑与导出")
        le = QVBoxLayout(); h_trim = QHBoxLayout(); h_trim.addWidget(QLabel("时间范围:"))
        self.sb_start = QDoubleSpinBox(); self.sb_start.setRange(0, 99999); self.sb_start.setSuffix("s"); self.sb_end = QDoubleSpinBox(); self.sb_end.setRange(0, 99999); self.sb_end.setSuffix("s")
        self.sb_start.valueChanged.connect(self.on_trim_change); self.sb_end.valueChanged.connect(self.on_trim_change)
        h_trim.addWidget(self.sb_start); h_trim.addWidget(QLabel("-")); h_trim.addWidget(self.sb_end); le.addLayout(h_trim)
        
        hr = QHBoxLayout(); hr.addWidget(QLabel("分辨率:"))
        self.sb_w = QSpinBox(); self.sb_w.setRange(100,4096); self.sb_w.setValue(1080); self.sb_h = QSpinBox(); self.sb_h.setRange(100,4096); self.sb_h.setValue(1080)
        self.sb_w.valueChanged.connect(self.update_res); self.sb_h.valueChanged.connect(self.update_res)
        hr.addWidget(self.sb_w); hr.addWidget(QLabel("x")); hr.addWidget(self.sb_h)
        self.btn_match = QPushButton("📏 匹配视频"); self.btn_match.setFixedWidth(80)
        self.btn_match.clicked.connect(self.match_video_res)
        hr.addWidget(self.btn_match)
        le.addLayout(hr)
        
        self.chk_supersample = QCheckBox("✨ 2x 超采样导出 (高清防锯齿)"); self.chk_supersample.setChecked(True)
        le.addWidget(self.chk_supersample)
        
        ht = QHBoxLayout(); self.rb_mov = QRadioButton("MOV(透明)"); self.rb_mov.setChecked(True); self.rb_mp4 = QRadioButton("MP4(黑底)"); bg = QButtonGroup(self); bg.addButton(self.rb_mov); bg.addButton(self.rb_mp4)
        ht.addWidget(self.rb_mov); ht.addWidget(self.rb_mp4); le.addLayout(ht)
        
        self.btn_play = QPushButton("▶ 播放选定范围"); self.btn_play.clicked.connect(self.toggle_play)
        self.slider = QSlider(Qt.Orientation.Horizontal); self.slider.sliderMoved.connect(self.seek)
        self.btn_exp = QPushButton("⏺ 渲染导出视频"); self.btn_exp.clicked.connect(self.export)
        self.pbar = QProgressBar()
        
        le.addWidget(self.btn_play); le.addWidget(self.slider); le.addWidget(self.btn_exp); le.addWidget(self.pbar)
        gb_exp.setLayout(le); r_layout.addWidget(gb_exp)
        
        r_layout.addWidget(self.tabs); layout.addWidget(right)
        self.tabs.setCurrentIndex(0); self.canvas.mode = MODE_STUDIO

    def match_video_res(self):
        if not hasattr(self, 'video_w') or self.video_w == 0:
            QMessageBox.warning(self, "提示", "请先加载视频文件")
            return
        self.sb_w.setValue(self.video_w)
        self.sb_h.setValue(self.video_h)
        QMessageBox.information(self, "成功", f"分辨率已设置为: {self.video_w} x {self.video_h}")

    def create_slider(self, name, min_v, max_v, init_v, callback):
        l = QVBoxLayout(); h = QHBoxLayout()
        h.addWidget(QLabel(name)); val_l = QLabel(str(init_v))
        h.addStretch(); h.addWidget(val_l); l.addLayout(h)
        sl = QSlider(Qt.Orientation.Horizontal); sl.setRange(min_v, max_v); sl.setValue(init_v)
        def on_change(v): val_l.setText(str(v)); callback(v); self.canvas.update()
        sl.valueChanged.connect(on_change)
        l.addWidget(sl)
        return l

    def update_studio_layout(self, key, param, value): self.renderer.layout[key][param] = value; self.canvas.update()
    
    def add_step(self, l, txt, min_v, max_v, val, step, cb):
        r = QHBoxLayout()
        val_l = QLabel(f"{val}")
        b1 = QPushButton("-"); b2 = QPushButton("+")
        r.addWidget(QLabel(txt)); r.addWidget(b1); r.addWidget(val_l); r.addWidget(b2); l.addLayout(r)
        def ch(d):
            try: current = float(val_l.text())
            except ValueError: current = val
            v = current + d; v = max(min_v, min(max_v, v))
            if step >= 1 and float(step).is_integer(): val_l.setText(f"{int(v)}")
            else: val_l.setText(f"{v:.2f}")
            cb(v); self.canvas.update()
        b1.clicked.connect(lambda: ch(-step)); b2.clicked.connect(lambda: ch(step))

    def add_combo(self, l, txt, items, cb): r=QHBoxLayout(); r.addWidget(QLabel(txt)); c=QComboBox(); c.addItems(items); c.currentIndexChanged.connect(cb); r.addWidget(c); l.addLayout(r)
    def add_check(self, l, txt, cb): c=QCheckBox(txt); c.stateChanged.connect(lambda: cb(c.isChecked()) or self.canvas.update()); l.addWidget(c)
    
    def load_csv(self):
        p, _ = QFileDialog.getOpenFileName(self, "CSV", "", "*.csv")
        if p:
            n, d = self.dm.load_csv(p); self.reprocess_data()
            self.lbl_info.setText(f"{d:.1f}s"); self.slider.setRange(0, int(d*100))
            self.sb_start.setMaximum(d); self.sb_end.setMaximum(d); self.sb_end.setValue(d)
            self.start_time = 0.0; self.end_time = d; self.canvas.update()

    # === 放在 MainWindow 类中 (请只覆盖修改的函数) ===

    def load_video(self):
        p, _ = QFileDialog.getOpenFileName(self, "Video", "", "Video (*.mp4 *.mov *.avi)")
        if p:
            self.cap_path = p
            
            # 1. 获取视频信息 & 🔥 立即读取第一帧
            temp_cap = cv2.VideoCapture(p)
            if temp_cap.isOpened():
                # 读取信息
                fps = temp_cap.get(cv2.CAP_PROP_FPS)
                self.video_fps = fps if fps > 0 else 30.0 
                frames = temp_cap.get(cv2.CAP_PROP_FRAME_COUNT)
                self.video_duration = frames / fps if fps > 0 else 0
                self.video_w = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.video_h = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.lbl_video.setText(f"{self.video_duration:.1f}s ({self.video_fps:.2f} fps)")

                # 🔥 关键修改：立即读取第一帧用于显示，解决“不知道导入对不对”的问题
                ret, frame = temp_cap.read()
                if ret:
                    # 同样进行缩放处理，防止第一帧就是 4K 导致渲染卡顿
                    h, w = frame.shape[:2]
                    preview_h = 540
                    if h > preview_h:
                        scale = preview_h / h
                        new_w = int(w * scale)
                        frame = cv2.resize(frame, (new_w, preview_h), interpolation=cv2.INTER_NEAREST)
                    
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = frame.shape
                    # 立即更新画布
                    qimg = QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
                    self.on_video_frame(qimg)

                temp_cap.release()
            
            # 2. 启动后台线程
            if hasattr(self, 'video_thread'): self.video_thread.stop()
            self.video_thread = VideoPlayerThread(p)
            self.video_thread.frame_ready.connect(self.on_video_frame)
            self.video_thread.start()
            self.video_thread.sync_to(0)
            
            if QMessageBox.question(self, "设置", "检测到视频，是否自动将画布分辨率设为视频大小？", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
                self.match_video_res()

    # 其他函数保持不变，为确保上下文完整，如果你覆盖时遇到缩进问题，
    # 只需要确保上面的 load_video 替换了原有的即可。

    def on_video_frame(self, qimg):
        self.canvas.bg_frame = qimg
        self.canvas.update()

    def update_g_smooth(self, val): self.g_smooth_val = val; self.reprocess_data()
    def reprocess_data(self): self.dm.process(10, 5, use_gaussian=self.chk_1hz.isChecked(), g_smooth_factor=self.g_smooth_val); self.canvas.update()
    
    def on_trim_change(self):
        s = self.sb_start.value(); e = self.sb_end.value()
        if s >= e: e = s + 1; self.sb_end.setValue(e)
        self.start_time = s; self.end_time = e
        self.slider.setRange(int(s*100), int(e*100)); self.canvas.t = s; 
        if hasattr(self, 'video_thread'): self.video_thread.sync_to(s)
        self.canvas.update()

    def on_tab_change(self, idx): 
        mode_map = {0: MODE_STUDIO, 1: MODE_MAP, 2: MODE_SPEED, 3: MODE_GFORCE, 4: MODE_ATTITUDE}
        self.canvas.mode = mode_map.get(idx, MODE_SPEED); self.canvas.update()

    def set_map_mode(self, i): self.renderer.map_type=i; self.canvas.update()
    def set_map_color(self, i): self.renderer.map_color_mode=i; self.canvas.update()
    def set_gauge_style(self, i): self.renderer.gauge_style=i; self.canvas.update()
    def update_res(self): self.canvas.rw = self.sb_w.value(); self.canvas.rh = self.sb_h.value(); self.canvas.update()
    def snapshot_map(self):
        if self.dm.df_proc is None: return
        p, _ = QFileDialog.getSaveFileName(self, "Save Map", "map.png", "PNG (*.png)")
        if not p: return
        w, h = 1920, 1080
        img = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied); img.fill(0)
        painter = QPainter(img); old_mode = self.renderer.map_type; self.renderer.map_type = MAP_STATIC 
        self.renderer.render_map(painter, w, h, self.dm.get_state_at_time(0), 0, True)
        self.renderer.map_type = old_mode; painter.end()
        img.save(p); QMessageBox.information(self, "成功", f"地图封面已保存至:\n{p}")

    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_play.setText("▶ 播放")
        else:
            # 🔥 修复：根据视频实际 FPS 设定定时器间隔
            # 例如 60fps -> 16ms, 30fps -> 33ms
            interval = int(1000.0 / self.video_fps)
            self.timer.start(interval)
            self.btn_play.setText("⏸ 暂停")
    
    def update_play(self):
        # 🔥 修复：时间增量也必须完全匹配视频 FPS
        dt = 1.0 / self.video_fps
        self.canvas.t += dt
        
        if self.canvas.t > self.end_time: self.canvas.t = self.start_time
        
        if hasattr(self, 'video_thread'):
            self.video_thread.sync_to(self.canvas.t)
            
        self.slider.blockSignals(True)
        self.slider.setValue(int(self.canvas.t*100))
        self.slider.blockSignals(False)
        self.canvas.update()

    def seek(self, v): 
        t = v/100.0
        self.canvas.t = t
        if hasattr(self, 'video_thread'): self.video_thread.sync_to(t)
        self.canvas.update()
    
    def export(self):
        default_ext = ".mov" if self.rb_mov.isChecked() else ".mp4"
        p, _ = QFileDialog.getSaveFileName(self, "Export Video", "race_overlay" + default_ext, f"Video (*{default_ext})")
        if not p: return
        self.btn_exp.setEnabled(False); self.btn_exp.setText("正在渲染...")
        
        self.worker = RecorderWorker(
            self.renderer, p, self.rb_mov.isChecked(), self.canvas.mode, 
            60, self.sb_w.value(), self.sb_h.value(), 
            self.start_time, self.end_time, 
            self.cap_path, 
            self.chk_supersample.isChecked()
        ) 
        self.worker.progress.connect(self.pbar.setValue)
        def on_finish(msg): self.btn_exp.setEnabled(True); self.btn_exp.setText("⏺ 渲染导出视频"); self.pbar.setValue(0); QMessageBox.information(self, "渲染结束", msg)
        self.worker.finished.connect(on_finish); self.worker.start()

if __name__ == '__main__':
    app = QApplication(sys.argv); win = MainWindow(); win.show(); sys.exit(app.exec())