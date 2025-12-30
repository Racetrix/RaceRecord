import sys, os
import imageio
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
    MODE_MAP, MODE_SPEED, MODE_GFORCE, MODE_ATTITUDE,
    MAP_STATIC, MAP_DYNAMIC, STYLE_DIGITAL, STYLE_NEEDLE, COLOR_SPEED, COLOR_WHITE
)

STYLESHEET = """
QMainWindow { background-color: #181818; }
QWidget { font-family: 'Microsoft YaHei', sans-serif; font-size: 14px; color: #EEE; }
QTabWidget::pane { border: 1px solid #444; background: #202020; }
QTabBar::tab { background: #333; color: #AAA; padding: 8px 15px; border-top-left-radius: 4px; border-top-right-radius: 4px; margin-right: 2px; }
QTabBar::tab:selected { background: #4EC9B0; color: #000; font-weight: bold; }
QGroupBox { border: 1px solid #444; border-radius: 4px; margin-top: 10px; padding-top: 10px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #4EC9B0; top: 0px; }
QPushButton { background: #333; border: 1px solid #555; padding: 6px; border-radius: 4px; }
QPushButton:hover { background: #444; border-color: #4EC9B0; }
"""

class RecorderWorker(QThread):
    progress = pyqtSignal(int); finished = pyqtSignal(str)
    def __init__(self, renderer, path, transparent, mode, fps, w, h, start_t, end_t):
        super().__init__()
        self.renderer=renderer; self.path=path; self.trans=transparent
        self.mode=mode; self.fps=fps; self.w=w; self.h=h
        self.start_t=start_t; self.end_t=end_t; self.running=True
    def run(self):
        duration = self.end_t - self.start_t
        if duration <= 0: self.finished.emit("时间范围无效"); return
        
        total_frames = int(duration * self.fps)
        dt = 1.0/self.fps
        t = self.start_t
        
        try:
            w, h = self.w - (self.w%2), self.h - (self.h%2)
            if self.trans: writer = imageio.get_writer(self.path, fps=self.fps, codec='png', pixelformat='rgba', format='FFMPEG')
            else: writer = imageio.get_writer(self.path, fps=self.fps, codec='libx264', pixelformat='yuv420p', quality=8)
            
            img = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
            for i in range(total_frames):
                if not self.running: break
                
                p = QPainter(img)
                if self.trans:
                    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
                    p.fillRect(0,0,w,h,Qt.GlobalColor.transparent)
                    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
                else: p.fillRect(0,0,w,h,Qt.GlobalColor.black)
                
                self.renderer.render(p, w, h, t, self.trans, self.mode)
                p.end()
                
                rgba = qimage_to_numpy(img)
                writer.append_data(rgba if self.trans else rgba[:,:,:3])
                
                t += dt
                if i%20==0: self.progress.emit(int(i/total_frames*100))
                
            writer.close(); self.finished.emit(self.path)
        except Exception as e: self.finished.emit(str(e))
    def stop(self): self.running=False

class RaceCanvas(QWidget):
    def __init__(self, renderer):
        super().__init__()
        self.renderer=renderer; self.t=0.0; self.mode=MODE_SPEED; self.rw=1080; self.rh=1080
        self.setStyleSheet("background: #000; border: 1px solid #444;")
    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        scale = min(self.width()/self.rw, self.height()/self.rh)
        dx = (self.width()-self.rw*scale)/2; dy = (self.height()-self.rh*scale)/2
        p.translate(dx, dy); p.scale(scale, scale)
        p.setPen(QPen(QColor(0,100,255), 2)); p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(0,0,self.rw, self.rh)
        self.renderer.render(p, self.rw, self.rh, self.t, False, self.mode)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Race Overlay - Pro Studio")
        self.resize(1250, 850)
        self.setStyleSheet(STYLESHEET)
        
        self.dm = DataManager(); self.renderer = Renderer(self.dm)
        self.timer = QTimer(); self.timer.timeout.connect(self.update_play)
        self.start_time = 0.0; self.end_time = 0.0
        
        # G值平滑系数变量
        self.g_smooth_val = 0.5 
        
        main = QWidget(); self.setCentralWidget(main)
        layout = QHBoxLayout(main)
        
        self.canvas = RaceCanvas(self.renderer)
        layout.addWidget(self.canvas, 1)
        
        right = QWidget(); right.setFixedWidth(420)
        r_layout = QVBoxLayout(right)
        
        # 1. 数据
        gb_data = QGroupBox("1. 数据源")
        ld = QVBoxLayout(); hb = QHBoxLayout()
        btn_load = QPushButton("📂 打开 CSV"); btn_load.clicked.connect(self.load_csv)
        self.lbl_info = QLabel("未加载")
        hb.addWidget(btn_load); hb.addWidget(self.lbl_info)
        ld.addLayout(hb)
        self.chk_1hz = QCheckBox("1Hz 优化 (高斯平滑)"); self.chk_1hz.stateChanged.connect(lambda: self.reprocess_data())
        ld.addWidget(self.chk_1hz)
        gb_data.setLayout(ld); r_layout.addWidget(gb_data)
        
        # 2. 设置
        self.tabs = QTabWidget(); self.tabs.currentChanged.connect(self.on_tab_change)
        
        # Map Tab
        t_map = QWidget(); lm = QVBoxLayout(t_map)
        self.add_combo(lm, "模式", ["静态北向", "动态车头"], self.set_map_mode)
        self.add_combo(lm, "颜色", ["速度渐变", "纯白"], self.set_map_color)
        self.add_step(lm, "缩放", 0.1, 5.0, 1.0, 0.1, lambda v: setattr(self.renderer, 'map_zoom_factor', v))
        
        h_grad = QHBoxLayout(); h_grad.addWidget(QLabel("渐变速度范围:"))
        sb_gmin = QSpinBox(); sb_gmin.setRange(0, 400); sb_gmin.setValue(0)
        sb_gmax = QSpinBox(); sb_gmax.setRange(10, 500); sb_gmax.setValue(160)
        sb_gmin.valueChanged.connect(lambda v: setattr(self.renderer, 'grad_min', v) or self.canvas.update())
        sb_gmax.valueChanged.connect(lambda v: setattr(self.renderer, 'grad_max', v) or self.canvas.update())
        h_grad.addWidget(sb_gmin); h_grad.addWidget(QLabel("-")); h_grad.addWidget(sb_gmax)
        lm.addLayout(h_grad)
        
        btn_snap = QPushButton("📸 导出透明地图 (封面)"); btn_snap.clicked.connect(self.snapshot_map)
        lm.addWidget(btn_snap)
        lm.addStretch(); self.tabs.addTab(t_map, "🗺️ 地图")
        
        # Speed Tab
        t_spd = QWidget(); ls = QVBoxLayout(t_spd)
        self.add_combo(ls, "样式", ["数字圆环", "物理指针"], self.set_gauge_style)
        self.add_step(ls, "表底速度", 80, 400, 260, 10, lambda v: setattr(self.renderer, 'max_speed', v))
        self.add_step(ls, "整体缩放", 0.1, 3.0, 1.0, 0.1, lambda v: setattr(self.renderer, 'gauge_scale', v))
        ls.addStretch(); self.tabs.addTab(t_spd, "🚀 速度")
        
        # G Tab
        t_g = QWidget(); lg = QVBoxLayout(t_g)
        self.add_check(lg, "反转横向 X", lambda v: setattr(self.renderer, 'g_invert_x', v))
        self.add_check(lg, "反转纵向 Y", lambda v: setattr(self.renderer, 'g_invert_y', v))
        self.add_step(lg, "整体缩放", 0.1, 3.0, 0.5, 0.1, lambda v: setattr(self.renderer, 'g_scale', v))
        self.add_step(lg, "表底量程 (Max G)", 0.5, 5.0, 1.0, 0.1, lambda v: setattr(self.renderer, 'max_g', v))
        
        # 🔥 新增：G值平滑滑块 (触发重新计算数据)
        lg.addSpacing(10)
        self.add_step(lg, "G值平滑 (秒)", 0.05, 3.0, 0.5, 0.05, self.update_g_smooth)
        
        lg.addStretch(); self.tabs.addTab(t_g, "🔴 G值")
        
        # Att Tab
        t_att = QWidget(); la = QVBoxLayout(t_att)
        self.add_check(la, "反转翻滚", lambda v: setattr(self.renderer, 'att_invert_roll', v))
        self.add_check(la, "反转俯仰", lambda v: setattr(self.renderer, 'att_invert_pitch', v))
        self.add_step(la, "整体缩放", 0.1, 3.0, 1.0, 0.1, lambda v: setattr(self.renderer, 'att_scale', v))
        la.addStretch(); self.tabs.addTab(t_att, "✈️ 姿态")
        
        r_layout.addWidget(self.tabs)
        
        # 3. 导出
        gb_exp = QGroupBox("3. 导出与剪辑")
        le = QVBoxLayout()
        
        h_trim = QHBoxLayout()
        h_trim.addWidget(QLabel("时间范围:"))
        self.sb_start = QDoubleSpinBox(); self.sb_start.setRange(0, 99999); self.sb_start.setSuffix("s")
        self.sb_end = QDoubleSpinBox(); self.sb_end.setRange(0, 99999); self.sb_end.setSuffix("s")
        self.sb_start.valueChanged.connect(self.on_trim_change)
        self.sb_end.valueChanged.connect(self.on_trim_change)
        h_trim.addWidget(self.sb_start); h_trim.addWidget(QLabel("-")); h_trim.addWidget(self.sb_end)
        le.addLayout(h_trim)
        
        hr = QHBoxLayout(); hr.addWidget(QLabel("分辨率:"))
        self.sb_w = QSpinBox(); self.sb_w.setRange(100,4096); self.sb_w.setValue(1080)
        self.sb_h = QSpinBox(); self.sb_h.setRange(100,4096); self.sb_h.setValue(1080)
        self.sb_w.valueChanged.connect(self.update_res); self.sb_h.valueChanged.connect(self.update_res)
        hr.addWidget(self.sb_w); hr.addWidget(QLabel("x")); hr.addWidget(self.sb_h)
        le.addLayout(hr)
        
        ht = QHBoxLayout()
        self.rb_mov = QRadioButton("透明MOV"); self.rb_mov.setChecked(True)
        self.rb_mp4 = QRadioButton("黑底MP4")
        bg = QButtonGroup(self); bg.addButton(self.rb_mov); bg.addButton(self.rb_mp4)
        ht.addWidget(self.rb_mov); ht.addWidget(self.rb_mp4)
        le.addLayout(ht)
        
        self.btn_play = QPushButton("▶ 播放选定范围"); self.btn_play.clicked.connect(self.toggle_play)
        self.slider = QSlider(Qt.Orientation.Horizontal); self.slider.sliderMoved.connect(self.seek)
        self.btn_exp = QPushButton("⏺ 渲染导出视频"); self.btn_exp.clicked.connect(self.export)
        self.pbar = QProgressBar()
        
        le.addWidget(self.btn_play); le.addWidget(self.slider)
        le.addWidget(self.btn_exp); le.addWidget(self.pbar)
        gb_exp.setLayout(le); r_layout.addWidget(gb_exp)
        
        r_layout.addWidget(self.tabs); layout.addWidget(right)
        self.tabs.setCurrentIndex(1); self.canvas.mode = MODE_SPEED

    def add_combo(self, l, txt, items, cb):
        r=QHBoxLayout(); r.addWidget(QLabel(txt)); c=QComboBox(); c.addItems(items); c.currentIndexChanged.connect(cb); r.addWidget(c); l.addLayout(r)
    def add_check(self, l, txt, cb):
        c=QCheckBox(txt); c.stateChanged.connect(lambda: cb(c.isChecked()) or self.canvas.update()); l.addWidget(c)
    def add_step(self, l, txt, min_v, max_v, val, step, cb):
        r=QHBoxLayout(); val_l=QLabel(f"{val}")
        def ch(d): v=float(val_l.text())+d; v=max(min_v,min(max_v,v)); val_l.setText(f"{v:.2f}" if step<1 else f"{int(v)}"); cb(v) # 注意不直接刷新，cb里处理
        b1=QPushButton("-"); b1.clicked.connect(lambda:ch(-step)); b2=QPushButton("+"); b2.clicked.connect(lambda:ch(step))
        r.addWidget(QLabel(txt)); r.addWidget(b1); r.addWidget(val_l); r.addWidget(b2); l.addLayout(r)

    def load_csv(self):
        p, _ = QFileDialog.getOpenFileName(self, "CSV", "", "*.csv")
        if p:
            n, d = self.dm.load_csv(p); self.reprocess_data()
            self.lbl_info.setText(f"{d:.1f}s"); self.slider.setRange(0, int(d*100))
            self.sb_start.setMaximum(d); self.sb_end.setMaximum(d); self.sb_end.setValue(d)
            self.start_time = 0.0; self.end_time = d
            self.canvas.update()

    def update_g_smooth(self, val):
        self.g_smooth_val = val
        self.reprocess_data()

    def reprocess_data(self):
        # 传递 1hz开关 和 G值平滑参数
        self.dm.process(10, 5, use_gaussian=self.chk_1hz.isChecked(), g_smooth_factor=self.g_smooth_val)
        self.canvas.update()

    def snapshot_map(self):
        if self.dm.df_proc is None: return
        p, _ = QFileDialog.getSaveFileName(self, "Save Map", "map.png", "PNG (*.png)")
        if not p: return
        w, h = 1920, 1080
        img = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied); img.fill(0)
        painter = QPainter(img)
        old_mode = self.renderer.map_type
        self.renderer.map_type = MAP_STATIC 
        self.renderer.render_map(painter, w, h, self.dm.get_state_at_time(0), 0, True)
        self.renderer.map_type = old_mode 
        painter.end()
        img.save(p)
        QMessageBox.information(self, "成功", f"地图封面已保存至:\n{p}")

    def on_trim_change(self):
        s = self.sb_start.value(); e = self.sb_end.value()
        if s >= e: e = s + 1; self.sb_end.setValue(e)
        self.start_time = s; self.end_time = e
        self.slider.setRange(int(s*100), int(e*100))
        self.canvas.t = s; self.canvas.update()

    def on_tab_change(self, idx): self.canvas.mode = idx; self.canvas.update()
    def set_map_mode(self, i): self.renderer.map_type=i; self.canvas.update()
    def set_map_color(self, i): self.renderer.map_color_mode=i; self.canvas.update()
    def set_gauge_style(self, i): self.renderer.gauge_style=i; self.canvas.update()
    def update_res(self): self.canvas.rw = self.sb_w.value(); self.canvas.rh = self.sb_h.value(); self.canvas.update()
    
    def toggle_play(self):
        if self.timer.isActive(): self.timer.stop(); self.btn_play.setText("▶ 播放")
        else: self.timer.start(16); self.btn_play.setText("⏸ 暂停")
    
    def update_play(self):
        self.canvas.t += 0.016
        if self.canvas.t > self.end_time: self.canvas.t = self.start_time
        self.slider.setValue(int(self.canvas.t*100)); self.canvas.update()

    def seek(self, v): self.canvas.t = v/100.0; self.canvas.update()
    
    def export(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save", "out.mov", "Video (*.mov)")
        if not p: return
        self.worker = RecorderWorker(self.renderer, p, self.rb_mov.isChecked(), self.canvas.mode, 60, 
                                     self.canvas.rw, self.canvas.rh, 
                                     self.start_time, self.end_time) 
        self.worker.progress.connect(self.pbar.setValue)
        self.worker.finished.connect(lambda x: QMessageBox.information(self,"Done",x))
        self.worker.start()

if __name__ == '__main__':
    app = QApplication(sys.argv); win = MainWindow(); win.show(); sys.exit(app.exec())