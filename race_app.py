import sys
import os
import math
import numpy as np
import imageio
# 确保安装了 imageio-ffmpeg: pip install imageio-ffmpeg

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QComboBox, QGroupBox, 
    QFrame, QMessageBox, QProgressBar, QRadioButton, QButtonGroup, 
    QStackedWidget, QCheckBox, QSlider, QScrollArea, QSizePolicy
)
from PyQt6.QtGui import QPainter, QImage, QColor, QFont, QPen, QLinearGradient, QBrush, QPainterPath
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QPointF, QRectF

from race_data import DataManager
from race_render import Renderer, qimage_to_numpy, MODE_PATH, MODE_GAUGE, STYLE_DIGITAL, STYLE_NEEDLE, MAP_STATIC_NORTH, MAP_DYNAMIC_HEAD, COLOR_SPEED, COLOR_WHITE, COLOR_RED, COLOR_CYAN, RESOLUTION
from race_intro import IntroRenderer, INTRO_NAMES, INTRO_CLASSIC
from race_gauges import GAUGE_NAMES, STYLE_LINEAR

STYLESHEET = """
QMainWindow { background-color: #181818; }
QWidget { font-family: 'Segoe UI', sans-serif; font-size: 14px; color: #E0E0E0; }
QLabel#title { font-size: 20px; font-weight: bold; color: #4EC9B0; letter-spacing: 1px; margin-bottom: 10px; }
QLabel#help_text { font-size: 11px; color: #888; font-style: italic; margin-bottom: 5px; }
QGroupBox { border: 1px solid #333; border-radius: 6px; margin-top: 15px; padding-top: 15px; padding-bottom: 5px; background-color: #202020; font-weight: bold; color: #888; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #4EC9B0; top: 0px; }
QPushButton { background-color: #2D2D2D; border: 1px solid #3E3E3E; border-radius: 4px; padding: 6px 12px; color: #EEE; font-weight: bold; }
QPushButton:hover { background-color: #383838; border-color: #4EC9B0; color: #FFF; }
QPushButton:pressed { background-color: #4EC9B0; color: #000; }
QPushButton:disabled { background-color: #222; border-color: #2A2A2A; color: #555; }
QPushButton#btn_record { background-color: #B71C1C; border-color: #D32F2F; color: white; padding: 10px; font-size: 15px; }
QPushButton#btn_record:hover { background-color: #D32F2F; border-color: #FF5252; }
QPushButton#btn_stop { background-color: #333; border-color: #555; color: #AAA; }
QPushButton#btn_snap { background-color: #007ACC; border-color: #0099FF; }
QPushButton#btn_snap:hover { background-color: #0099FF; }
QPushButton.step_btn { min-width: 25px; max-width: 25px; padding: 4px; background-color: #252526; }
QComboBox { background-color: #252526; border: 1px solid #3E3E3E; border-radius: 4px; padding: 5px; color: #FFF; font-weight: bold; }
QComboBox::drop-down { border: none; }
QComboBox:hover { border-color: #4EC9B0; }
QProgressBar { border: 1px solid #333; border-radius: 4px; text-align: center; background-color: #111; color: #FFF; }
QProgressBar::chunk { background-color: #4EC9B0; }
QSlider::groove:horizontal { border: 1px solid #333; height: 6px; background: #222; border-radius: 3px; }
QSlider::handle:horizontal { background: #4EC9B0; width: 16px; margin: -5px 0; border-radius: 8px; }
QCheckBox { spacing: 8px; color: #BBB; }
QCheckBox::indicator { width: 16px; height: 16px; border-radius: 3px; border: 1px solid #555; background: #222; }
QCheckBox::indicator:checked { background: #4EC9B0; border-color: #4EC9B0; image: none; }
QScrollArea { border: none; background: transparent; }
"""

class RecorderWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, renderer, intro_renderer, output_path, transparent, render_mode, fps, show_intro, intro_style, logo_size):
        super().__init__()
        self.renderer = renderer
        self.intro_renderer = intro_renderer
        self.path = output_path
        self.transparent = transparent
        self.render_mode = render_mode
        self.fps = fps
        self.show_intro = show_intro
        self.intro_style = intro_style
        self.logo_size = logo_size
        self.is_running = True

    def run(self):
        try:
            w, h = RESOLUTION
            # 修复：分行写
            if w % 2 != 0: w -= 1
            if h % 2 != 0: h -= 1
            
            data_duration = self.renderer.dm.total_duration
            if data_duration <= 0: raise ValueError("没有有效数据")

            data_frames = int(data_duration * self.fps)
            intro_duration = self.intro_renderer.duration if self.show_intro else 0
            intro_frames = int(intro_duration * self.fps)
            total_frames = intro_frames + data_frames
            
            if self.transparent: 
                writer = imageio.get_writer(self.path, fps=self.fps, codec='png', pixelformat='rgba', format='FFMPEG', macro_block_size=1)
            else: 
                writer = imageio.get_writer(self.path, fps=self.fps, codec='libx264', pixelformat='yuv420p', bitrate='12000k', macro_block_size=1)
            
            image = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
            dt = 1.0/self.fps
            
            for i in range(total_frames):
                if not self.is_running: break
                
                current_total_time = i * dt
                
                image.fill(Qt.GlobalColor.transparent if self.transparent else Qt.GlobalColor.black)
                painter = QPainter(image)
                
                try:
                    if self.show_intro and current_total_time < intro_duration:
                        self.intro_renderer.render(painter, w, h, current_total_time, self.intro_style, self.logo_size)
                    else:
                        data_time = current_total_time - intro_duration
                        self.renderer.render(painter, w, h, data_time, self.transparent, self.render_mode)
                finally:
                    painter.end()
                
                rgba = qimage_to_numpy(image)
                if self.transparent: writer.append_data(rgba)
                else: writer.append_data(rgba[:,:,:3])
                
                if i % 10 == 0: 
                    prog = int((i / total_frames) * 100)
                    self.progress.emit(prog)

            writer.close()
            self.finished.emit(self.path)

        except Exception as e:
            self.error_occurred.emit(str(e))

    def stop(self): self.is_running = False

class RaceCanvas(QWidget):
    def __init__(self, renderer, intro_renderer):
        super().__init__()
        self.renderer = renderer
        self.intro_renderer = intro_renderer
        self.current_app_time = 0.0 
        self.render_mode = MODE_PATH
        self.preview_show_intro = True
        self.preview_intro_style = INTRO_CLASSIC
        self.preview_logo_size = 140

    def set_mode(self, mode): self.render_mode = mode; self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, Qt.GlobalColor.black) 
        
        intro_dur = self.intro_renderer.duration if self.preview_show_intro else 0
        
        if self.preview_show_intro and self.current_app_time < intro_dur:
            self.intro_renderer.render(painter, w, h, self.current_app_time, self.preview_intro_style, self.preview_logo_size)
        else:
            data_time = self.current_app_time - intro_dur
            if data_time > self.renderer.dm.total_duration: data_time = self.renderer.dm.total_duration
            self.renderer.render(painter, w, h, data_time, False, self.render_mode)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Racetrix v43.2 - 语法终极修复版")
        self.resize(1350, 900)
        self.setStyleSheet(STYLESHEET)
        
        self.data_manager = DataManager()
        self.renderer = Renderer(self.data_manager)
        self.intro_renderer = IntroRenderer()
        
        self.target_hz = 5.0
        self.smooth_window = 5
        self.recorder = None
        self.logo_size_state = {'val': 140}
        
        self.playback_timer = QTimer(); self.playback_timer.timeout.connect(self.update_playback)
        self.init_ui()

    def add_stepper(self, layout, label_text, min_val, max_val, current_val, step, callback, is_float=False):
        row = QHBoxLayout()
        row.setContentsMargins(0,0,0,0)
        lbl_title = QLabel(label_text)
        lbl_val = QLabel(f"{current_val}")
        lbl_val.setFixedWidth(50)
        lbl_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_val.setStyleSheet("color: #4EC9B0; font-weight: bold; background: #222; border-radius: 3px;")
        btn_minus = QPushButton("-"); btn_minus.setProperty("class", "step_btn")
        btn_plus = QPushButton("+"); btn_plus.setProperty("class", "step_btn")
        state = {'val': current_val}
        def update_display():
            txt = f"{state['val']:.1f}" if is_float else f"{state['val']}"
            lbl_val.setText(txt)
            if callback: callback(state['val'])
        def change(delta):
            new_v = state['val'] + delta
            if new_v < min_val: new_v = min_val
            if new_v > max_val: new_v = max_val
            state['val'] = new_v
            update_display()
        btn_minus.clicked.connect(lambda: change(-step))
        btn_plus.clicked.connect(lambda: change(step))
        row.addWidget(lbl_title); row.addStretch(); row.addWidget(btn_minus); row.addWidget(lbl_val); row.addWidget(btn_plus)
        layout.addLayout(row)
        return state 

    def init_ui(self):
        main_widget = QWidget(); layout = QHBoxLayout(); layout.setContentsMargins(0,0,0,0)
        self.canvas = RaceCanvas(self.renderer, self.intro_renderer)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(400)
        
        ctrl_content = QWidget()
        ctrl_panel = QVBoxLayout(ctrl_content)
        ctrl_content.setStyleSheet("background-color: #1E1E1E;")
        ctrl_panel.setSpacing(12)
        
        title = QLabel("RACE CONTROLLER", objectName="title")
        ctrl_panel.addWidget(title)

        gb_data = QGroupBox("数据源")
        l_data = QVBoxLayout()
        hl_file = QHBoxLayout()
        self.btn_load = QPushButton("📂 加载 CSV"); self.btn_load.clicked.connect(self.load_csv)
        self.lbl_info = QLabel("未加载")
        self.lbl_info.setStyleSheet("color: #666; font-size: 12px;")
        hl_file.addWidget(self.btn_load); hl_file.addWidget(self.lbl_info)
        l_data.addLayout(hl_file)
        self.add_stepper(l_data, "重采样 (Hz)", 0.5, 50.0, 5.0, 0.5, self.set_hz, is_float=True)
        self.add_stepper(l_data, "平滑窗口", 1, 50, 5, 1, self.set_smooth)
        gb_data.setLayout(l_data); ctrl_panel.addWidget(gb_data)

        hl_mode = QHBoxLayout()
        hl_mode.addWidget(QLabel("渲染模式:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItem("🗺️ 仅路径模式", MODE_PATH)
        self.combo_mode.addItem("📟 仅仪表模式", MODE_GAUGE)
        self.combo_mode.currentIndexChanged.connect(self.switch_mode)
        hl_mode.addWidget(self.combo_mode)
        ctrl_panel.addLayout(hl_mode)

        self.stack_settings = QStackedWidget()
        
        # Panel 1
        p1 = QWidget(); l1 = QVBoxLayout(); l1.setContentsMargins(0,0,0,0); l1.setSpacing(10)
        gb1 = QGroupBox("路径参数")
        gl1 = QVBoxLayout(); gl1.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.combo_map_style = QComboBox()
        self.combo_map_style.addItem("🌐 静态 (北向)", MAP_STATIC_NORTH)
        self.combo_map_style.addItem("🦅 动态 (车头向)", MAP_DYNAMIC_HEAD)
        self.combo_map_style.currentIndexChanged.connect(self.update_settings)
        gl1.addWidget(QLabel("地图视角:"))
        gl1.addWidget(self.combo_map_style)
        self.chk_zoom = QCheckBox("动感呼吸缩放 (Breathing Zoom)")
        self.chk_zoom.setChecked(True)
        self.chk_zoom.stateChanged.connect(lambda: setattr(self.renderer, 'enable_dynamic_zoom', self.chk_zoom.isChecked()) or self.canvas.update())
        gl1.addWidget(self.chk_zoom)
        self.combo_path_color = QComboBox()
        self.combo_path_color.addItem("🌈 速度渐变", COLOR_SPEED)
        self.combo_path_color.addItem("⚪ 纯白", COLOR_WHITE)
        self.combo_path_color.addItem("🔴 纯红", COLOR_RED)
        self.combo_path_color.addItem("🔵 赛博青", COLOR_CYAN)
        self.combo_path_color.currentIndexChanged.connect(self.update_settings)
        gl1.addWidget(QLabel("轨迹配色:"))
        gl1.addWidget(self.combo_path_color)
        self.add_stepper(gl1, "渐变起始 (Min)", 0, 300, 0, 10, lambda v: setattr(self.renderer, 'grad_min', v) or self.canvas.update())
        self.add_stepper(gl1, "渐变结束 (Max)", 10, 400, 160, 10, lambda v: setattr(self.renderer, 'grad_max', v) or self.canvas.update())
        self.add_stepper(gl1, "路径粗细", 1, 50, 15, 1, lambda v: setattr(self.renderer, 'track_width', v) or self.canvas.update())
        self.add_stepper(gl1, "车标大小", 5, 100, 30, 2, lambda v: setattr(self.renderer, 'car_size', v) or self.canvas.update())
        gb1.setLayout(gl1); l1.addWidget(gb1); l1.addStretch(); p1.setLayout(l1); self.stack_settings.addWidget(p1)

        # Panel 2
        p2 = QWidget(); l2 = QVBoxLayout(); l2.setContentsMargins(0,0,0,0); l2.setSpacing(10)
        gb_layout = QGroupBox("布局调节")
        gl_layout = QVBoxLayout()
        self.add_stepper(gl_layout, "整体缩放", 0.2, 3.0, 1.0, 0.1, lambda v: setattr(self.renderer, 'gauge_scale', v) or self.canvas.update(), True)
        self.add_stepper(gl_layout, "位置 X", -900, 900, 0, 20, lambda v: setattr(self.renderer, 'gauge_offset_x', v) or self.canvas.update())
        self.add_stepper(gl_layout, "位置 Y", -900, 900, 0, 20, lambda v: setattr(self.renderer, 'gauge_offset_y', v) or self.canvas.update())
        self.add_stepper(gl_layout, "刻度粗细", 0.5, 5.0, 1.0, 0.5, lambda v: setattr(self.renderer, 'tick_width_scale', v) or self.canvas.update(), True)
        gb_layout.setLayout(gl_layout); l2.addWidget(gb_layout)
        gb2 = QGroupBox("显示参数")
        gl2 = QVBoxLayout()
        self.chk_show_gauge = QCheckBox("显示速度表"); self.chk_show_gauge.setChecked(True); self.chk_show_gauge.stateChanged.connect(self.update_settings)
        gl2.addWidget(self.chk_show_gauge)
        
        self.combo_style = QComboBox()
        for k, v in GAUGE_NAMES.items(): self.combo_style.addItem(v, k)
        self.combo_style.setCurrentIndex(STYLE_LINEAR) 
        self.combo_style.currentIndexChanged.connect(self.update_settings)
        gl2.addWidget(self.combo_style)
        
        self.add_stepper(gl2, "表底速度", 60, 400, 200, 20, lambda v: setattr(self.renderer, 'max_speed', v) or self.canvas.update())
        self.chk_show_extra = QCheckBox("显示额外信息栏"); self.chk_show_extra.setChecked(False); self.chk_show_extra.stateChanged.connect(self.update_settings)
        gl2.addWidget(self.chk_show_extra)
        sub_info = QHBoxLayout()
        self.chk_time = QCheckBox("时间"); self.chk_time.setChecked(True); self.chk_time.stateChanged.connect(self.update_settings)
        self.chk_sats = QCheckBox("卫星"); self.chk_sats.setChecked(True); self.chk_sats.stateChanged.connect(self.update_settings)
        self.chk_alt = QCheckBox("高度"); self.chk_alt.setChecked(False); self.chk_alt.stateChanged.connect(self.update_settings)
        sub_info.addWidget(self.chk_time); sub_info.addWidget(self.chk_sats); sub_info.addWidget(self.chk_alt)
        gl2.addLayout(sub_info)
        gb2.setLayout(gl2); l2.addWidget(gb2); l2.addStretch(); p2.setLayout(l2); self.stack_settings.addWidget(p2)
        
        ctrl_panel.addWidget(self.stack_settings)

        # 播放 & 录制
        gb_ctrl = QGroupBox("时间轴")
        cl = QVBoxLayout()
        hl_play = QHBoxLayout()
        self.btn_play = QPushButton("▶ 播放"); self.btn_play.setFixedWidth(80)
        self.btn_play.clicked.connect(self.toggle_play)
        self.slider = QSlider(Qt.Orientation.Horizontal); self.slider.sliderMoved.connect(self.seek)
        hl_play.addWidget(self.btn_play); hl_play.addWidget(self.slider)
        gb_ctrl.setLayout(hl_play); ctrl_panel.addWidget(gb_ctrl)
        
        gb_export = QGroupBox("导出与片头")
        el = QVBoxLayout()
        hl_opt = QHBoxLayout()
        self.rb_trans = QRadioButton("透明"); self.rb_black = QRadioButton("黑底"); self.rb_trans.setChecked(True)
        bgg = QButtonGroup(self); bgg.addButton(self.rb_trans); bgg.addButton(self.rb_black)
        self.combo_fps = QComboBox(); self.combo_fps.addItems(["30 FPS", "60 FPS"]); self.combo_fps.setCurrentIndex(1)
        hl_opt.addWidget(self.rb_trans); hl_opt.addWidget(self.rb_black); hl_opt.addStretch(); hl_opt.addWidget(self.combo_fps)
        el.addLayout(hl_opt)
        
        self.chk_show_intro = QCheckBox("显示 Racetrix 片头 (Intro)")
        self.chk_show_intro.setChecked(True)
        self.chk_show_intro.stateChanged.connect(self.update_settings)
        el.addWidget(self.chk_show_intro)
        
        hl_intro = QHBoxLayout()
        self.combo_intro_style = QComboBox()
        for k, v in INTRO_NAMES.items(): self.combo_intro_style.addItem(v, k)
        self.combo_intro_style.currentIndexChanged.connect(self.update_settings)
        hl_intro.addWidget(QLabel("样式:"))
        hl_intro.addWidget(self.combo_intro_style)
        el.addLayout(hl_intro)
        
        self.logo_size_state = self.add_stepper(el, "Logo 字号", 50, 300, 140, 10, lambda v: self.update_settings())
        el.addWidget(QLabel("💡 保持开启可帮助 Racetrix 提升知名度，感谢支持！", objectName="help_text"))
        
        self.btn_snap = QPushButton("📸 生成路径封面图"); self.btn_snap.setObjectName("btn_snap"); self.btn_snap.clicked.connect(self.generate_cover_image)
        el.addWidget(self.btn_snap)
        self.btn_record = QPushButton("⏺ 开始渲染视频"); self.btn_record.setObjectName("btn_record"); self.btn_record.clicked.connect(self.toggle_record)
        self.progress = QProgressBar(); self.progress.setValue(0)
        el.addWidget(self.btn_record); el.addWidget(self.progress)
        gb_export.setLayout(el); ctrl_panel.addWidget(gb_export)

        ctrl_panel.addStretch()
        scroll.setWidget(ctrl_content)
        layout.addWidget(self.canvas, 1); layout.addWidget(scroll)
        main_widget.setLayout(layout); self.setCentralWidget(main_widget)

    def update_settings(self):
        self.renderer.map_style = self.combo_map_style.currentData()
        self.renderer.path_color_mode = self.combo_path_color.currentData()
        self.renderer.show_gauge = self.chk_show_gauge.isChecked()
        self.renderer.gauge_style = self.combo_style.currentData()
        self.renderer.show_extra = self.chk_show_extra.isChecked()
        self.renderer.show_time = self.chk_time.isChecked()
        self.renderer.show_sats = self.chk_sats.isChecked()
        self.renderer.show_alt = self.chk_alt.isChecked()
        
        self.canvas.preview_show_intro = self.chk_show_intro.isChecked()
        self.canvas.preview_intro_style = self.combo_intro_style.currentData()
        self.canvas.preview_logo_size = self.logo_size_state['val']
        
        intro_dur = self.intro_renderer.duration if self.canvas.preview_show_intro else 0
        total_time = self.data_manager.total_duration + intro_dur
        if total_time > 0: self.slider.setRange(0, int(total_time * 10))
        self.canvas.update()

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "CSV", "", "CSV (*.csv)")
        if not path: return
        try:
            count, dur = self.data_manager.load_csv(path)
            self.lbl_info.setText(f"{os.path.basename(path)} | {dur:.1f}s")
            self.reprocess_data()
            self.update_settings()
        except Exception as e: QMessageBox.critical(self, "Error", str(e))

    def set_hz(self, val): self.target_hz = val; self.reprocess_data()
    def set_smooth(self, val): self.smooth_window = int(val); self.reprocess_data()
    def reprocess_data(self): 
        self.data_manager.process(self.target_hz, self.smooth_window)
        self.update_settings()

    def switch_mode(self, index):
        self.canvas.set_mode(index)
        self.stack_settings.setCurrentIndex(index)

    def toggle_play(self):
        if self.playback_timer.isActive(): self.playback_timer.stop(); self.btn_play.setText("▶ 播放")
        else: self.playback_timer.start(16); self.btn_play.setText("⏸ 暂停")

    def update_playback(self):
        intro_dur = self.intro_renderer.duration if self.canvas.preview_show_intro else 0
        total_time = self.data_manager.total_duration + intro_dur
        self.canvas.current_app_time += 0.016
        if self.canvas.current_app_time > total_time:
            self.canvas.current_app_time = 0
            self.playback_timer.stop()
            self.btn_play.setText("▶ 播放")
        self.slider.setValue(int(self.canvas.current_app_time * 10))
        self.canvas.update()

    def seek(self, val): 
        self.canvas.current_app_time = val / 10.0
        self.canvas.update()

    def generate_cover_image(self):
        if self.data_manager.total_duration == 0:
             QMessageBox.warning(self, "警告", "请先加载 CSV 数据文件")
             return
        path, _ = QFileDialog.getSaveFileName(self, "保存封面图", "track_cover.png", "PNG Image (*.png)")
        if not path: return
        old_style = self.renderer.map_style
        old_gauge = self.renderer.show_gauge
        old_extra = self.renderer.show_extra
        self.renderer.map_style = MAP_STATIC_NORTH
        self.renderer.show_gauge = False
        self.renderer.show_extra = False
        w, h = RESOLUTION
        # 🔥 修复：分开写，符合 Python 语法
        if w % 2 != 0: w -= 1
        if h % 2 != 0: h -= 1
        image = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(Qt.GlobalColor.transparent) 
        painter = QPainter(image)
        try:
            self.renderer.render(painter, w, h, 0.0, True, MODE_PATH)
        finally:
            painter.end()
        image.save(path)
        self.renderer.map_style = old_style
        self.renderer.show_gauge = old_gauge
        self.renderer.show_extra = old_extra
        self.chk_show_gauge.setChecked(old_gauge)
        self.chk_show_extra.setChecked(old_extra)
        QMessageBox.information(self, "成功", f"封面图已保存")

    def toggle_record(self):
        if self.recorder and self.recorder.isRunning(): 
            self.recorder.stop()
            self.btn_record.setText("⏺ 开始渲染视频"); self.btn_record.setObjectName("btn_record"); self.btn_record.setStyle(self.btn_record.style())
            return
        if self.data_manager.total_duration <= 0:
            QMessageBox.warning(self, "无法开始", "请先加载 CSV 数据文件")
            return
        is_transparent = self.rb_trans.isChecked()
        mode = self.combo_mode.currentIndex()
        fps = 60 if self.combo_fps.currentIndex() == 1 else 30
        ext = "mov" if is_transparent else "mp4"
        path, _ = QFileDialog.getSaveFileName(self, "Save", f"race_out.{ext}", f"Video (*.{ext})")
        if not path: return
        self.btn_record.setText("⏹ 停止渲染"); self.btn_record.setObjectName("btn_stop"); self.btn_record.setStyle(self.btn_record.style())
        self.progress.setValue(0); self.btn_load.setEnabled(False)
        show_intro = self.chk_show_intro.isChecked()
        intro_style = self.combo_intro_style.currentData()
        logo_size = self.logo_size_state['val']
        self.recorder = RecorderWorker(self.renderer, self.intro_renderer, path, is_transparent, mode, fps, show_intro, intro_style, logo_size)
        self.recorder.progress.connect(self.progress.setValue)
        self.recorder.finished.connect(self.on_fin)
        self.recorder.error_occurred.connect(self.on_error)
        self.recorder.start()

    def on_fin(self, path):
        self.btn_record.setText("⏺ 开始渲染视频"); self.btn_record.setObjectName("btn_record"); self.btn_record.setStyle(self.btn_record.style())
        self.btn_load.setEnabled(True); self.progress.setValue(100)
        QMessageBox.information(self, "完成", f"Saved:\n{path}")

    def on_error(self, err_msg):
        self.btn_record.setText("⏺ 开始渲染视频"); self.btn_record.setObjectName("btn_record"); self.btn_record.setStyle(self.btn_record.style())
        self.btn_load.setEnabled(True)
        QMessageBox.critical(self, "渲染错误", f"Error:\n{err_msg}")

if __name__ == '__main__':
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())