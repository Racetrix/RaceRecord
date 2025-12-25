import sys
import os
import numpy as np
import imageio
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QComboBox, QGroupBox, 
    QFrame, QMessageBox, QProgressBar, QRadioButton, QButtonGroup, 
    QStackedWidget, QCheckBox, QSlider, QScrollArea, QSizePolicy
)
from PyQt6.QtGui import QPainter, QImage
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal

from race_data import DataManager
from race_render import Renderer, qimage_to_numpy, MODE_PATH, MODE_GAUGE, STYLE_DIGITAL, STYLE_NEEDLE, MAP_STATIC_NORTH, MAP_DYNAMIC_HEAD, COLOR_SPEED, COLOR_WHITE, COLOR_RED, COLOR_CYAN, RESOLUTION

STYLESHEET = """
QMainWindow { background-color: #181818; }
QWidget { font-family: 'Segoe UI', sans-serif; font-size: 14px; color: #E0E0E0; }
QLabel#title { font-size: 20px; font-weight: bold; color: #4EC9B0; letter-spacing: 1px; margin-bottom: 10px; }
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
    progress = pyqtSignal(int); finished = pyqtSignal(str)
    def __init__(self, renderer, output_path, transparent, render_mode, fps):
        super().__init__()
        self.renderer = renderer; self.path = output_path
        self.transparent = transparent; self.render_mode = render_mode
        self.fps = fps; self.is_running = True
    def run(self):
        w, h = RESOLUTION
        if w%2!=0: w-=1; 
        if h%2!=0: h-=1
        duration = self.renderer.dm.total_duration
        total_frames = int(duration * self.fps)
        dt = 1.0/self.fps; current_t = 0.0
        
        if self.transparent: 
            writer = imageio.get_writer(self.path, fps=self.fps, codec='png', pixelformat='rgba', format='FFMPEG', macro_block_size=1)
        else: 
            writer = imageio.get_writer(self.path, fps=self.fps, codec='libx264', pixelformat='yuv420p', quality=8, macro_block_size=1)
        
        image = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
        try:
            for i in range(total_frames):
                if not self.is_running: break
                painter = QPainter(image)
                try:
                    self.renderer.render(painter, w, h, current_t, self.transparent, self.render_mode)
                finally:
                    painter.end()
                rgba = qimage_to_numpy(image)
                if self.transparent: writer.append_data(rgba)
                else: writer.append_data(rgba[:,:,:3])
                current_t += dt
                if i%10==0: self.progress.emit(int((i/total_frames)*100))
        except Exception as e: print(f"Rec Error: {e}")
        finally: writer.close(); self.finished.emit(self.path)
    def stop(self): self.is_running = False

class RaceCanvas(QWidget):
    def __init__(self, renderer):
        super().__init__()
        self.renderer = renderer; self.current_time = 0.0; self.render_mode = MODE_PATH
    def set_mode(self, mode): self.render_mode = mode; self.update()
    def paintEvent(self, event):
        painter = QPainter(self)
        self.renderer.render(painter, self.width(), self.height(), self.current_time, False, self.render_mode)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RaceRecord v41.0 - 封面生成版")
        self.resize(1350, 900)
        self.setStyleSheet(STYLESHEET)
        
        self.data_manager = DataManager()
        self.renderer = Renderer(self.data_manager)
        self.target_hz = 5.0
        self.smooth_window = 5
        self.recorder = None
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
        
        btn_minus = QPushButton("-")
        btn_minus.setProperty("class", "step_btn")
        btn_plus = QPushButton("+")
        btn_plus.setProperty("class", "step_btn")
        
        state = {'val': current_val}

        def update_display():
            txt = f"{state['val']:.1f}" if is_float else f"{state['val']}"
            lbl_val.setText(txt)
            callback(state['val'])

        def change(delta):
            new_v = state['val'] + delta
            if new_v < min_val: new_v = min_val
            if new_v > max_val: new_v = max_val
            state['val'] = new_v
            update_display()

        btn_minus.clicked.connect(lambda: change(-step))
        btn_plus.clicked.connect(lambda: change(step))
        
        row.addWidget(lbl_title)
        row.addStretch()
        row.addWidget(btn_minus)
        row.addWidget(lbl_val)
        row.addWidget(btn_plus)
        layout.addLayout(row)
        return state 

    def init_ui(self):
        main_widget = QWidget(); layout = QHBoxLayout(); layout.setContentsMargins(0,0,0,0)
        self.canvas = RaceCanvas(self.renderer)
        
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
        
        # === Panel 1: 路径设置 ===
        p1 = QWidget(); l1 = QVBoxLayout(); l1.setContentsMargins(0,0,0,0); l1.setSpacing(10)
        gb1 = QGroupBox("路径参数")
        gl1 = QVBoxLayout()
        gl1.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        self.combo_map_style = QComboBox()
        self.combo_map_style.addItem("🌐 静态 (北向)", MAP_STATIC_NORTH)
        self.combo_map_style.addItem("🦅 动态 (车头向)", MAP_DYNAMIC_HEAD)
        # 🔥 已移除: 道路预瞄选项
        self.combo_map_style.currentIndexChanged.connect(self.update_settings)
        gl1.addWidget(QLabel("地图视角:"))
        gl1.addWidget(self.combo_map_style)
        
        # 🔥 新增：呼吸缩放开关 (保留这个好功能)
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
        
        gb1.setLayout(gl1); l1.addWidget(gb1)
        l1.addStretch() 
        p1.setLayout(l1); self.stack_settings.addWidget(p1)

        # === Panel 2: 仪表设置 ===
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
        
        self.combo_style = QComboBox(); self.combo_style.addItem("🔮 科技圆环", STYLE_DIGITAL); self.combo_style.addItem("🏎️ 物理指针", STYLE_NEEDLE)
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
        
        gb2.setLayout(gl2); l2.addWidget(gb2)
        l2.addStretch()
        p2.setLayout(l2); self.stack_settings.addWidget(p2)
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
        
        gb_export = QGroupBox("导出")
        el = QVBoxLayout()
        hl_opt = QHBoxLayout()
        self.rb_trans = QRadioButton("透明"); self.rb_black = QRadioButton("黑底"); self.rb_trans.setChecked(True)
        bgg = QButtonGroup(self); bgg.addButton(self.rb_trans); bgg.addButton(self.rb_black)
        
        self.combo_fps = QComboBox(); self.combo_fps.addItems(["30 FPS", "60 FPS"]); self.combo_fps.setCurrentIndex(1)
        hl_opt.addWidget(self.rb_trans); hl_opt.addWidget(self.rb_black); hl_opt.addStretch(); hl_opt.addWidget(self.combo_fps)
        
        # 🔥 新增：生成封面图按钮
        self.btn_snap = QPushButton("📸 生成路径封面图")
        self.btn_snap.setObjectName("btn_snap")
        self.btn_snap.clicked.connect(self.generate_cover_image)
        el.addWidget(self.btn_snap)

        self.btn_record = QPushButton("⏺ 开始渲染视频"); self.btn_record.setObjectName("btn_record")
        self.btn_record.clicked.connect(self.toggle_record)
        self.progress = QProgressBar(); self.progress.setValue(0)
        
        el.addLayout(hl_opt); el.addWidget(self.btn_record); el.addWidget(self.progress)
        gb_export.setLayout(el); ctrl_panel.addWidget(gb_export)

        ctrl_panel.addStretch()
        scroll.setWidget(ctrl_content)
        
        layout.addWidget(self.canvas, 1); layout.addWidget(scroll)
        main_widget.setLayout(layout); self.setCentralWidget(main_widget)

    # 🔥🔥🔥 核心：生成封面图的逻辑 🔥🔥🔥
    def generate_cover_image(self):
        if self.data_manager.total_duration == 0:
             QMessageBox.warning(self, "警告", "请先加载 CSV 数据文件")
             return

        path, _ = QFileDialog.getSaveFileName(self, "保存封面图", "track_cover.png", "PNG Image (*.png)")
        if not path: return

        # 1. 记住当前设置
        old_style = self.renderer.map_style
        old_gauge = self.renderer.show_gauge
        old_extra = self.renderer.show_extra

        # 2. 强制切换到适合封面的模式 (静态全景，无UI)
        self.renderer.map_style = MAP_STATIC_NORTH
        self.renderer.show_gauge = False
        self.renderer.show_extra = False

        # 3. 创建画布并渲染第0帧
        w, h = RESOLUTION
        if w%2!=0: w-=1; 
        if h%2!=0: h-=1
        
        image = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(Qt.GlobalColor.transparent) # 确保背景透明
        painter = QPainter(image)
        try:
            # 渲染 t=0 时刻，透明背景，仅路径模式
            self.renderer.render(painter, w, h, 0.0, True, MODE_PATH)
        finally:
            painter.end()

        # 4. 保存
        success = image.save(path)

        # 5. 恢复之前的设置
        self.renderer.map_style = old_style
        self.renderer.show_gauge = old_gauge
        self.renderer.show_extra = old_extra
        # 同步UI状态
        self.chk_show_gauge.setChecked(old_gauge)
        self.chk_show_extra.setChecked(old_extra)

        if success:
            QMessageBox.information(self, "成功", f"封面图已成功保存至:\n{path}")
        else:
            QMessageBox.critical(self, "失败", "保存图像文件失败")

    def set_hz(self, val):
        self.target_hz = val
        self.reprocess_data()
    
    def set_smooth(self, val):
        self.smooth_window = int(val)
        self.reprocess_data()

    def reprocess_data(self):
        self.data_manager.process(self.target_hz, self.smooth_window)
        self.canvas.update()

    def switch_mode(self, index):
        self.canvas.set_mode(index)
        self.stack_settings.setCurrentIndex(index)

    def update_settings(self):
        self.renderer.map_style = self.combo_map_style.currentData()
        self.renderer.path_color_mode = self.combo_path_color.currentData()
        self.renderer.show_gauge = self.chk_show_gauge.isChecked()
        self.renderer.gauge_style = self.combo_style.currentData()
        self.renderer.show_extra = self.chk_show_extra.isChecked()
        self.renderer.show_time = self.chk_time.isChecked()
        self.renderer.show_sats = self.chk_sats.isChecked()
        self.renderer.show_alt = self.chk_alt.isChecked()
        self.canvas.update()

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "CSV", "", "CSV (*.csv)")
        if not path: return
        try:
            count, dur = self.data_manager.load_csv(path)
            self.lbl_info.setText(f"{os.path.basename(path)} | {dur:.1f}s")
            self.reprocess_data()
            self.slider.setRange(0, int(dur*10))
            self.canvas.update()
        except Exception as e: QMessageBox.critical(self, "Error", str(e))

    def toggle_play(self):
        if self.playback_timer.isActive(): self.playback_timer.stop(); self.btn_play.setText("▶ 播放")
        else: self.playback_timer.start(16); self.btn_play.setText("⏸ 暂停")

    def update_playback(self):
        self.canvas.current_time += 0.016
        if self.canvas.current_time > self.data_manager.total_duration:
            self.canvas.current_time = 0; self.playback_timer.stop(); self.btn_play.setText("▶ 播放")
        self.slider.setValue(int(self.canvas.current_time*10)); self.canvas.update()

    def seek(self, val): self.canvas.current_time = val/10.0; self.canvas.update()

    def toggle_record(self):
        if self.recorder and self.recorder.isRunning(): 
            self.recorder.stop()
            self.btn_record.setText("⏺ 开始渲染视频")
            self.btn_record.setObjectName("btn_record") 
            self.btn_record.setStyle(self.btn_record.style())
            return
            
        is_transparent = self.rb_trans.isChecked()
        mode = self.combo_mode.currentIndex()
        fps = 60 if self.combo_fps.currentIndex() == 1 else 30
        ext = "mov" if is_transparent else "mp4"
        path, _ = QFileDialog.getSaveFileName(self, "Save", f"race_out.{ext}", f"Video (*.{ext})")
        if not path: return
        
        self.btn_record.setText("⏹ 停止渲染")
        self.btn_record.setObjectName("btn_stop") 
        self.btn_record.setStyle(self.btn_record.style())
        
        self.progress.setValue(0); self.btn_load.setEnabled(False)
        self.recorder = RecorderWorker(self.renderer, path, is_transparent, mode, fps)
        self.recorder.progress.connect(self.progress.setValue)
        self.recorder.finished.connect(self.on_fin)
        self.recorder.start()

    def on_fin(self, path):
        self.btn_record.setText("⏺ 开始渲染视频")
        self.btn_record.setObjectName("btn_record")
        self.btn_record.setStyle(self.btn_record.style())
        self.btn_load.setEnabled(True); self.progress.setValue(100)
        QMessageBox.information(self, "完成", f"Saved:\n{path}")

if __name__ == '__main__':
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())