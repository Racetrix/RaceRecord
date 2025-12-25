import numpy as np
import pandas as pd

class DataManager:
    def __init__(self):
        self.df_raw = None      
        self.df_proc = None     
        self.total_duration = 0
        self.time_arr = None
        
        self.meter_x = None; self.meter_y = None; self.headings = None
        self.cached_speeds = None; self.cached_sats = None; self.cached_alt = None
        self.norm_x = None; self.norm_y = None; self.aspect_ratio = 1.0

    def load_csv(self, path):
        df = pd.read_csv(path)
        if 'Fix' in df.columns: df = df[df['Fix'] == 1]
        df['Time'] = pd.to_datetime(df['Time'])
        start_time = df['Time'].iloc[0]
        df['Elapsed'] = (df['Time'] - start_time).dt.total_seconds()
        
        self.df_raw = df
        self.total_duration = df['Elapsed'].max()
        return len(df), self.total_duration

    def process(self, target_hz, smooth_window):
        if self.df_raw is None: return

        # 1. 重采样 (Resample)
        df_tmp = self.df_raw.set_index('Time').copy()
        if target_hz <= 0: target_hz = 1.0
        interval_ms = int(1000 / target_hz)
        rule = f'{interval_ms}ms'
        
        df_resampled = df_tmp.resample(rule).mean().interpolate()
        df_resampled = df_resampled.reset_index()
        
        if len(df_resampled) > 0:
            start_time = df_resampled['Time'].iloc[0]
            df_resampled['Elapsed'] = (df_resampled['Time'] - start_time).dt.total_seconds()
        
        # 2. 基础数据平滑 (位置/速度)
        window = max(1, int(smooth_window))
        cols = ['Lat', 'Lon', 'Speed_kmh']
        if 'Alt' in df_resampled.columns: cols.append('Alt')
        smoothed = df_resampled[cols].rolling(window=window, min_periods=1, center=True).mean()
        df_resampled[cols] = smoothed
        
        self.df_proc = df_resampled
        self.time_arr = df_resampled['Elapsed'].values

        # 3. 投影 (Lat/Lon -> Meters)
        mid_lat = df_resampled['Lat'].mean()
        mid_lon = df_resampled['Lon'].mean()
        R = 6371000
        x = (df_resampled['Lon'] - mid_lon) * (np.pi/180) * R * np.cos(mid_lat * np.pi/180)
        y = (df_resampled['Lat'] - mid_lat) * (np.pi/180) * R
        
        self.meter_x = x.values
        self.meter_y = y.values
        
        min_x, max_x = x.min(), x.max()
        min_y, max_y = y.min(), y.max()
        range_x = max_x - min_x or 1
        range_y = max_y - min_y or 1
        self.norm_x = ((x - min_x) / range_x).values
        self.norm_y = ((y - min_y) / range_y).values
        self.aspect_ratio = range_y / range_x

        # 4. 🔥🔥🔥 核心升级：速度加权矢量平滑 (Speed-Weighted Vector Smoothing) 🔥🔥🔥
        
        # (1) 计算基础位移向量
        dx = np.gradient(self.meter_x)
        dy = np.gradient(self.meter_y)
        
        # (2) 获取速度权重 (Speed Weighting)
        # 速度越快，权重越大；速度越慢，权重越小 (平方处理以增强对比)
        speeds = df_resampled['Speed_kmh'].values
        # 加上一个小数值防止除以0，并做平方处理增强高信噪比区域的权重
        weights = np.square(speeds + 1.0) 
        
        # (3) 加权向量
        w_dx = dx * weights
        w_dy = dy * weights
        
        # (4) 对加权向量进行大窗口平滑
        # 这里的窗口要大，因为它不仅平滑噪点，还负责连接入弯和出弯的趋势
        vec_window = max(5, window * 3) 
        
        smooth_w_dx = pd.Series(w_dx).rolling(vec_window, min_periods=1, center=True).mean().values
        smooth_w_dy = pd.Series(w_dy).rolling(vec_window, min_periods=1, center=True).mean().values
        
        # (5) 计算最终角度
        # 在弯心速度极低时，smooth_w_dx/dy 依然会保留入弯时的大权重惯性，
        # 直到出弯加速后的新权重占据主导，从而实现完美过渡。
        raw_angles = np.degrees(np.arctan2(smooth_w_dy, smooth_w_dx))
        self.headings = np.degrees(np.unwrap(np.radians(raw_angles)))

        # 缓存
        self.cached_speeds = speeds
        self.cached_sats = df_resampled['Sats'].values if 'Sats' in df_resampled else np.zeros(len(df_resampled))
        self.cached_alt = df_resampled['Alt'].values if 'Alt' in df_resampled else np.zeros(len(df_resampled))

    def get_state_at_time(self, t):
        if self.time_arr is None or len(self.time_arr) == 0: return None
        
        idx = np.searchsorted(self.time_arr, t)
        max_idx = len(self.time_arr) - 1
        
        if idx > max_idx: idx = max_idx; return self._pack_state(idx, idx, 0)
        if idx == 0: return self._pack_state(0, 0, 0)
            
        t0 = self.time_arr[idx-1]
        t1 = self.time_arr[idx]
        alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0
        
        return self._pack_state(idx-1, idx, alpha)

    def _pack_state(self, i0, i1, alpha):
        N = len(self.meter_x)
        i0 = min(i0, N-1); i1 = min(i1, N-1)
        
        mx = self.meter_x[i0] * (1-alpha) + self.meter_x[i1] * alpha
        my = self.meter_y[i0] * (1-alpha) + self.meter_y[i1] * alpha
        
        nx = self.norm_x[i0] * (1-alpha) + self.norm_x[i1] * alpha
        ny = self.norm_y[i0] * (1-alpha) + self.norm_y[i1] * alpha
        
        spd = self.cached_speeds[i0] * (1-alpha) + self.cached_speeds[i1] * alpha
        
        # 简单的线性插值即可，因为数据已经极其平滑
        heading = self.headings[i0] * (1-alpha) + self.headings[i1] * alpha
        
        sats = self.cached_sats[i1]
        alt = self.cached_alt[i0] * (1-alpha) + self.cached_alt[i1] * alpha
        
        return {
            'mx': mx, 'my': my, 'nx': nx, 'ny': ny,
            'speed': spd, 'heading': heading,
            'sats': sats, 'alt': alt
        }