import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

class DataManager:
    def __init__(self):
        self.df_raw = None
        self.df_proc = None
        self.total_duration = 0
        self.aspect_ratio = 1.0
        # 原始数据数组
        self.raw_t = np.array([])
        self.norm_x = np.array([])
        self.norm_y = np.array([])
        # 处理后数据数组 (用于快速访问)
        self.proc_t = None
        self.cached_speeds = None
        self.cached_headings = None
        self.meter_x = None
        self.meter_y = None
        # 🔥 新增数据缓存数组
        self.cached_roll = None
        self.cached_pitch = None
        self.cached_lon_g = None
        self.cached_lat_g = None
        self.cached_sats = None
        self.cached_alt = None

    def load_csv(self, file_path):
        # 🔥 更新：增加新列的读取
        usecols = ['Time', 'Lat', 'Lon', 'Alt', 'Speed_kmh', 'Sats', 'Heading', 'Roll', 'Pitch', 'Lon_G', 'Lat_G']
        self.df_raw = pd.read_csv(file_path, usecols=usecols)
        
        self.df_raw['Time'] = pd.to_datetime(self.df_raw['Time'])
        start_time = self.df_raw['Time'].iloc[0]
        self.df_raw['RelTime'] = (self.df_raw['Time'] - start_time).dt.total_seconds()
        
        self.total_duration = self.df_raw['RelTime'].iloc[-1]
        self.raw_t = self.df_raw['RelTime'].values
        
        # 归一化经纬度 (用于静态地图)
        lat = self.df_raw['Lat'].values
        lon = self.df_raw['Lon'].values
        
        # 🔥 安全检查：防止数据全为0导致除以零错误
        lat_range = lat.max() - lat.min()
        lon_range = lon.max() - lon.min()
        
        if lat_range == 0 or lon_range == 0:
             # 如果没有有效的GPS移动数据，设置默认值
             self.norm_x = np.zeros_like(lon)
             self.norm_y = np.zeros_like(lat)
             self.aspect_ratio = 1.0
             print("警告：经纬度数据无效或无移动，静态地图将不可用。")
        else:
            self.norm_y = (lat - lat.min()) / lat_range
            self.norm_x = (lon - lon.min()) / lon_range
            
            mid_lat = np.radians(lat.mean())
            lat_m = lat_range * 111320
            lon_m = lon_range * 111320 * np.cos(mid_lat)
            self.aspect_ratio = lat_m / lon_m if lon_m != 0 else 1.0

        return len(self.df_raw), self.total_duration

    def process(self, target_hz=60.0, smooth_window=5):
        if self.df_raw is None: return
        
        new_t = np.arange(0, self.total_duration, 1/target_hz)
        self.proc_t = new_t
        
        # 需要插值和存在的列
        cols_to_interp = {
            'Lat': 'lat', 'Lon': 'lon', 'Alt': 'alt', 
            'Speed_kmh': 'speed', 'Sats': 'sats', 'Heading': 'heading',
            'Roll': 'roll', 'Pitch': 'pitch', 'Lon_G': 'lon_g', 'Lat_G': 'lat_g'
        }
        
        interp_data = {}
        for col, key in cols_to_interp.items():
            if col in self.df_raw.columns:
                f = interp1d(self.raw_t, self.df_raw[col].values, kind='linear', fill_value="extrapolate")
                interp_data[key] = f(new_t)
            else:
                # 如果CSV里缺少某列，填充0
                interp_data[key] = np.zeros_like(new_t)

        self.df_proc = pd.DataFrame(interp_data)
        
        # 🔥 更新：对新数据也进行平滑处理
        # Heading 需要特殊平滑处理(角度回绕)，这里暂时简单平均，未来可优化
        cols_to_smooth = ['speed', 'alt', 'heading', 'roll', 'pitch', 'lon_g', 'lat_g']
        if smooth_window > 1:
            # 确保窗口是奇数
            window = smooth_window if smooth_window % 2 != 0 else smooth_window + 1
            for col in cols_to_smooth:
                self.df_proc[col] = self.df_proc[col].rolling(window=window, center=True, min_periods=1).mean()
        
        # 计算米制坐标 (用于动态地图)
        if 'lat' in self.df_proc and 'lon' in self.df_proc:
            lat_p = self.df_proc['lat'].values
            lon_p = self.df_proc['lon'].values
            mid_lat_rad = np.radians(lat_p.mean())
            self.meter_y = (lat_p - lat_p[0]) * 111320
            self.meter_x = (lon_p - lon_p[0]) * 111320 * np.cos(mid_lat_rad)
        else:
             self.meter_x = np.zeros_like(new_t)
             self.meter_y = np.zeros_like(new_t)

        # 缓存常用数据
        self.cached_speeds = self.df_proc['speed'].values
        self.cached_headings = self.df_proc['heading'].values
        # 🔥 缓存新数据
        self.cached_roll = self.df_proc['roll'].values
        self.cached_pitch = self.df_proc['pitch'].values
        self.cached_lon_g = self.df_proc['lon_g'].values
        self.cached_lat_g = self.df_proc['lat_g'].values
        self.cached_sats = self.df_proc['sats'].values
        self.cached_alt = self.df_proc['alt'].values
        
        self.df_proc.fillna(0, inplace=True)

    def get_state_at_time(self, t_target):
        if self.proc_t is None: return None
        idx = np.searchsorted(self.proc_t, t_target)
        if idx >= len(self.proc_t): idx = len(self.proc_t) - 1
        
        row = self.df_proc.iloc[idx]
        
        # 获取静态地图归一化坐标
        raw_idx = np.searchsorted(self.raw_t, t_target)
        if raw_idx >= len(self.raw_t): raw_idx = len(self.raw_t) - 1
        
        return {
            'speed': row['speed'],
            'heading': row['heading'],
            'sats': row['sats'],
            'alt': row['alt'],
            # 🔥 返回新数据
            'roll': row['roll'],
            'pitch': row['pitch'],
            'lon_g': row['lon_g'],
            'lat_g': row['lat_g'],
            'mx': self.meter_x[idx],
            'my': self.meter_y[idx],
            'nx': self.norm_x[raw_idx],
            'ny': self.norm_y[raw_idx]
        }