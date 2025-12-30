import numpy as np
import pandas as pd

class DataManager:
    def __init__(self):
        self.df_raw = None      
        self.df_proc = None     
        self.total_duration = 0
        self.time_arr = None
        
        self.meter_x = None; self.meter_y = None; self.headings = None
        self.norm_x = None; self.norm_y = None; self.aspect_ratio = 1.0
        self.cache = {}

    def load_csv(self, path):
        try:
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip() 
            
            if 'Time' not in df.columns: return 0, 0
            
            if 'Fix' in df.columns and df['Fix'].sum() > 0:
                pass 
            
            df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
            df = df.dropna(subset=['Time'])
            
            if len(df) == 0: return 0, 0

            df = df.sort_values('Time') 

            start_time = df['Time'].iloc[0]
            df['Elapsed'] = (df['Time'] - start_time).dt.total_seconds()
            df = df.drop_duplicates(subset=['Elapsed'], keep='first')
            
            self.df_raw = df
            self.total_duration = df['Elapsed'].max()
            return len(df), self.total_duration
        except: return 0, 0

    def apply_gaussian_smoothing(self, series, sigma):
        """带边缘填充的高斯平滑"""
        # sigma 越小越灵敏，越大越平滑
        window_size = int(6 * sigma + 1)
        if window_size % 2 == 0: window_size += 1
        radius = window_size // 2
        
        x = np.arange(-radius, radius + 1)
        # 高斯公式
        kernel = np.exp(-(x**2) / (2 * max(0.1, sigma)**2))
        kernel /= np.sum(kernel)
        
        padded_series = np.pad(series, (radius, radius), mode='edge')
        return np.convolve(padded_series, kernel, mode='valid')

    def process(self, target_hz, smooth_window, use_gaussian=False, g_smooth_factor=0.5):
        """
        g_smooth_factor: G值平滑系数 (秒)。
        例如 0.5 表示平滑掉 0.5秒内的抖动。1Hz数据建议 0.5-1.0，高频数据建议 0.1-0.2
        """
        if self.df_raw is None: return

        # 1. 频率检测
        raw_dur = self.df_raw['Elapsed'].max() - self.df_raw['Elapsed'].min()
        avg_hz = len(self.df_raw) / raw_dur if raw_dur > 0 else 10
        is_low_freq = avg_hz < 5.0
        enable_gaussian = use_gaussian or is_low_freq

        # 2. 重采样
        df_tmp = self.df_raw.set_index('Time').copy()
        df_tmp = df_tmp[~df_tmp.index.duplicated(keep='first')]
        
        interval_ms = int(1000 / target_hz) if target_hz > 0 else 100
        cols = ['Lat', 'Lon', 'Speed_kmh', 'Alt', 'Sats', 'Heading', 'Roll', 'Pitch', 'Lon_G', 'Lat_G']
        use_cols = [c for c in cols if c in df_tmp.columns]
        
        df_res = df_tmp[use_cols].resample(f'{interval_ms}ms').mean().interpolate().reset_index()
        
        if len(df_res) == 0: return
        
        st = df_res['Time'].iloc[0]
        df_res['Elapsed'] = (df_res['Time'] - st).dt.total_seconds()
        self.total_duration = df_res['Elapsed'].max()
        
        # 3. 平滑处理
        base_window = max(1, int(smooth_window))
        
        # === A. 轨迹平滑 (Lat/Lon) ===
        if 'Lat' in use_cols:
            if enable_gaussian:
                sigma = target_hz * 0.5 
                df_res['Lat'] = self.apply_gaussian_smoothing(df_res['Lat'].values, sigma)
                df_res['Lon'] = self.apply_gaussian_smoothing(df_res['Lon'].values, sigma)
            else:
                df_res['Lat'] = df_res['Lat'].rolling(base_window, center=True, min_periods=1).mean()
                df_res['Lon'] = df_res['Lon'].rolling(base_window, center=True, min_periods=1).mean()

        # === B. G值独立高斯平滑 (关键修改) ===
        # G值始终使用高斯平滑，因为它能模拟物理惯性
        # sigma = target_hz * 秒数
        g_sigma = target_hz * g_smooth_factor
        
        for c in ['Lon_G', 'Lat_G']:
            if c in use_cols:
                # 只有当用户真的给了G值数据时才平滑
                # 防止全0数据被平滑出奇怪的东西(虽然通常不会)
                df_res[c] = self.apply_gaussian_smoothing(df_res[c].values, g_sigma)

        # === C. 其他常规数据平滑 ===
        for c in use_cols:
            if c in ['Lat', 'Lon', 'Lon_G', 'Lat_G']: continue # 跳过已处理的
            
            w = max(1, int(base_window/2)) if c in ['Roll','Pitch'] else base_window
            if enable_gaussian and c == 'Speed_kmh': w = max(w, int(target_hz * 1.0))
            df_res[c] = df_res[c].rolling(window=w, min_periods=1, center=True).mean()

        # 4. 投影 & Heading
        if 'Lat' in use_cols:
            mx = df_res['Lat'].mean(); my = df_res['Lon'].mean(); R=6371000
            x = (df_res['Lon']-my)*(np.pi/180)*R*np.cos(mx*np.pi/180)
            y = (df_res['Lat']-mx)*(np.pi/180)*R
            self.meter_x = x.values; self.meter_y = y.values
            
            rx = x.max()-x.min(); ry = y.max()-y.min()
            if rx < 1.0: rx = 1.0; 
            if ry < 1.0: ry = 1.0
            self.norm_x = ((x-x.min())/rx).values
            self.norm_y = ((y-y.min())/ry).values
            self.aspect_ratio = ry/rx
            
            # Heading
            dx = np.gradient(self.meter_x); dy = np.gradient(self.meter_y)
            spds = df_res['Speed_kmh'].fillna(0).values
            wts = np.square(spds) 
            wts[spds < 2.0] = 0.0001 
            
            vec_win = int(target_hz * 2.0) if enable_gaussian else max(5, base_window * 2)
            w_dx = pd.Series(dx * wts).rolling(vec_win, center=True, min_periods=1).mean().values
            w_dy = pd.Series(dy * wts).rolling(vec_win, center=True, min_periods=1).mean().values
            
            vec_mag = np.sqrt(w_dx**2 + w_dy**2)
            raw_headings = np.degrees(np.arctan2(w_dy, w_dx))
            
            valid_mask = vec_mag > 1e-4
            clean_headings = pd.Series(np.where(valid_mask, raw_headings, np.nan))
            clean_headings = clean_headings.ffill().fillna(0).values
            
            self.headings = np.degrees(np.unwrap(np.radians(clean_headings)))
            
        else:
            N = len(df_res)
            self.meter_x=np.zeros(N); self.meter_y=np.zeros(N); self.headings=np.zeros(N)
            self.norm_x=np.zeros(N); self.norm_y=np.zeros(N)

        self.time_arr = df_res['Elapsed'].values
        for c in use_cols: self.cache[c] = df_res[c].fillna(0).values
        self.df_proc = df_res

    def get_state_at_time(self, t):
        if self.time_arr is None: return None
        idx = np.searchsorted(self.time_arr, t); idx = np.clip(idx, 0, len(self.time_arr)-1)
        
        if idx == 0: return self._pack_state(0, 0, 0)
        t0 = self.time_arr[idx-1]; t1 = self.time_arr[idx]
        alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0
        
        return self._pack_state(idx-1, idx, alpha)

    def _pack_state(self, i0, i1, alpha):
        def lerp(arr, i0, i1, a):
            return arr[i0] * (1-a) + arr[i1] * a
            
        return {
            'mx': lerp(self.meter_x, i0, i1, alpha),
            'my': lerp(self.meter_y, i0, i1, alpha),
            'nx': lerp(self.norm_x, i0, i1, alpha),
            'ny': lerp(self.norm_y, i0, i1, alpha),
            'heading': lerp(self.headings, i0, i1, alpha), 
            'speed': lerp(self.cache.get('Speed_kmh', np.zeros_like(self.time_arr)), i0, i1, alpha),
            'sats': self.cache.get('Sats', np.zeros_like(self.time_arr))[i1],
            'alt': lerp(self.cache.get('Alt', np.zeros_like(self.time_arr)), i0, i1, alpha),
            'roll': lerp(self.cache.get('Roll', np.zeros_like(self.time_arr)), i0, i1, alpha),
            'pitch': lerp(self.cache.get('Pitch', np.zeros_like(self.time_arr)), i0, i1, alpha),
            'lon_g': lerp(self.cache.get('Lon_G', np.zeros_like(self.time_arr)), i0, i1, alpha),
            'lat_g': lerp(self.cache.get('Lat_G', np.zeros_like(self.time_arr)), i0, i1, alpha),
        }