"""
Xử lý extinction correction và cleaning cơ bản
Đã sửa để tương thích với cả 'flux' và 'Flux' column names
"""
import numpy as np
import pandas as pd
import extinction
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
from config import EFF_WAVELENGTHS, R_V
import warnings
warnings.filterwarnings('ignore')


class RobustPreprocessor:
    """Xử lý extinction và tính toán vật lý cơ bản - ROBUST VERSION"""
    
    def __init__(self):
        self.eff_wavelengths = EFF_WAVELENGTHS
        self.R_V = R_V
        
    def _get_column_name(self, df, possible_names):
        """Tìm column name trong các options"""
        for name in possible_names:
            if name in df.columns:
                return name
        raise KeyError(f"Không tìm thấy column nào trong {possible_names}")
        
    def correct_extinction(self, df):
        """
        Hiệu chỉnh extinction cho flux dùng Fitzpatrick (1999)
        Hỗ trợ cả 'flux' và 'Flux' column names
        """
        df = df.copy()
        
        if 'flux_corrected' in df.columns:
            print(" flux_corrected đã tồn tại, bỏ qua")
            return df
        
        print(" Đang hiệu chỉnh extinction...")
        
        # Xác định column names
        flux_col = self._get_column_name(df, ['flux', 'Flux', 'FLUX'])
        flux_err_col = self._get_column_name(df, ['flux_err', 'Flux_err', 'FLUX_ERR'])
        
        # Kiểm tra required columns
        required_cols = [flux_col, 'Filter', 'EBV']
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"Thiếu column bắt buộc: {col}")
        
        # Tạo cột mới
        flux_corrected_list = []
        
        for _, row in df.iterrows():
            try:
                # Lấy hiệu chỉnh extinction
                A_v = self.R_V * row['EBV']
                wave = np.array([self.eff_wavelengths[row['Filter']]])
                A_lambda = extinction.fitzpatrick99(wave, A_v, self.R_V)[0]
                
                # Hiệu chỉnh flux
                flux_corr = row[flux_col] * 10**(0.4 * A_lambda)
                flux_corrected_list.append(flux_corr)
            except Exception as e:
                # Nếu có lỗi, dùng flux gốc
                print(f"  Lỗi extinction correction: {e}, dùng flux gốc")
                flux_corrected_list.append(row[flux_col])
        
        df['flux_corrected'] = flux_corrected_list
        
        # Xử lý flux quá nhỏ/âm
        df['flux_positive'] = df['flux_corrected'].clip(lower=1e-9)
        
        print(f" Đã hiệu chỉnh extinction cho {len(df)} observations")
        return df
    
    def calculate_absolute_magnitude(self, df):
        """
        Tính độ sáng tuyệt đối từ redshift
        """
        df = df.copy()
        
        if 'absolute_mag' in df.columns:
            print("  absolute_mag đã tồn tại, bỏ qua")
            return df
        
        print(" Đang tính độ sáng tuyệt đối...")
        
        # Kiểm tra required columns
        required_cols = ['flux_positive', 'Z', 'object_id']
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"Thiếu column bắt buộc: {col}")
        
        # Tính khoảng cách từ redshift (xử lý redshift <= 0)
        df['Z_clean'] = df['Z'].copy()
        df.loc[df['Z_clean'] <= 0, 'Z_clean'] = 1e-6
        
        try:
            # Tính luminosity distance (pc)
            dist_pc = cosmo.luminosity_distance(df['Z_clean'].values).to(u.pc).value
            df['distance_pc'] = dist_pc
            
            # Tính độ sáng biểu kiến
            df['apparent_mag'] = -2.5 * np.log10(df['flux_positive'])
            
            # Tính độ sáng tuyệt đối: M = m - 5*log10(d/10)
            df['absolute_mag'] = df['apparent_mag'] - 5 * (np.log10(df['distance_pc']) - 1)
            
            # Tính SNR
            flux_err_col = self._get_column_name(df, ['flux_err', 'Flux_err', 'FLUX_ERR'])
            df['snr'] = df['flux_corrected'] / (df[flux_err_col] + 1e-9)
            
        except Exception as e:
            print(f"  Lỗi tính absolute magnitude: {e}")
            # Set default values
            df['distance_pc'] = 1e6
            df['apparent_mag'] = 0
            df['absolute_mag'] = 0
            df['snr'] = 1
        
        # Xóa cột tạm
        if 'Z_clean' in df.columns:
            df = df.drop(columns=['Z_clean'])
        
        print(f" Đã tính absolute magnitude cho {len(df)} observations")
        return df
    
    def clean_data(self, df):
        """
        Làm sạch dữ liệu cơ bản
        """
        df = df.copy()
        
        print(" Đang làm sạch dữ liệu...")
        
        # 1. Sắp xếp theo object_id và thời gian
        sort_cols = []
        if 'object_id' in df.columns:
            sort_cols.append('object_id')
        if 'mjd' in df.columns:
            sort_cols.append('mjd')
        
        if sort_cols:
            df = df.sort_values(sort_cols)
        
        # 2. Kiểm tra và xử lý missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            print(f" Có missing values trong: {missing_cols[:5]}...")  # Hiển thị 5 cột đầu
            
            # Fill missing EBV với 0 (không extinction)
            if 'EBV' in df.columns:
                df['EBV'] = df['EBV'].fillna(0)
            
            # Fill missing Filter (rất hiếm)
            if 'Filter' in df.columns:
                df['Filter'] = df['Filter'].fillna('r')  # mặc định filter r
            
            # Fill missing flux_err với median
            flux_err_col = self._get_column_name(df, ['flux_err', 'Flux_err', 'FLUX_ERR'])
            if flux_err_col in df.columns:
                median_flux_err = df[flux_err_col].median()
                df[flux_err_col] = df[flux_err_col].fillna(median_flux_err)
        
        # 3. Loại bỏ flux_err = 0 (không hợp lệ)
        flux_err_col = self._get_column_name(df, ['flux_err', 'Flux_err', 'FLUX_ERR'])
        if flux_err_col in df.columns:
            zero_err_mask = df[flux_err_col] == 0
            if zero_err_mask.any():
                print(f"  Loại bỏ {zero_err_mask.sum()} observations có flux_err = 0")
                df = df[~zero_err_mask].copy()
        
        print(f" Đã làm sạch: {len(df)} observations")
        return df
    
    def add_basic_stats(self, df):
        """Thêm thống kê cơ bản cho mỗi object"""
        df = df.copy()
        
        # Kiểm tra required columns
        if 'object_id' not in df.columns:
            print(" Không có object_id, bỏ qua basic stats")
            return df
        
        try:
            # Tính toán theo object_id
            grouped = df.groupby('object_id')
            
            # Số observations mỗi object
            obs_counts = grouped.size()
            df['obs_count_total'] = df['object_id'].map(obs_counts)
            
            # Số filter khác nhau
            if 'Filter' in df.columns:
                filter_counts = grouped['Filter'].nunique()
                df['filter_count'] = df['object_id'].map(filter_counts)
            
            # Thời gian quan sát span
            if 'mjd' in df.columns:
                time_spans = grouped['mjd'].apply(lambda x: x.max() - x.min())
                df['mjd_span'] = df['object_id'].map(time_spans)
            
        except Exception as e:
            print(f"  Lỗi tính basic stats: {e}")
        
        return df
    
    def preprocess_pipeline(self, df):
        """
        Pipeline xử lý đầy đủ - ROBUST VERSION
        """
        print("="*60)
        print(" BẮT ĐẦU PREPROCESSING PIPELINE")
        print("="*60)
        
        # Lưu số observations ban đầu
        initial_count = len(df)
        initial_objects = df['object_id'].nunique() if 'object_id' in df.columns else 0
        
        try:
            # 1. Hiệu chỉnh extinction
            df = self.correct_extinction(df)
            
            # 2. Tính absolute magnitude
            df = self.calculate_absolute_magnitude(df)
            
            # 3. Làm sạch dữ liệu
            df = self.clean_data(df)
            
            # 4. Thêm thông tin thống kê cơ bản
            df = self.add_basic_stats(df)
            
            print(f"\n KẾT QUẢ PREPROCESSING:")
            print(f"   Số observations: {initial_count} → {len(df)}")
            print(f"   Số object_id duy nhất: {df['object_id'].nunique()}")
            
            # Hiển thị columns mới được tạo
            original_cols = ['flux', 'Flux', 'flux_err', 'Flux_err', 'object_id', 'mjd', 
                           'Filter', 'EBV', 'Z', 'FLUX', 'FLUX_ERR']
            new_cols = [c for c in df.columns if c not in original_cols]
            print(f"   Columns mới được tạo: {len(new_cols)} columns")
            if len(new_cols) <= 10:
                print(f"   New columns: {new_cols}")
            
            return df
            
        except Exception as e:
            print(f" Lỗi trong preprocessing pipeline: {e}")
            print("  Trả về dataframe gốc với minimal processing")
            
            # Minimal preprocessing
            df = df.copy()
            if 'object_id' in df.columns and 'mjd' in df.columns:
                df = df.sort_values(['object_id', 'mjd'])
            
            return df


# Legacy class for backward compatibility
class Preprocessor(RobustPreprocessor):
    """Legacy class - extends RobustPreprocessor"""
    def __init__(self):
        super().__init__()
        print("  Using legacy Preprocessor. Consider using RobustPreprocessor.")