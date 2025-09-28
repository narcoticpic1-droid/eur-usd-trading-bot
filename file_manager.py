import os
import shutil
import json
import pickle
import gzip
from typing import Any, List, Dict, Optional, Union
from datetime import datetime
import hashlib

class FileError(Exception):
    """خطای مدیریت فایل"""
    pass

class FileManager:
    """
    مدیریت فایل‌ها و پوشه‌ها
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.auto_backup = self.config.get('auto_backup', True)
        self.backup_count = self.config.get('backup_count', 5)
        self.max_file_size = self.config.get('max_file_size_mb', 100) * 1024 * 1024
        
        # پوشه‌های پیش‌فرض
        self.data_dir = self.config.get('data_dir', 'data')
        self.backup_dir = self.config.get('backup_dir', 'backups')
        self.temp_dir = self.config.get('temp_dir', 'temp')
        
        # ایجاد پوشه‌ها
        for directory in [self.data_dir, self.backup_dir, self.temp_dir]:
            self.ensure_directory(directory)
    
    def ensure_directory(self, path: str) -> bool:
        """اطمینان از وجود پوشه"""
        try:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
            return True
        except Exception as e:
            raise FileError(f"خطا در ایجاد پوشه {path}: {e}")
    
    def save_data(self, data: Any, filename: str, format_type: str = 'json', 
                 compress: bool = False) -> bool:
        """ذخیره داده در فایل"""
        try:
            file_path = os.path.join(self.data_dir, filename)
            
            # پشتیبان‌گیری خودکار
            if self.auto_backup and os.path.exists(file_path):
                self.create_backup(file_path)
            
            # بررسی اندازه فایل
            if format_type == 'json':
                test_data = json.dumps(data, ensure_ascii=False)
                if len(test_data.encode('utf-8')) > self.max_file_size:
                    raise FileError(f"اندازه فایل بیش از حد مجاز: {self.max_file_size/1024/1024}MB")
            
            # ذخیره بر اساس فرمت
            if format_type == 'json':
                self._save_json(data, file_path, compress)
            elif format_type == 'pickle':
                self._save_pickle(data, file_path, compress)
            elif format_type == 'text':
                self._save_text(data, file_path, compress)
            else:
                raise FileError(f"فرمت پشتیبانی نشده: {format_type}")
            
            return True
            
        except Exception as e:
            raise FileError(f"خطا در ذخیره فایل {filename}: {e}")
    
    def load_data(self, filename: str, format_type: str = 'json', 
                 decompress: bool = False) -> Any:
        """بارگذاری داده از فایل"""
        try:
            file_path = os.path.join(self.data_dir, filename)
            
            if not os.path.exists(file_path):
                raise FileError(f"فایل یافت نشد: {file_path}")
            
            # بارگذاری بر اساس فرمت
            if format_type == 'json':
                return self._load_json(file_path, decompress)
            elif format_type == 'pickle':
                return self._load_pickle(file_path, decompress)
            elif format_type == 'text':
                return self._load_text(file_path, decompress)
            else:
                raise FileError(f"فرمت پشتیبانی نشده: {format_type}")
                
        except Exception as e:
            raise FileError(f"خطا در بارگذاری فایل {filename}: {e}")
    
    def _save_json(self, data: Any, file_path: str, compress: bool = False):
        """ذخیره JSON"""
        json_data = json.dumps(data, ensure_ascii=False, indent=2)
        
        if compress:
            with gzip.open(f"{file_path}.gz", 'wt', encoding='utf-8') as f:
                f.write(json_data)
        else:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(json_data)
    
    def _load_json(self, file_path: str, decompress: bool = False) -> Any:
        """بارگذاری JSON"""
        if decompress:
            with gzip.open(f"{file_path}.gz", 'rt', encoding='utf-8') as f:
                return json.load(f)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    
    def _save_pickle(self, data: Any, file_path: str, compress: bool = False):
        """ذخیره Pickle"""
        if compress:
            with gzip.open(f"{file_path}.gz", 'wb') as f:
                pickle.dump(data, f)
        else:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
    
    def _load_pickle(self, file_path: str, decompress: bool = False) -> Any:
        """بارگذاری Pickle"""
        if decompress:
            with gzip.open(f"{file_path}.gz", 'rb') as f:
                return pickle.load(f)
        else:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
    
    def _save_text(self, data: str, file_path: str, compress: bool = False):
        """ذخیره متن"""
        if compress:
            with gzip.open(f"{file_path}.gz", 'wt', encoding='utf-8') as f:
                f.write(str(data))
        else:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(str(data))
    
    def _load_text(self, file_path: str, decompress: bool = False) -> str:
        """بارگذاری متن"""
        if decompress:
            with gzip.open(f"{file_path}.gz", 'rt', encoding='utf-8') as f:
                return f.read()
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    def create_backup(self, file_path: str) -> str:
        """ایجاد پشتیبان از فایل"""
        try:
            if not os.path.exists(file_path):
                raise FileError(f"فایل برای پشتیبان‌گیری یافت نشد: {file_path}")
            
            filename = os.path.basename(file_path)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"{filename}.backup_{timestamp}"
            backup_path = os.path.join(self.backup_dir, backup_filename)
            
            shutil.copy2(file_path, backup_path)
            
            # پاک‌سازی پشتیبان‌های قدیمی
            self._cleanup_old_backups(filename)
            
            return backup_path
            
        except Exception as e:
            raise FileError(f"خطا در ایجاد پشتیبان: {e}")
    
    def _cleanup_old_backups(self, filename: str):
        """پاک‌سازی پشتیبان‌های قدیمی"""
        try:
            backup_pattern = f"{filename}.backup_"
            backups = []
            
            for file in os.listdir(self.backup_dir):
                if file.startswith(backup_pattern):
                    file_path = os.path.join(self.backup_dir, file)
                    backups.append((file_path, os.path.getmtime(file_path)))
            
            # مرتب‌سازی بر اساس زمان (جدیدترین اول)
            backups.sort(key=lambda x: x[1], reverse=True)
            
            # حذف پشتیبان‌های اضافی
            for backup_path, _ in backups[self.backup_count:]:
                os.remove(backup_path)
                
        except Exception as e:
            print(f"هشدار: خطا در پاک‌سازی پشتیبان‌ها: {e}")
    
    def restore_backup(self, filename: str, backup_timestamp: str = None) -> bool:
        """بازیابی از پشتیبان"""
        try:
            if backup_timestamp:
                backup_filename = f"{filename}.backup_{backup_timestamp}"
            else:
                # استفاده از آخرین پشتیبان
                backup_pattern = f"{filename}.backup_"
                backups = []
                
                for file in os.listdir(self.backup_dir):
                    if file.startswith(backup_pattern):
                        file_path = os.path.join(self.backup_dir, file)
                        backups.append((file, os.path.getmtime(file_path)))
                
                if not backups:
                    raise FileError(f"پشتیبانی برای {filename} یافت نشد")
                
                backups.sort(key=lambda x: x[1], reverse=True)
                backup_filename = backups[0][0]
            
            backup_path = os.path.join(self.backup_dir, backup_filename)
            
            if not os.path.exists(backup_path):
                raise FileError(f"فایل پشتیبان یافت نشد: {backup_path}")
            
            target_path = os.path.join(self.data_dir, filename)
            shutil.copy2(backup_path, target_path)
            
            return True
            
        except Exception as e:
            raise FileError(f"خطا در بازیابی پشتیبان: {e}")
    
    def delete_file(self, filename: str, create_backup: bool = True) -> bool:
        """حذف فایل"""
        try:
            file_path = os.path.join(self.data_dir, filename)
            
            if not os.path.exists(file_path):
                raise FileError(f"فایل برای حذف یافت نشد: {file_path}")
            
            if create_backup:
                self.create_backup(file_path)
            
            os.remove(file_path)
            return True
            
        except Exception as e:
            raise FileError(f"خطا در حذف فایل: {e}")
    
    def list_files(self, pattern: str = None, directory: str = None) -> List[Dict[str, Any]]:
        """لیست فایل‌ها"""
        try:
            if directory is None:
                directory = self.data_dir
            
            files = []
            
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                
                if os.path.isfile(file_path):
                    if pattern is None or pattern in filename:
                        stat = os.stat(file_path)
                        files.append({
                            'name': filename,
                            'path': file_path,
                            'size': stat.st_size,
                            'size_mb': round(stat.st_size / 1024 / 1024, 2),
                            'modified': datetime.fromtimestamp(stat.st_mtime),
                            'created': datetime.fromtimestamp(stat.st_ctime)
                        })
            
            return sorted(files, key=lambda x: x['modified'], reverse=True)
            
        except Exception as e:
            raise FileError(f"خطا در لیست فایل‌ها: {e}")
    
    def get_file_info(self, filename: str) -> Dict[str, Any]:
        """اطلاعات فایل"""
        try:
            file_path = os.path.join(self.data_dir, filename)
            
            if not os.path.exists(file_path):
                raise FileError(f"فایل یافت نشد: {file_path}")
            
            stat = os.stat(file_path)
            
            # محاسبه hash
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            return {
                'name': filename,
                'path': file_path,
                'size': stat.st_size,
                'size_mb': round(stat.st_size / 1024 / 1024, 2),
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'created': datetime.fromtimestamp(stat.st_ctime),
                'hash': file_hash,
                'readable': os.access(file_path, os.R_OK),
                'writable': os.access(file_path, os.W_OK)
            }
            
        except Exception as e:
            raise FileError(f"خطا در دریافت اطلاعات فایل: {e}")
    
    def copy_file(self, source_filename: str, target_filename: str) -> bool:
        """کپی فایل"""
        try:
            source_path = os.path.join(self.data_dir, source_filename)
            target_path = os.path.join(self.data_dir, target_filename)
            
            if not os.path.exists(source_path):
                raise FileError(f"فایل مبدأ یافت نشد: {source_path}")
            
            shutil.copy2(source_path, target_path)
            return True
            
        except Exception as e:
            raise FileError(f"خطا در کپی فایل: {e}")
    
    def move_file(self, source_filename: str, target_filename: str) -> bool:
        """جابجایی فایل"""
        try:
            source_path = os.path.join(self.data_dir, source_filename)
            target_path = os.path.join(self.data_dir, target_filename)
            
            if not os.path.exists(source_path):
                raise FileError(f"فایل مبدأ یافت نشد: {source_path}")
            
            shutil.move(source_path, target_path)
            return True
            
        except Exception as e:
            raise FileError(f"خطا در جابجایی فایل: {e}")
    
    def compress_file(self, filename: str, remove_original: bool = False) -> str:
        """فشرده‌سازی فایل"""
        try:
            file_path = os.path.join(self.data_dir, filename)
            
            if not os.path.exists(file_path):
                raise FileError(f"فایل یافت نشد: {file_path}")
            
            compressed_path = f"{file_path}.gz"
            
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            if remove_original:
                os.remove(file_path)
            
            return compressed_path
            
        except Exception as e:
            raise FileError(f"خطا در فشرده‌سازی: {e}")
    
    def decompress_file(self, compressed_filename: str, remove_compressed: bool = False) -> str:
        """از حالت فشرده خارج کردن"""
        try:
            compressed_path = os.path.join(self.data_dir, compressed_filename)
            
            if not os.path.exists(compressed_path):
                raise FileError(f"فایل فشرده یافت نشد: {compressed_path}")
            
            if compressed_filename.endswith('.gz'):
                output_path = compressed_path[:-3]  # حذف .gz
            else:
                output_path = f"{compressed_path}.decompressed"
            
            with gzip.open(compressed_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            if remove_compressed:
                os.remove(compressed_path)
            
            return output_path
            
        except Exception as e:
            raise FileError(f"خطا در از حالت فشرده خارج کردن: {e}")
    
    def cleanup_temp_files(self, older_than_hours: int = 24) -> int:
        """پاک‌سازی فایل‌های موقت"""
        try:
            cleanup_count = 0
            current_time = datetime.now()
            
            for filename in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, filename)
                
                if os.path.isfile(file_path):
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    age_hours = (current_time - file_time).total_seconds() / 3600
                    
                    if age_hours > older_than_hours:
                        os.remove(file_path)
                        cleanup_count += 1
            
            return cleanup_count
            
        except Exception as e:
            raise FileError(f"خطا در پاک‌سازی فایل‌های موقت: {e}")
