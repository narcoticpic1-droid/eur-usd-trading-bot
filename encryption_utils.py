import os
import base64
import hashlib
import secrets
from typing import Dict, Optional, Tuple, Union
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import json
from datetime import datetime, timedelta

class EncryptionError(Exception):
    """خطای رمزگذاری"""
    pass

class EncryptionUtils:
    """ابزارهای رمزگذاری و امنیت"""
    
    def __init__(self, master_key: Optional[str] = None):
        self.master_key = master_key
        self._fernet_instance = None
        
        # تنظیمات پیش‌فرض
        self.salt_length = 32
        self.key_iterations = 100000
        self.aes_key_length = 32  # 256 بیت
        
        if master_key:
            self._initialize_fernet(master_key)
    
    def _initialize_fernet(self, password: str):
        """راه‌اندازی Fernet با کلید master"""
        try:
            # تولید salt ثابت از password (در محیط تولید باید salt ذخیره شود)
            salt = hashlib.sha256(password.encode()).digest()[:16]
            
            # تولید کلید
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=self.key_iterations,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
            
            self._fernet_instance = Fernet(key)
            
        except Exception as e:
            raise EncryptionError(f"خطا در راه‌اندازی رمزگذاری: {e}")
    
    def generate_key(self) -> str:
        """تولید کلید تصادفی قوی"""
        return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
    
    def generate_salt(self, length: int = None) -> bytes:
        """تولید salt تصادفی"""
        return secrets.token_bytes(length or self.salt_length)
    
    def hash_password(self, password: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
        """هش کردن پسورد با salt"""
        if salt is None:
            salt = self.generate_salt()
        
        # PBKDF2 برای هش پسورد
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.key_iterations,
        )
        
        hashed = kdf.derive(password.encode())
        
        return (
            base64.b64encode(hashed).decode(),
            base64.b64encode(salt).decode()
        )
    
    def verify_password(self, password: str, hashed: str, salt: str) -> bool:
        """تایید پسورد"""
        try:
            salt_bytes = base64.b64decode(salt.encode())
            stored_hash = base64.b64decode(hashed.encode())
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt_bytes,
                iterations=self.key_iterations,
            )
            
            # تایید پسورد
            kdf.verify(password.encode(), stored_hash)
            return True
            
        except Exception:
            return False
    
    def encrypt_text(self, text: str, key: Optional[str] = None) -> str:
        """رمزگذاری متن"""
        if not self._fernet_instance and not key:
            raise EncryptionError("کلید رمزگذاری تنظیم نشده")
        
        try:
            if key:
                # استفاده از کلید ارائه شده
                temp_fernet = Fernet(key.encode() if isinstance(key, str) else key)
                encrypted = temp_fernet.encrypt(text.encode())
            else:
                # استفاده از کلید master
                encrypted = self._fernet_instance.encrypt(text.encode())
            
            return base64.urlsafe_b64encode(encrypted).decode()
            
        except Exception as e:
            raise EncryptionError(f"خطا در رمزگذاری: {e}")
    
    def decrypt_text(self, encrypted_text: str, key: Optional[str] = None) -> str:
        """رمزگشایی متن"""
        if not self._fernet_instance and not key:
            raise EncryptionError("کلید رمزگذاری تنظیم نشده")
        
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_text.encode())
            
            if key:
                # استفاده از کلید ارائه شده
                temp_fernet = Fernet(key.encode() if isinstance(key, str) else key)
                decrypted = temp_fernet.decrypt(encrypted_bytes)
            else:
                # استفاده از کلید master
                decrypted = self._fernet_instance.decrypt(encrypted_bytes)
            
            return decrypted.decode()
            
        except Exception as e:
            raise EncryptionError(f"خطا در رمزگشایی: {e}")
    
    def encrypt_json(self, data: Dict, key: Optional[str] = None) -> str:
        """رمزگذاری داده‌های JSON"""
        try:
            json_str = json.dumps(data, ensure_ascii=False)
            return self.encrypt_text(json_str, key)
        except Exception as e:
            raise EncryptionError(f"خطا در رمزگذاری JSON: {e}")
    
    def decrypt_json(self, encrypted_data: str, key: Optional[str] = None) -> Dict:
        """رمزگشایی داده‌های JSON"""
        try:
            decrypted_str = self.decrypt_text(encrypted_data, key)
            return json.loads(decrypted_str)
        except Exception as e:
            raise EncryptionError(f"خطا در رمزگشایی JSON: {e}")
    
    def encrypt_api_key(self, api_key: str, service_name: str) -> Dict[str, str]:
        """رمزگذاری کلید API"""
        try:
            # تولید کلید منحصر به فرد برای هر سرویس
            service_salt = hashlib.sha256(f"{service_name}_{datetime.now().date()}".encode()).digest()
            
            # ایجاد کلید برای این سرویس
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=service_salt[:16],
                iterations=self.key_iterations,
            )
            
            if self.master_key:
                service_key = base64.urlsafe_b64encode(kdf.derive(self.master_key.encode()))
            else:
                raise EncryptionError("Master key تنظیم نشده")
            
            # رمزگذاری
            fernet = Fernet(service_key)
            encrypted = fernet.encrypt(api_key.encode())
            
            return {
                'encrypted_key': base64.urlsafe_b64encode(encrypted).decode(),
                'service': service_name,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            raise EncryptionError(f"خطا در رمزگذاری API key: {e}")
    
    def decrypt_api_key(self, encrypted_data: Dict[str, str]) -> str:
        """رمزگشایی کلید API"""
        try:
            service_name = encrypted_data['service']
            encrypted_key = encrypted_data['encrypted_key']
            
            # بازسازی کلید سرویس
            service_salt = hashlib.sha256(f"{service_name}_{datetime.now().date()}".encode()).digest()
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=service_salt[:16],
                iterations=self.key_iterations,
            )
            
            if self.master_key:
                service_key = base64.urlsafe_b64encode(kdf.derive(self.master_key.encode()))
            else:
                raise EncryptionError("Master key تنظیم نشده")
            
            # رمزگشایی
            fernet = Fernet(service_key)
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_key.encode())
            decrypted = fernet.decrypt(encrypted_bytes)
            
            return decrypted.decode()
            
        except Exception as e:
            raise EncryptionError(f"خطا در رمزگشایی API key: {e}")
    
    def generate_rsa_keypair(self, key_size: int = 2048) -> Tuple[str, str]:
        """تولید جفت کلید RSA"""
        try:
            # تولید کلید خصوصی
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size,
            )
            
            # تولید کلید عمومی
            public_key = private_key.public_key()
            
            # سریالایز کردن کلیدها
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            return (
                public_pem.decode(),
                private_pem.decode()
            )
            
        except Exception as e:
            raise EncryptionError(f"خطا در تولید کلید RSA: {e}")
    
    def encrypt_with_rsa(self, data: str, public_key_pem: str) -> str:
        """رمزگذاری با کلید عمومی RSA"""
        try:
            public_key = serialization.load_pem_public_key(public_key_pem.encode())
            
            encrypted = public_key.encrypt(
                data.encode(),
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            return base64.b64encode(encrypted).decode()
            
        except Exception as e:
            raise EncryptionError(f"خطا در رمزگذاری RSA: {e}")
    
    def decrypt_with_rsa(self, encrypted_data: str, private_key_pem: str) -> str:
        """رمزگشایی با کلید خصوصی RSA"""
        try:
            private_key = serialization.load_pem_private_key(
                private_key_pem.encode(),
                password=None
            )
            
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            
            decrypted = private_key.decrypt(
                encrypted_bytes,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            return decrypted.decode()
            
        except Exception as e:
            raise EncryptionError(f"خطا در رمزگشایی RSA: {e}")
    
    def create_secure_token(self, length: int = 32) -> str:
        """تولید توکن امن"""
        return secrets.token_urlsafe(length)
    
    def hash_data(self, data: str, algorithm: str = "sha256") -> str:
        """هش کردن داده‌ها"""
        try:
            if algorithm == "sha256":
                hasher = hashlib.sha256()
            elif algorithm == "sha512":
                hasher = hashlib.sha512()
            elif algorithm == "md5":
                hasher = hashlib.md5()
            else:
                raise ValueError(f"الگوریتم پشتیبانی نشده: {algorithm}")
            
            hasher.update(data.encode())
            return hasher.hexdigest()
            
        except Exception as e:
            raise EncryptionError(f"خطا در هش کردن: {e}")
    
    def secure_compare(self, a: str, b: str) -> bool:
        """مقایسه امن strings"""
        return secrets.compare_digest(a.encode(), b.encode())
    
    def encrypt_file(self, file_path: str, output_path: str, key: Optional[str] = None):
        """رمزگذاری فایل"""
        if not self._fernet_instance and not key:
            raise EncryptionError("کلید رمزگذاری تنظیم نشده")
        
        try:
            with open(file_path, 'rb') as file:
                file_data = file.read()
            
            if key:
                temp_fernet = Fernet(key.encode() if isinstance(key, str) else key)
                encrypted_data = temp_fernet.encrypt(file_data)
            else:
                encrypted_data = self._fernet_instance.encrypt(file_data)
            
            with open(output_path, 'wb') as encrypted_file:
                encrypted_file.write(encrypted_data)
                
        except Exception as e:
            raise EncryptionError(f"خطا در رمزگذاری فایل: {e}")
    
    def decrypt_file(self, encrypted_file_path: str, output_path: str, key: Optional[str] = None):
        """رمزگشایی فایل"""
        if not self._fernet_instance and not key:
            raise EncryptionError("کلید رمزگذاری تنظیم نشده")
        
        try:
            with open(encrypted_file_path, 'rb') as encrypted_file:
                encrypted_data = encrypted_file.read()
            
            if key:
                temp_fernet = Fernet(key.encode() if isinstance(key, str) else key)
                file_data = temp_fernet.decrypt(encrypted_data)
            else:
                file_data = self._fernet_instance.decrypt(encrypted_data)
            
            with open(output_path, 'wb') as file:
                file.write(file_data)
                
        except Exception as e:
            raise EncryptionError(f"خطا در رمزگشایی فایل: {e}")
    
    def create_session_token(self, user_id: str, expire_hours: int = 24) -> Dict[str, str]:
        """ایجاد توکن session"""
        try:
            session_data = {
                'user_id': user_id,
                'created_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + timedelta(hours=expire_hours)).isoformat(),
                'random': self.create_secure_token(16)
            }
            
            token = self.encrypt_json(session_data)
            
            return {
                'token': token,
                'expires_at': session_data['expires_at']
            }
            
        except Exception as e:
            raise EncryptionError(f"خطا در ایجاد session token: {e}")
    
    def validate_session_token(self, token: str) -> Optional[Dict[str, str]]:
        """اعتبارسنجی توکن session"""
        try:
            session_data = self.decrypt_json(token)
            
            # بررسی انقضا
            expires_at = datetime.fromisoformat(session_data['expires_at'])
            if datetime.now() > expires_at:
                return None
            
            return session_data
            
        except Exception:
            return None

# تابع راحت برای استفاده
def create_encryption_manager(master_key: str) -> EncryptionUtils:
    """ایجاد مدیر رمزگذاری"""
    return EncryptionUtils(master_key)

def quick_encrypt(text: str, password: str) -> str:
    """رمزگذاری سریع"""
    manager = EncryptionUtils(password)
    return manager.encrypt_text(text)

def quick_decrypt(encrypted_text: str, password: str) -> str:
    """رمزگشایی سریع"""
    manager = EncryptionUtils(password)
    return manager.decrypt_text(encrypted_text)
