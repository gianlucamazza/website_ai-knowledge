"""
Secrets Management System

Provides secure storage and retrieval of sensitive credentials including
API keys, database passwords, and other secrets with encryption, rotation,
and audit logging.
"""

import os
import json
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import keyring
import hvac  # HashiCorp Vault client
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential


logger = logging.getLogger(__name__)


@dataclass
class SecretConfig:
    """Configuration for secrets management."""
    
    # Storage backend
    backend: str = "file"  # file, keyring, vault, azure_keyvault
    
    # File backend settings
    secrets_file_path: str = ".secrets/secrets.json.enc"
    master_key_env_var: str = "MASTER_SECRET_KEY"
    
    # Encryption settings
    encryption_algorithm: str = "fernet"
    key_derivation_iterations: int = 100000
    
    # Vault settings (for HashiCorp Vault)
    vault_url: Optional[str] = None
    vault_token: Optional[str] = None
    vault_mount_point: str = "secret"
    
    # Azure Key Vault settings
    azure_keyvault_url: Optional[str] = None
    
    # Secret rotation settings
    enable_rotation: bool = True
    rotation_warning_days: int = 30
    max_secret_age_days: int = 90
    
    # Audit settings
    enable_audit_logging: bool = True
    audit_log_path: str = ".security/secrets_audit.log"
    
    # Validation settings
    validate_secrets: bool = True
    required_secrets: List[str] = field(default_factory=lambda: [
        "DATABASE_PASSWORD",
        "JWT_SECRET_KEY",
        "ENCRYPTION_KEY"
    ])


@dataclass
class Secret:
    """Individual secret data structure."""
    
    key: str
    value: str
    description: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    is_encrypted: bool = True
    rotation_enabled: bool = True
    
    def is_expired(self) -> bool:
        """Check if secret has expired."""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at
        
    def needs_rotation(self, warning_days: int = 30) -> bool:
        """Check if secret needs rotation soon."""
        if not self.expires_at or not self.rotation_enabled:
            return False
        warning_date = self.expires_at - timedelta(days=warning_days)
        return datetime.now(timezone.utc) > warning_date


@dataclass
class SecretAuditEvent:
    """Audit event for secret operations."""
    
    event_type: str  # CREATE, READ, UPDATE, DELETE, ROTATE
    secret_key: str
    user: str = "system"
    ip_address: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


class SecretValidationError(Exception):
    """Secret validation error."""
    pass


class SecretAccessError(Exception):
    """Secret access error.""" 
    pass


class SecretsManager:
    """
    Comprehensive secrets management system.
    
    Features:
    - Multiple storage backends (file, keyring, Vault, Azure Key Vault)
    - Encryption at rest
    - Secret rotation and expiration
    - Audit logging
    - Validation and compliance checks
    - Access control and monitoring
    """
    
    def __init__(self, config: SecretConfig):
        self.config = config
        self._setup_encryption()
        self._setup_backend()
        self._setup_audit_logging()
        
        # In-memory cache for performance
        self._cache: Dict[str, Secret] = {}
        self._cache_ttl: Dict[str, datetime] = {}
        
    def _setup_encryption(self) -> None:
        """Setup encryption components."""
        master_key = os.environ.get(self.config.master_key_env_var)
        
        if not master_key:
            # Generate a new master key if none exists
            master_key = base64.urlsafe_b64encode(os.urandom(32)).decode()
            logger.warning(
                f"No master key found in {self.config.master_key_env_var}. "
                f"Generated temporary key: {master_key[:8]}..."
            )
            
        # Derive encryption key from master key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'ai_knowledge_salt',  # In production, use random salt
            iterations=self.config.key_derivation_iterations,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
        self.fernet = Fernet(key)
        
    def _setup_backend(self) -> None:
        """Setup storage backend."""
        if self.config.backend == "vault":
            self._setup_vault()
        elif self.config.backend == "azure_keyvault":
            self._setup_azure_keyvault()
        elif self.config.backend == "keyring":
            self._setup_keyring()
        else:
            self._setup_file_backend()
            
    def _setup_vault(self) -> None:
        """Setup HashiCorp Vault backend."""
        if not self.config.vault_url:
            raise ValueError("Vault URL must be configured")
            
        self.vault_client = hvac.Client(
            url=self.config.vault_url,
            token=self.config.vault_token
        )
        
        if not self.vault_client.is_authenticated():
            raise ValueError("Failed to authenticate with Vault")
            
    def _setup_azure_keyvault(self) -> None:
        """Setup Azure Key Vault backend."""
        if not self.config.azure_keyvault_url:
            raise ValueError("Azure Key Vault URL must be configured")
            
        credential = DefaultAzureCredential()
        self.keyvault_client = SecretClient(
            vault_url=self.config.azure_keyvault_url,
            credential=credential
        )
        
    def _setup_keyring(self) -> None:
        """Setup keyring backend."""
        # Test keyring availability
        try:
            keyring.get_keyring()
        except Exception as e:
            logger.error(f"Keyring not available: {e}")
            raise ValueError("Keyring backend not available")
            
    def _setup_file_backend(self) -> None:
        """Setup file-based backend."""
        secrets_file = Path(self.config.secrets_file_path)
        secrets_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create empty secrets file if it doesn't exist
        if not secrets_file.exists():
            self._save_secrets_file({})
            
    def _setup_audit_logging(self) -> None:
        """Setup audit logging."""
        if not self.config.enable_audit_logging:
            return
            
        audit_file = Path(self.config.audit_log_path)
        audit_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup audit logger
        self.audit_logger = logging.getLogger("secrets_audit")
        self.audit_logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(audit_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.audit_logger.addHandler(handler)
        
    def create_secret(self, 
                     key: str, 
                     value: str,
                     description: str = "",
                     expires_in_days: Optional[int] = None,
                     tags: Optional[List[str]] = None,
                     rotation_enabled: bool = True) -> Secret:
        """
        Create a new secret.
        
        Args:
            key: Secret identifier
            value: Secret value
            description: Human-readable description
            expires_in_days: Expiration in days (None for no expiration)
            tags: List of tags for categorization
            rotation_enabled: Whether secret supports rotation
            
        Returns:
            Created secret object
            
        Raises:
            SecretValidationError: If secret validation fails
        """
        # Validate secret
        self._validate_secret(key, value)
        
        # Check if secret already exists
        if self._secret_exists(key):
            raise SecretValidationError(f"Secret '{key}' already exists")
            
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)
            
        secret = Secret(
            key=key,
            value=value,
            description=description,
            expires_at=expires_at,
            tags=tags or [],
            rotation_enabled=rotation_enabled
        )
        
        # Store secret
        self._store_secret(secret)
        
        # Audit log
        self._audit_log(SecretAuditEvent(
            event_type="CREATE",
            secret_key=key,
            metadata={"description": description, "tags": tags}
        ))
        
        logger.info(f"Created secret: {key}")
        return secret
        
    def get_secret(self, key: str, user: str = "system") -> str:
        """
        Retrieve secret value.
        
        Args:
            key: Secret identifier
            user: User requesting the secret
            
        Returns:
            Secret value
            
        Raises:
            SecretAccessError: If secret cannot be accessed
        """
        # Check cache first
        if key in self._cache and self._is_cache_valid(key):
            secret = self._cache[key]
        else:
            secret = self._load_secret(key)
            if not secret:
                self._audit_log(SecretAuditEvent(
                    event_type="READ",
                    secret_key=key,
                    user=user,
                    success=False,
                    error_message="Secret not found"
                ))
                raise SecretAccessError(f"Secret '{key}' not found")
                
            # Update cache
            self._cache[key] = secret
            self._cache_ttl[key] = datetime.now(timezone.utc) + timedelta(minutes=5)
            
        # Check if expired
        if secret.is_expired():
            self._audit_log(SecretAuditEvent(
                event_type="READ",
                secret_key=key,
                user=user,
                success=False,
                error_message="Secret expired"
            ))
            raise SecretAccessError(f"Secret '{key}' has expired")
            
        # Update access tracking
        secret.access_count += 1
        secret.last_accessed = datetime.now(timezone.utc)
        self._store_secret(secret)
        
        # Audit log
        self._audit_log(SecretAuditEvent(
            event_type="READ",
            secret_key=key,
            user=user,
            success=True
        ))
        
        return secret.value
        
    def update_secret(self, 
                     key: str, 
                     value: str,
                     user: str = "system") -> Secret:
        """
        Update existing secret.
        
        Args:
            key: Secret identifier
            value: New secret value
            user: User updating the secret
            
        Returns:
            Updated secret object
        """
        secret = self._load_secret(key)
        if not secret:
            raise SecretAccessError(f"Secret '{key}' not found")
            
        # Validate new value
        self._validate_secret(key, value)
        
        # Update secret
        old_value_hash = hashlib.sha256(secret.value.encode()).hexdigest()[:8]
        secret.value = value
        secret.updated_at = datetime.now(timezone.utc)
        
        # Store updated secret
        self._store_secret(secret)
        
        # Invalidate cache
        self._cache.pop(key, None)
        self._cache_ttl.pop(key, None)
        
        # Audit log
        self._audit_log(SecretAuditEvent(
            event_type="UPDATE",
            secret_key=key,
            user=user,
            metadata={"old_value_hash": old_value_hash}
        ))
        
        logger.info(f"Updated secret: {key}")
        return secret
        
    def delete_secret(self, key: str, user: str = "system") -> bool:
        """
        Delete a secret.
        
        Args:
            key: Secret identifier
            user: User deleting the secret
            
        Returns:
            True if deleted successfully
        """
        if not self._secret_exists(key):
            return False
            
        # Remove from storage
        self._delete_secret(key)
        
        # Remove from cache
        self._cache.pop(key, None)
        self._cache_ttl.pop(key, None)
        
        # Audit log
        self._audit_log(SecretAuditEvent(
            event_type="DELETE",
            secret_key=key,
            user=user
        ))
        
        logger.info(f"Deleted secret: {key}")
        return True
        
    def rotate_secret(self, 
                     key: str, 
                     new_value: str,
                     user: str = "system") -> Secret:
        """
        Rotate a secret (update with new expiration).
        
        Args:
            key: Secret identifier
            new_value: New secret value
            user: User rotating the secret
            
        Returns:
            Rotated secret object
        """
        secret = self._load_secret(key)
        if not secret:
            raise SecretAccessError(f"Secret '{key}' not found")
            
        if not secret.rotation_enabled:
            raise SecretValidationError(f"Secret '{key}' does not support rotation")
            
        # Update secret with new value and reset expiration
        old_value_hash = hashlib.sha256(secret.value.encode()).hexdigest()[:8]
        secret.value = new_value
        secret.updated_at = datetime.now(timezone.utc)
        
        # Reset expiration if configured
        if self.config.max_secret_age_days > 0:
            secret.expires_at = datetime.now(timezone.utc) + timedelta(
                days=self.config.max_secret_age_days
            )
            
        # Store rotated secret
        self._store_secret(secret)
        
        # Invalidate cache
        self._cache.pop(key, None)
        self._cache_ttl.pop(key, None)
        
        # Audit log
        self._audit_log(SecretAuditEvent(
            event_type="ROTATE",
            secret_key=key,
            user=user,
            metadata={"old_value_hash": old_value_hash}
        ))
        
        logger.info(f"Rotated secret: {key}")
        return secret
        
    def list_secrets(self, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        List all secrets (without values).
        
        Args:
            tags: Filter by tags
            
        Returns:
            List of secret metadata
        """
        secrets = self._list_all_secrets()
        result = []
        
        for secret in secrets:
            # Filter by tags if specified
            if tags and not any(tag in secret.tags for tag in tags):
                continue
                
            result.append({
                'key': secret.key,
                'description': secret.description,
                'created_at': secret.created_at.isoformat(),
                'updated_at': secret.updated_at.isoformat(),
                'expires_at': secret.expires_at.isoformat() if secret.expires_at else None,
                'tags': secret.tags,
                'access_count': secret.access_count,
                'last_accessed': secret.last_accessed.isoformat() if secret.last_accessed else None,
                'is_expired': secret.is_expired(),
                'needs_rotation': secret.needs_rotation(self.config.rotation_warning_days)
            })
            
        return result
        
    def get_secrets_needing_rotation(self) -> List[Dict[str, Any]]:
        """Get list of secrets that need rotation."""
        secrets = self._list_all_secrets()
        return [
            {
                'key': secret.key,
                'expires_at': secret.expires_at.isoformat() if secret.expires_at else None,
                'days_until_expiry': (secret.expires_at - datetime.now(timezone.utc)).days
                                   if secret.expires_at else None
            }
            for secret in secrets
            if secret.needs_rotation(self.config.rotation_warning_days)
        ]
        
    def validate_all_secrets(self) -> Dict[str, List[str]]:
        """
        Validate all secrets and return validation report.
        
        Returns:
            Dictionary with validation results
        """
        report = {
            'valid_secrets': [],
            'invalid_secrets': [],
            'expired_secrets': [],
            'missing_required': [],
            'rotation_needed': []
        }
        
        secrets = self._list_all_secrets()
        secret_keys = {secret.key for secret in secrets}
        
        # Check for missing required secrets
        for required_key in self.config.required_secrets:
            if required_key not in secret_keys:
                report['missing_required'].append(required_key)
                
        # Validate each secret
        for secret in secrets:
            try:
                self._validate_secret(secret.key, secret.value)
                
                if secret.is_expired():
                    report['expired_secrets'].append(secret.key)
                elif secret.needs_rotation(self.config.rotation_warning_days):
                    report['rotation_needed'].append(secret.key)
                else:
                    report['valid_secrets'].append(secret.key)
                    
            except SecretValidationError:
                report['invalid_secrets'].append(secret.key)
                
        return report
        
    def _validate_secret(self, key: str, value: str) -> None:
        """Validate secret key and value."""
        if not key or not isinstance(key, str):
            raise SecretValidationError("Secret key must be a non-empty string")
            
        if not value or not isinstance(value, str):
            raise SecretValidationError("Secret value must be a non-empty string")
            
        # Check key format
        if not key.replace('_', '').replace('-', '').isalnum():
            raise SecretValidationError("Secret key must contain only alphanumeric characters, hyphens, and underscores")
            
        # Check value strength for certain types
        if 'password' in key.lower():
            if len(value) < 12:
                raise SecretValidationError("Password secrets must be at least 12 characters")
                
        if 'api_key' in key.lower():
            if len(value) < 32:
                raise SecretValidationError("API key secrets must be at least 32 characters")
                
        # Check for common weak values
        weak_values = {'password', '123456', 'admin', 'secret'}
        if value.lower() in weak_values:
            raise SecretValidationError("Secret value is too weak")
            
    def _secret_exists(self, key: str) -> bool:
        """Check if secret exists."""
        return self._load_secret(key) is not None
        
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached secret is still valid."""
        if key not in self._cache_ttl:
            return False
        return datetime.now(timezone.utc) < self._cache_ttl[key]
        
    def _store_secret(self, secret: Secret) -> None:
        """Store secret using configured backend."""
        if self.config.backend == "vault":
            self._store_secret_vault(secret)
        elif self.config.backend == "azure_keyvault":
            self._store_secret_azure(secret)
        elif self.config.backend == "keyring":
            self._store_secret_keyring(secret)
        else:
            self._store_secret_file(secret)
            
    def _load_secret(self, key: str) -> Optional[Secret]:
        """Load secret using configured backend."""
        if self.config.backend == "vault":
            return self._load_secret_vault(key)
        elif self.config.backend == "azure_keyvault":
            return self._load_secret_azure(key)
        elif self.config.backend == "keyring":
            return self._load_secret_keyring(key)
        else:
            return self._load_secret_file(key)
            
    def _delete_secret(self, key: str) -> None:
        """Delete secret using configured backend."""
        if self.config.backend == "vault":
            self._delete_secret_vault(key)
        elif self.config.backend == "azure_keyvault":
            self._delete_secret_azure(key)
        elif self.config.backend == "keyring":
            self._delete_secret_keyring(key)
        else:
            self._delete_secret_file(key)
            
    def _list_all_secrets(self) -> List[Secret]:
        """List all secrets using configured backend."""
        if self.config.backend == "vault":
            return self._list_secrets_vault()
        elif self.config.backend == "azure_keyvault":
            return self._list_secrets_azure()
        elif self.config.backend == "keyring":
            return self._list_secrets_keyring()
        else:
            return self._list_secrets_file()
            
    # File backend implementations
    def _store_secret_file(self, secret: Secret) -> None:
        """Store secret in encrypted file."""
        secrets = self._load_secrets_file()
        secrets[secret.key] = {
            'value': secret.value,
            'description': secret.description,
            'created_at': secret.created_at.isoformat(),
            'updated_at': secret.updated_at.isoformat(),
            'expires_at': secret.expires_at.isoformat() if secret.expires_at else None,
            'tags': secret.tags,
            'access_count': secret.access_count,
            'last_accessed': secret.last_accessed.isoformat() if secret.last_accessed else None,
            'rotation_enabled': secret.rotation_enabled
        }
        self._save_secrets_file(secrets)
        
    def _load_secret_file(self, key: str) -> Optional[Secret]:
        """Load secret from encrypted file."""
        secrets = self._load_secrets_file()
        if key not in secrets:
            return None
            
        data = secrets[key]
        return Secret(
            key=key,
            value=data['value'],
            description=data.get('description', ''),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            expires_at=datetime.fromisoformat(data['expires_at']) if data.get('expires_at') else None,
            tags=data.get('tags', []),
            access_count=data.get('access_count', 0),
            last_accessed=datetime.fromisoformat(data['last_accessed']) if data.get('last_accessed') else None,
            rotation_enabled=data.get('rotation_enabled', True)
        )
        
    def _delete_secret_file(self, key: str) -> None:
        """Delete secret from encrypted file."""
        secrets = self._load_secrets_file()
        secrets.pop(key, None)
        self._save_secrets_file(secrets)
        
    def _list_secrets_file(self) -> List[Secret]:
        """List all secrets from encrypted file."""
        secrets_data = self._load_secrets_file()
        return [self._load_secret_file(key) for key in secrets_data.keys()]
        
    def _load_secrets_file(self) -> Dict[str, Any]:
        """Load and decrypt secrets file."""
        secrets_file = Path(self.config.secrets_file_path)
        
        if not secrets_file.exists():
            return {}
            
        try:
            encrypted_data = secrets_file.read_bytes()
            decrypted_data = self.fernet.decrypt(encrypted_data)
            return json.loads(decrypted_data.decode())
        except Exception as e:
            logger.error(f"Failed to load secrets file: {e}")
            return {}
            
    def _save_secrets_file(self, secrets: Dict[str, Any]) -> None:
        """Encrypt and save secrets file."""
        try:
            json_data = json.dumps(secrets, indent=2)
            encrypted_data = self.fernet.encrypt(json_data.encode())
            
            secrets_file = Path(self.config.secrets_file_path)
            secrets_file.write_bytes(encrypted_data)
            
            # Set restrictive permissions
            secrets_file.chmod(0o600)
            
        except Exception as e:
            logger.error(f"Failed to save secrets file: {e}")
            raise
            
    def _audit_log(self, event: SecretAuditEvent) -> None:
        """Log audit event."""
        if not self.config.enable_audit_logging:
            return
            
        event_data = {
            'event_type': event.event_type,
            'secret_key': event.secret_key,
            'user': event.user,
            'ip_address': event.ip_address,
            'success': event.success,
            'error_message': event.error_message,
            'timestamp': event.timestamp.isoformat(),
            'metadata': event.metadata
        }
        
        self.audit_logger.info(json.dumps(event_data))
        
    # Placeholder implementations for other backends
    def _store_secret_vault(self, secret: Secret) -> None:
        """Store secret in HashiCorp Vault."""
        # Implementation would use hvac client
        pass
        
    def _load_secret_vault(self, key: str) -> Optional[Secret]:
        """Load secret from HashiCorp Vault."""
        # Implementation would use hvac client
        return None
        
    # Additional backend methods would be implemented similarly...


# Factory function for easy initialization
def create_secrets_manager(config: Optional[SecretConfig] = None) -> SecretsManager:
    """Create configured secrets manager instance."""
    config = config or SecretConfig()
    return SecretsManager(config)


# Utility functions
def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Quick function to get a secret value."""
    try:
        manager = create_secrets_manager()
        return manager.get_secret(key)
    except Exception:
        return default