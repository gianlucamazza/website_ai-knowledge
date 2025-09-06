"""
Authentication and Authorization Middleware

Provides comprehensive authentication and authorization for the AI knowledge website,
including API token management, session handling, and role-based access control.
"""

import hashlib
import hmac
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import dataclass, field
from functools import wraps
import jwt
from cryptography.fernet import Fernet
from werkzeug.security import generate_password_hash, check_password_hash
import redis


logger = logging.getLogger(__name__)


@dataclass
class AuthConfig:
    """Configuration for authentication and authorization."""
    
    # JWT Configuration
    jwt_secret_key: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    jwt_refresh_expiration_days: int = 30
    
    # Session Configuration
    session_secret_key: str = ""
    session_timeout_minutes: int = 60
    max_concurrent_sessions: int = 5
    
    # API Key Configuration
    api_key_length: int = 64
    api_key_prefix: str = "ak_"
    
    # Password Requirements
    min_password_length: int = 12
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_digits: bool = True
    require_special_chars: bool = True
    max_password_age_days: int = 90
    
    # Rate Limiting
    max_login_attempts: int = 5
    login_lockout_minutes: int = 15
    max_requests_per_minute: int = 100
    
    # Redis Configuration (for session storage)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # Security Settings
    enable_mfa: bool = False
    mfa_secret_length: int = 32
    enable_session_ip_validation: bool = True
    enable_user_agent_validation: bool = True
    
    # Allowed roles and permissions
    roles: Dict[str, List[str]] = field(default_factory=lambda: {
        'admin': ['read', 'write', 'delete', 'manage_users', 'view_analytics'],
        'editor': ['read', 'write'],
        'viewer': ['read'],
        'api_user': ['api_access', 'read'],
        'pipeline': ['pipeline_execute', 'read', 'write']
    })


class AuthenticationError(Exception):
    """Authentication-related errors."""
    pass


class AuthorizationError(Exception):
    """Authorization-related errors."""
    pass


class RateLimitExceeded(Exception):
    """Rate limit exceeded error."""
    pass


@dataclass
class User:
    """User data structure."""
    
    user_id: str
    username: str
    email: str
    password_hash: str
    roles: List[str]
    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_login: Optional[datetime] = None
    login_attempts: int = 0
    locked_until: Optional[datetime] = None
    password_changed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None


@dataclass
class Session:
    """Session data structure."""
    
    session_id: str
    user_id: str
    created_at: datetime
    last_accessed: datetime
    expires_at: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_active: bool = True


@dataclass
class APIKey:
    """API Key data structure."""
    
    key_id: str
    key_hash: str
    name: str
    user_id: str
    permissions: List[str]
    created_at: datetime
    expires_at: Optional[datetime] = None
    is_active: bool = True
    last_used: Optional[datetime] = None
    usage_count: int = 0


class AuthMiddleware:
    """
    Comprehensive authentication and authorization middleware.
    
    Features:
    - JWT token management
    - Session management with Redis
    - API key authentication
    - Role-based access control
    - Rate limiting
    - Multi-factor authentication
    - Security monitoring
    """
    
    def __init__(self, config: AuthConfig):
        self.config = config
        self._setup_crypto()
        self._setup_redis()
        self._setup_rate_limiting()
        
        # In-memory stores (should be replaced with database in production)
        self.users: Dict[str, User] = {}
        self.api_keys: Dict[str, APIKey] = {}
        self.sessions: Dict[str, Session] = {}
        
    def _setup_crypto(self) -> None:
        """Setup cryptographic components."""
        if not self.config.jwt_secret_key:
            raise ValueError("JWT secret key must be configured")
            
        if not self.config.session_secret_key:
            raise ValueError("Session secret key must be configured")
            
        # Setup Fernet for symmetric encryption
        key = hashlib.sha256(self.config.session_secret_key.encode()).digest()
        self.fernet = Fernet(Fernet.generate_key())
        
    def _setup_redis(self) -> None:
        """Setup Redis connection for session storage."""
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                decode_responses=True
            )
            self.redis_client.ping()
        except Exception as e:
            logger.warning(f"Redis connection failed, using in-memory storage: {e}")
            self.redis_client = None
            
    def _setup_rate_limiting(self) -> None:
        """Setup rate limiting tracking."""
        self.rate_limit_tracker: Dict[str, List[float]] = {}
        
    def create_user(self, 
                   username: str, 
                   email: str, 
                   password: str, 
                   roles: List[str]) -> User:
        """
        Create a new user.
        
        Args:
            username: Unique username
            email: User email address
            password: Plain text password
            roles: List of roles to assign
            
        Returns:
            Created user object
            
        Raises:
            ValueError: If user data is invalid
        """
        # Validate password strength
        self._validate_password(password)
        
        # Check if user already exists
        for user in self.users.values():
            if user.username == username or user.email == email:
                raise ValueError("User already exists")
                
        # Validate roles
        invalid_roles = set(roles) - set(self.config.roles.keys())
        if invalid_roles:
            raise ValueError(f"Invalid roles: {invalid_roles}")
            
        user_id = self._generate_user_id()
        password_hash = generate_password_hash(password)
        
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            roles=roles
        )
        
        self.users[user_id] = user
        logger.info(f"Created user: {username}")
        
        return user
        
    def authenticate_user(self, 
                         username: str, 
                         password: str,
                         ip_address: Optional[str] = None) -> Dict[str, str]:
        """
        Authenticate user with username/password.
        
        Args:
            username: Username or email
            password: Plain text password
            ip_address: Client IP address
            
        Returns:
            Dictionary with access and refresh tokens
            
        Raises:
            AuthenticationError: If authentication fails
        """
        # Find user
        user = None
        for u in self.users.values():
            if u.username == username or u.email == username:
                user = u
                break
                
        if not user:
            logger.warning(f"Login attempt for non-existent user: {username}")
            raise AuthenticationError("Invalid credentials")
            
        # Check if account is locked
        if user.locked_until and user.locked_until > datetime.now(timezone.utc):
            raise AuthenticationError("Account temporarily locked due to too many failed attempts")
            
        # Check if account is active
        if not user.is_active:
            raise AuthenticationError("Account is disabled")
            
        # Verify password
        if not check_password_hash(user.password_hash, password):
            user.login_attempts += 1
            
            # Lock account if too many attempts
            if user.login_attempts >= self.config.max_login_attempts:
                user.locked_until = datetime.now(timezone.utc) + timedelta(
                    minutes=self.config.login_lockout_minutes
                )
                logger.warning(f"Account locked for user: {username}")
                
            raise AuthenticationError("Invalid credentials")
            
        # Reset login attempts on successful login
        user.login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.now(timezone.utc)
        
        # Generate tokens
        access_token = self._generate_jwt(user, is_refresh=False)
        refresh_token = self._generate_jwt(user, is_refresh=True)
        
        # Create session
        session = self._create_session(user.user_id, ip_address)
        
        logger.info(f"User authenticated: {username}")
        
        return {
            'access_token': access_token,
            'refresh_token': refresh_token,
            'session_id': session.session_id,
            'expires_at': session.expires_at.isoformat()
        }
        
    def create_api_key(self, 
                      user_id: str, 
                      name: str, 
                      permissions: List[str],
                      expires_in_days: Optional[int] = None) -> Dict[str, str]:
        """
        Create API key for a user.
        
        Args:
            user_id: User ID
            name: Descriptive name for the API key
            permissions: List of permissions
            expires_in_days: Expiration in days (None for no expiration)
            
        Returns:
            Dictionary with API key information
            
        Raises:
            ValueError: If user or permissions are invalid
        """
        if user_id not in self.users:
            raise ValueError("User not found")
            
        user = self.users[user_id]
        
        # Validate permissions against user roles
        allowed_permissions = set()
        for role in user.roles:
            allowed_permissions.update(self.config.roles.get(role, []))
            
        invalid_permissions = set(permissions) - allowed_permissions
        if invalid_permissions:
            raise ValueError(f"User does not have permissions: {invalid_permissions}")
            
        # Generate API key
        key_id = self._generate_key_id()
        api_key = self._generate_api_key()
        key_hash = self._hash_api_key(api_key)
        
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)
            
        api_key_obj = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            user_id=user_id,
            permissions=permissions,
            created_at=datetime.now(timezone.utc),
            expires_at=expires_at
        )
        
        self.api_keys[key_id] = api_key_obj
        
        logger.info(f"Created API key '{name}' for user: {user.username}")
        
        return {
            'key_id': key_id,
            'api_key': api_key,
            'name': name,
            'permissions': permissions,
            'expires_at': expires_at.isoformat() if expires_at else None
        }
        
    def verify_jwt(self, token: str) -> Dict[str, Any]:
        """
        Verify JWT token.
        
        Args:
            token: JWT token to verify
            
        Returns:
            Decoded token payload
            
        Raises:
            AuthenticationError: If token is invalid
        """
        try:
            payload = jwt.decode(
                token,
                self.config.jwt_secret_key,
                algorithms=[self.config.jwt_algorithm]
            )
            
            # Verify user still exists and is active
            user_id = payload.get('user_id')
            if user_id not in self.users or not self.users[user_id].is_active:
                raise AuthenticationError("User no longer active")
                
            return payload
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid token")
            
    def verify_api_key(self, api_key: str) -> APIKey:
        """
        Verify API key.
        
        Args:
            api_key: API key to verify
            
        Returns:
            API key object
            
        Raises:
            AuthenticationError: If API key is invalid
        """
        key_hash = self._hash_api_key(api_key)
        
        # Find API key
        api_key_obj = None
        for key_obj in self.api_keys.values():
            if key_obj.key_hash == key_hash:
                api_key_obj = key_obj
                break
                
        if not api_key_obj:
            raise AuthenticationError("Invalid API key")
            
        # Check if key is active
        if not api_key_obj.is_active:
            raise AuthenticationError("API key is disabled")
            
        # Check expiration
        if api_key_obj.expires_at and api_key_obj.expires_at < datetime.now(timezone.utc):
            raise AuthenticationError("API key has expired")
            
        # Update usage tracking
        api_key_obj.last_used = datetime.now(timezone.utc)
        api_key_obj.usage_count += 1
        
        return api_key_obj
        
    def check_permission(self, user_id: str, required_permission: str) -> bool:
        """
        Check if user has required permission.
        
        Args:
            user_id: User ID
            required_permission: Permission to check
            
        Returns:
            True if user has permission
        """
        if user_id not in self.users:
            return False
            
        user = self.users[user_id]
        
        # Get all permissions for user's roles
        user_permissions = set()
        for role in user.roles:
            user_permissions.update(self.config.roles.get(role, []))
            
        return required_permission in user_permissions
        
    def check_rate_limit(self, 
                        identifier: str, 
                        max_requests: Optional[int] = None) -> bool:
        """
        Check rate limit for identifier (IP, user, API key).
        
        Args:
            identifier: Unique identifier
            max_requests: Max requests per minute (uses config default if None)
            
        Returns:
            True if within rate limit
            
        Raises:
            RateLimitExceeded: If rate limit exceeded
        """
        max_requests = max_requests or self.config.max_requests_per_minute
        current_time = time.time()
        window_start = current_time - 60  # 1 minute window
        
        # Get request history for identifier
        if identifier not in self.rate_limit_tracker:
            self.rate_limit_tracker[identifier] = []
            
        requests = self.rate_limit_tracker[identifier]
        
        # Remove old requests outside the window
        requests[:] = [req_time for req_time in requests if req_time > window_start]
        
        # Check if limit exceeded
        if len(requests) >= max_requests:
            logger.warning(f"Rate limit exceeded for: {identifier}")
            raise RateLimitExceeded(f"Rate limit exceeded: {len(requests)}/{max_requests}")
            
        # Add current request
        requests.append(current_time)
        
        return True
        
    def require_auth(self, required_permission: Optional[str] = None):
        """
        Decorator for requiring authentication and optionally specific permissions.
        
        Args:
            required_permission: Optional permission requirement
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # This would typically extract token from request headers
                # For now, we'll assume token is passed as a keyword argument
                token = kwargs.get('auth_token')
                if not token:
                    raise AuthenticationError("Authentication required")
                    
                # Verify token
                payload = self.verify_jwt(token)
                user_id = payload['user_id']
                
                # Check permission if required
                if required_permission and not self.check_permission(user_id, required_permission):
                    raise AuthorizationError(f"Permission required: {required_permission}")
                    
                # Add user context to kwargs
                kwargs['current_user'] = self.users[user_id]
                
                return func(*args, **kwargs)
            return wrapper
        return decorator
        
    def _generate_jwt(self, user: User, is_refresh: bool = False) -> str:
        """Generate JWT token for user."""
        now = datetime.now(timezone.utc)
        
        if is_refresh:
            exp = now + timedelta(days=self.config.jwt_refresh_expiration_days)
            token_type = "refresh"
        else:
            exp = now + timedelta(hours=self.config.jwt_expiration_hours)
            token_type = "access"
            
        payload = {
            'user_id': user.user_id,
            'username': user.username,
            'roles': user.roles,
            'iat': now.timestamp(),
            'exp': exp.timestamp(),
            'type': token_type
        }
        
        return jwt.encode(
            payload,
            self.config.jwt_secret_key,
            algorithm=self.config.jwt_algorithm
        )
        
    def _create_session(self, user_id: str, ip_address: Optional[str] = None) -> Session:
        """Create new session for user."""
        session_id = self._generate_session_id()
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(minutes=self.config.session_timeout_minutes)
        
        session = Session(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            last_accessed=now,
            expires_at=expires_at,
            ip_address=ip_address
        )
        
        self.sessions[session_id] = session
        
        # Store in Redis if available
        if self.redis_client:
            try:
                session_data = {
                    'user_id': user_id,
                    'created_at': now.isoformat(),
                    'ip_address': ip_address or ''
                }
                self.redis_client.setex(
                    f"session:{session_id}",
                    int(self.config.session_timeout_minutes * 60),
                    json.dumps(session_data)
                )
            except Exception as e:
                logger.warning(f"Failed to store session in Redis: {e}")
                
        return session
        
    def _validate_password(self, password: str) -> None:
        """Validate password meets requirements."""
        if len(password) < self.config.min_password_length:
            raise ValueError(f"Password must be at least {self.config.min_password_length} characters")
            
        if self.config.require_uppercase and not any(c.isupper() for c in password):
            raise ValueError("Password must contain uppercase letters")
            
        if self.config.require_lowercase and not any(c.islower() for c in password):
            raise ValueError("Password must contain lowercase letters")
            
        if self.config.require_digits and not any(c.isdigit() for c in password):
            raise ValueError("Password must contain digits")
            
        if self.config.require_special_chars and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            raise ValueError("Password must contain special characters")
            
        # Check for common weak passwords
        weak_passwords = {'password', '12345678', 'qwerty', 'admin', 'password123'}
        if password.lower() in weak_passwords:
            raise ValueError("Password is too common")
            
    def _generate_user_id(self) -> str:
        """Generate unique user ID."""
        import uuid
        return str(uuid.uuid4())
        
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        import uuid
        return str(uuid.uuid4())
        
    def _generate_key_id(self) -> str:
        """Generate unique key ID."""
        import uuid
        return str(uuid.uuid4())
        
    def _generate_api_key(self) -> str:
        """Generate API key."""
        import secrets
        key = secrets.token_urlsafe(self.config.api_key_length)
        return f"{self.config.api_key_prefix}{key}"
        
    def _hash_api_key(self, api_key: str) -> str:
        """Hash API key for secure storage."""
        return hashlib.sha256(api_key.encode()).hexdigest()
        
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics and statistics."""
        now = datetime.now(timezone.utc)
        
        # Count active sessions
        active_sessions = sum(1 for s in self.sessions.values() if s.expires_at > now)
        
        # Count active API keys
        active_api_keys = sum(1 for k in self.api_keys.values() if k.is_active)
        
        # Count locked users
        locked_users = sum(1 for u in self.users.values() 
                          if u.locked_until and u.locked_until > now)
        
        return {
            'total_users': len(self.users),
            'active_users': sum(1 for u in self.users.values() if u.is_active),
            'locked_users': locked_users,
            'active_sessions': active_sessions,
            'total_api_keys': len(self.api_keys),
            'active_api_keys': active_api_keys,
            'rate_limit_tracked_identifiers': len(self.rate_limit_tracker)
        }


# Factory function for easy initialization
def create_auth_middleware(config: Optional[AuthConfig] = None) -> AuthMiddleware:
    """Create configured auth middleware instance."""
    config = config or AuthConfig()
    return AuthMiddleware(config)