"""
Security Monitoring System

Provides real-time security monitoring, threat detection, and alerting
for the AI knowledge website including intrusion detection, anomaly
detection, and security event correlation.
"""

import json
import logging
import time
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from pathlib import Path
import statistics
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import requests
import psutil


logger = logging.getLogger(__name__)


@dataclass
class MonitorConfig:
    """Configuration for security monitoring."""
    
    # Monitoring settings
    monitoring_enabled: bool = True
    monitoring_interval: int = 60  # seconds
    event_retention_hours: int = 72
    
    # Alert thresholds
    failed_login_threshold: int = 5
    failed_login_window: int = 900  # 15 minutes
    suspicious_ip_threshold: int = 100
    suspicious_ip_window: int = 3600  # 1 hour
    error_rate_threshold: float = 0.05  # 5%
    response_time_threshold: int = 5000  # 5 seconds
    
    # Intrusion detection
    intrusion_detection_enabled: bool = True
    payload_size_threshold: int = 10 * 1024 * 1024  # 10MB
    request_rate_threshold: int = 1000  # requests per minute per IP
    
    # Anomaly detection
    anomaly_detection_enabled: bool = True
    anomaly_sensitivity: float = 2.0  # standard deviations
    baseline_window_hours: int = 24
    
    # Alerting
    email_alerts_enabled: bool = True
    slack_alerts_enabled: bool = False
    webhook_alerts_enabled: bool = False
    
    smtp_host: str = "localhost"
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    alert_email_from: str = "security@ai-knowledge.com"
    alert_email_to: List[str] = field(default_factory=list)
    
    slack_webhook_url: Optional[str] = None
    webhook_url: Optional[str] = None
    
    # Log analysis
    log_analysis_enabled: bool = True
    log_files: List[str] = field(default_factory=lambda: [
        "/var/log/nginx/access.log",
        "/var/log/nginx/error.log",
        ".security/security_events.log"
    ])
    
    # System monitoring
    system_monitoring_enabled: bool = True
    cpu_threshold: float = 80.0  # percentage
    memory_threshold: float = 85.0  # percentage
    disk_threshold: float = 90.0  # percentage


@dataclass
class SecurityEvent:
    """Security event data structure."""
    
    event_id: str
    event_type: str  # INTRUSION, ANOMALY, AUTHENTICATION, SYSTEM, COMPLIANCE
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    title: str
    description: str
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    url: Optional[str] = None
    user_id: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    
@dataclass
class SecurityAlert:
    """Security alert data structure."""
    
    alert_id: str
    event_ids: List[str]
    alert_type: str
    severity: str
    title: str
    description: str
    triggered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged: bool = False
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreatIndicator:
    """Threat indicator for pattern matching."""
    
    indicator_type: str  # IP, USER_AGENT, URL_PATTERN, PAYLOAD_PATTERN
    pattern: str
    severity: str
    description: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: Optional[datetime] = None
    hit_count: int = 0


class SecurityMonitor:
    """
    Real-time security monitoring system.
    
    Features:
    - Intrusion detection and prevention
    - Anomaly detection with baseline learning
    - Real-time threat correlation
    - Automated alerting and notifications
    - System resource monitoring
    - Log analysis and pattern detection
    """
    
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Event storage
        self.events: deque[SecurityEvent] = deque(maxlen=10000)
        self.alerts: List[SecurityAlert] = []
        
        # Threat intelligence
        self.threat_indicators: List[ThreatIndicator] = []
        self.blocked_ips: Set[str] = set()
        self.suspicious_ips: Dict[str, int] = defaultdict(int)
        
        # Anomaly detection baselines
        self.request_baselines: Dict[str, List[float]] = defaultdict(list)
        self.error_baselines: Dict[str, List[float]] = defaultdict(list)
        self.response_time_baselines: Dict[str, List[float]] = defaultdict(list)
        
        # Statistics tracking
        self.request_counts: Dict[str, int] = defaultdict(int)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.response_times: Dict[str, List[float]] = defaultdict(list)
        
        self._load_threat_indicators()
        self._setup_monitoring()
        
    def _load_threat_indicators(self) -> None:
        """Load threat indicators from file or database."""
        # Default threat indicators
        default_indicators = [
            ThreatIndicator(
                indicator_type="USER_AGENT",
                pattern=r"(?i)(sqlmap|nikto|nessus|masscan|nmap)",
                severity="HIGH",
                description="Security scanning tools"
            ),
            ThreatIndicator(
                indicator_type="URL_PATTERN", 
                pattern=r"(?i)(\.\./|\.\.\\|/etc/passwd|/proc/|/sys/)",
                severity="CRITICAL",
                description="Path traversal attempts"
            ),
            ThreatIndicator(
                indicator_type="PAYLOAD_PATTERN",
                pattern=r"(?i)(union\s+select|drop\s+table|<script|javascript:|eval\()",
                severity="HIGH",
                description="Injection attack patterns"
            ),
            ThreatIndicator(
                indicator_type="IP",
                pattern="127.0.0.1",  # Example - would be real threat IPs
                severity="LOW",
                description="Local testing IP"
            )
        ]
        
        self.threat_indicators.extend(default_indicators)
        
    def _setup_monitoring(self) -> None:
        """Setup monitoring components."""
        # Create monitoring directory
        monitoring_dir = Path(".security/monitoring")
        monitoring_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup event logging
        self.event_logger = logging.getLogger("security_events")
        self.event_logger.setLevel(logging.INFO)
        
        event_handler = logging.FileHandler(monitoring_dir / "security_events.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        event_handler.setFormatter(formatter)
        self.event_logger.addHandler(event_handler)
        
    def start_monitoring(self) -> None:
        """Start the security monitoring system."""
        if self.running:
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Security monitoring started")
        
    def stop_monitoring(self) -> None:
        """Stop the security monitoring system."""
        self.running = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
            
        logger.info("Security monitoring stopped")
        
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                # System resource monitoring
                if self.config.system_monitoring_enabled:
                    self._monitor_system_resources()
                    
                # Anomaly detection
                if self.config.anomaly_detection_enabled:
                    self._detect_anomalies()
                    
                # Process alerts
                self._process_alerts()
                
                # Cleanup old events
                self._cleanup_old_events()
                
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)  # Wait before retrying
                
    def log_security_event(self, 
                          event_type: str,
                          severity: str,
                          title: str,
                          description: str,
                          source_ip: Optional[str] = None,
                          user_agent: Optional[str] = None,
                          url: Optional[str] = None,
                          user_id: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> SecurityEvent:
        """
        Log a security event.
        
        Args:
            event_type: Type of security event
            severity: Event severity (LOW, MEDIUM, HIGH, CRITICAL)
            title: Brief event title
            description: Detailed description
            source_ip: Source IP address if applicable
            user_agent: User agent string if applicable
            url: URL involved in the event
            user_id: User ID if applicable
            metadata: Additional event metadata
            
        Returns:
            Created security event
        """
        event_id = self._generate_event_id()
        
        event = SecurityEvent(
            event_id=event_id,
            event_type=event_type,
            severity=severity,
            title=title,
            description=description,
            source_ip=source_ip,
            user_agent=user_agent,
            url=url,
            user_id=user_id,
            metadata=metadata or {}
        )
        
        # Add to event storage
        self.events.append(event)
        
        # Log to file
        event_data = {
            'event_id': event_id,
            'event_type': event_type,
            'severity': severity,
            'title': title,
            'description': description,
            'source_ip': source_ip,
            'user_agent': user_agent,
            'url': url,
            'user_id': user_id,
            'timestamp': event.timestamp.isoformat(),
            'metadata': metadata
        }
        
        self.event_logger.info(json.dumps(event_data))
        
        # Check for threat indicators
        self._check_threat_indicators(event)
        
        # Check for intrusion patterns
        if self.config.intrusion_detection_enabled:
            self._check_intrusion_patterns(event)
            
        # Auto-generate alerts for critical events
        if severity == "CRITICAL":
            self._create_alert([event_id], "CRITICAL_EVENT", severity, title, description)
            
        logger.info(f"Security event logged: {event_type} - {title}")
        
        return event
        
    def _check_threat_indicators(self, event: SecurityEvent) -> None:
        """Check event against threat indicators."""
        for indicator in self.threat_indicators:
            match_found = False
            
            if indicator.indicator_type == "IP" and event.source_ip:
                if indicator.pattern == event.source_ip:
                    match_found = True
                    
            elif indicator.indicator_type == "USER_AGENT" and event.user_agent:
                import re
                if re.search(indicator.pattern, event.user_agent):
                    match_found = True
                    
            elif indicator.indicator_type == "URL_PATTERN" and event.url:
                import re
                if re.search(indicator.pattern, event.url):
                    match_found = True
                    
            elif indicator.indicator_type == "PAYLOAD_PATTERN":
                # Check all string fields
                search_text = f"{event.description} {event.metadata}"
                import re
                if re.search(indicator.pattern, search_text):
                    match_found = True
                    
            if match_found:
                indicator.hit_count += 1
                indicator.last_seen = datetime.now(timezone.utc)
                
                # Create alert for threat indicator match
                self._create_alert(
                    [event.event_id],
                    "THREAT_INDICATOR",
                    indicator.severity,
                    f"Threat indicator detected: {indicator.description}",
                    f"Event {event.event_id} matched threat indicator: {indicator.pattern}"
                )
                
                # Auto-block IPs for critical threats
                if (indicator.severity == "CRITICAL" and 
                    indicator.indicator_type == "IP" and 
                    event.source_ip):
                    self.blocked_ips.add(event.source_ip)
                    
    def _check_intrusion_patterns(self, event: SecurityEvent) -> None:
        """Check for intrusion detection patterns."""
        if not event.source_ip:
            return
            
        # Track failed login attempts
        if event.event_type == "AUTHENTICATION" and "failed" in event.description.lower():
            key = f"failed_login_{event.source_ip}"
            count = self._increment_counter(key, self.config.failed_login_window)
            
            if count >= self.config.failed_login_threshold:
                self._create_alert(
                    [event.event_id],
                    "BRUTE_FORCE",
                    "HIGH",
                    f"Brute force attack detected from {event.source_ip}",
                    f"IP {event.source_ip} has {count} failed login attempts in {self.config.failed_login_window} seconds"
                )
                
                # Add to suspicious IPs
                self.suspicious_ips[event.source_ip] += count
                
        # Track request rates per IP
        if event.source_ip:
            key = f"requests_{event.source_ip}"
            count = self._increment_counter(key, 60)  # 1 minute window
            
            if count >= self.config.request_rate_threshold:
                self._create_alert(
                    [event.event_id],
                    "RATE_LIMIT_EXCEEDED",
                    "MEDIUM",
                    f"High request rate from {event.source_ip}",
                    f"IP {event.source_ip} has {count} requests in the last minute"
                )
                
    def _increment_counter(self, key: str, window: int) -> int:
        """Increment time-windowed counter."""
        now = time.time()
        
        if not hasattr(self, '_counters'):
            self._counters = defaultdict(list)
            
        # Remove old entries outside the window
        self._counters[key] = [
            timestamp for timestamp in self._counters[key]
            if now - timestamp <= window
        ]
        
        # Add current timestamp
        self._counters[key].append(now)
        
        return len(self._counters[key])
        
    def _monitor_system_resources(self) -> None:
        """Monitor system resource usage."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > self.config.cpu_threshold:
                self.log_security_event(
                    "SYSTEM",
                    "MEDIUM",
                    "High CPU usage",
                    f"CPU usage at {cpu_percent:.1f}%",
                    metadata={'cpu_percent': cpu_percent}
                )
                
            # Memory usage
            memory = psutil.virtual_memory()
            if memory.percent > self.config.memory_threshold:
                self.log_security_event(
                    "SYSTEM",
                    "MEDIUM",
                    "High memory usage",
                    f"Memory usage at {memory.percent:.1f}%",
                    metadata={'memory_percent': memory.percent}
                )
                
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            if disk_percent > self.config.disk_threshold:
                self.log_security_event(
                    "SYSTEM",
                    "HIGH",
                    "High disk usage",
                    f"Disk usage at {disk_percent:.1f}%",
                    metadata={'disk_percent': disk_percent}
                )
                
        except Exception as e:
            logger.error(f"Error monitoring system resources: {e}")
            
    def _detect_anomalies(self) -> None:
        """Detect anomalies in system behavior."""
        try:
            now = datetime.now(timezone.utc)
            
            # Check request rate anomalies
            current_hour = now.hour
            recent_requests = self.request_counts.get(str(current_hour), 0)
            
            if len(self.request_baselines[str(current_hour)]) >= 7:  # Need at least a week of data
                baseline_mean = statistics.mean(self.request_baselines[str(current_hour)])
                baseline_std = statistics.stdev(self.request_baselines[str(current_hour)])
                
                # Check if current rate is anomalous
                if abs(recent_requests - baseline_mean) > self.config.anomaly_sensitivity * baseline_std:
                    severity = "HIGH" if recent_requests > baseline_mean * 2 else "MEDIUM"
                    
                    self.log_security_event(
                        "ANOMALY",
                        severity,
                        "Request rate anomaly detected",
                        f"Request rate {recent_requests} deviates from baseline {baseline_mean:.1f}±{baseline_std:.1f}",
                        metadata={
                            'current_requests': recent_requests,
                            'baseline_mean': baseline_mean,
                            'baseline_std': baseline_std
                        }
                    )
                    
            # Update baselines
            self.request_baselines[str(current_hour)].append(recent_requests)
            if len(self.request_baselines[str(current_hour)]) > 30:  # Keep last 30 days
                self.request_baselines[str(current_hour)].pop(0)
                
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            
    def _create_alert(self, 
                     event_ids: List[str],
                     alert_type: str,
                     severity: str,
                     title: str,
                     description: str,
                     metadata: Optional[Dict[str, Any]] = None) -> SecurityAlert:
        """Create a security alert."""
        alert_id = self._generate_alert_id()
        
        alert = SecurityAlert(
            alert_id=alert_id,
            event_ids=event_ids,
            alert_type=alert_type,
            severity=severity,
            title=title,
            description=description,
            metadata=metadata or {}
        )
        
        self.alerts.append(alert)
        
        # Send notifications
        self._send_alert_notifications(alert)
        
        logger.warning(f"Security alert created: {alert_type} - {title}")
        
        return alert
        
    def _send_alert_notifications(self, alert: SecurityAlert) -> None:
        """Send alert notifications via configured channels."""
        try:
            # Email notifications
            if self.config.email_alerts_enabled and self.config.alert_email_to:
                self._send_email_alert(alert)
                
            # Slack notifications
            if self.config.slack_alerts_enabled and self.config.slack_webhook_url:
                self._send_slack_alert(alert)
                
            # Webhook notifications
            if self.config.webhook_alerts_enabled and self.config.webhook_url:
                self._send_webhook_alert(alert)
                
        except Exception as e:
            logger.error(f"Error sending alert notifications: {e}")
            
    def _send_email_alert(self, alert: SecurityAlert) -> None:
        """Send email alert."""
        try:
            msg = MimeMultipart()
            msg['From'] = self.config.alert_email_from
            msg['To'] = ', '.join(self.config.alert_email_to)
            msg['Subject'] = f"[SECURITY ALERT] {alert.severity}: {alert.title}"
            
            body = f"""
Security Alert Details:

Alert ID: {alert.alert_id}
Type: {alert.alert_type}
Severity: {alert.severity}
Triggered: {alert.triggered_at.isoformat()}

Title: {alert.title}

Description:
{alert.description}

Event IDs: {', '.join(alert.event_ids)}

Metadata:
{json.dumps(alert.metadata, indent=2)}

This is an automated security alert from the AI Knowledge Website monitoring system.
"""
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(self.config.smtp_host, self.config.smtp_port)
            if self.config.smtp_user and self.config.smtp_password:
                server.starttls()
                server.login(self.config.smtp_user, self.config.smtp_password)
                
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent for {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            
    def _send_slack_alert(self, alert: SecurityAlert) -> None:
        """Send Slack alert."""
        try:
            color = {
                'LOW': '#36a64f',
                'MEDIUM': '#ff9500', 
                'HIGH': '#ff0000',
                'CRITICAL': '#8B0000'
            }.get(alert.severity, '#808080')
            
            payload = {
                'attachments': [
                    {
                        'color': color,
                        'title': f"Security Alert: {alert.title}",
                        'text': alert.description,
                        'fields': [
                            {
                                'title': 'Alert ID',
                                'value': alert.alert_id,
                                'short': True
                            },
                            {
                                'title': 'Type',
                                'value': alert.alert_type,
                                'short': True
                            },
                            {
                                'title': 'Severity',
                                'value': alert.severity,
                                'short': True
                            },
                            {
                                'title': 'Triggered',
                                'value': alert.triggered_at.isoformat(),
                                'short': True
                            }
                        ]
                    }
                ]
            }
            
            response = requests.post(
                self.config.slack_webhook_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
            logger.info(f"Slack alert sent for {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            
    def _send_webhook_alert(self, alert: SecurityAlert) -> None:
        """Send webhook alert."""
        try:
            payload = {
                'alert_id': alert.alert_id,
                'alert_type': alert.alert_type,
                'severity': alert.severity,
                'title': alert.title,
                'description': alert.description,
                'triggered_at': alert.triggered_at.isoformat(),
                'event_ids': alert.event_ids,
                'metadata': alert.metadata
            }
            
            response = requests.post(
                self.config.webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            response.raise_for_status()
            
            logger.info(f"Webhook alert sent for {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            
    def _process_alerts(self) -> None:
        """Process and correlate alerts."""
        try:
            # Group recent similar alerts
            now = datetime.now(timezone.utc)
            recent_alerts = [
                alert for alert in self.alerts
                if (now - alert.triggered_at).total_seconds() < 3600  # Last hour
                and not alert.resolved
            ]
            
            # Correlate alerts by type and source IP
            alert_groups = defaultdict(list)
            for alert in recent_alerts:
                # Get source IPs from related events
                source_ips = set()
                for event_id in alert.event_ids:
                    event = self._get_event_by_id(event_id)
                    if event and event.source_ip:
                        source_ips.add(event.source_ip)
                        
                key = (alert.alert_type, tuple(sorted(source_ips)))
                alert_groups[key].append(alert)
                
            # Create correlated alerts for groups with multiple alerts
            for (alert_type, source_ips), alerts in alert_groups.items():
                if len(alerts) >= 3 and source_ips:  # 3+ related alerts
                    correlated_alert_exists = any(
                        alert.alert_type == "CORRELATED_ATTACK" 
                        and set(alert.metadata.get('source_ips', [])) == set(source_ips)
                        for alert in self.alerts
                        if (now - alert.triggered_at).total_seconds() < 3600
                    )
                    
                    if not correlated_alert_exists:
                        event_ids = [eid for alert in alerts for eid in alert.event_ids]
                        
                        self._create_alert(
                            event_ids,
                            "CORRELATED_ATTACK",
                            "CRITICAL",
                            f"Coordinated attack detected: {alert_type}",
                            f"Multiple {alert_type} alerts from IPs: {', '.join(source_ips)}",
                            metadata={'source_ips': list(source_ips), 'alert_count': len(alerts)}
                        )
                        
        except Exception as e:
            logger.error(f"Error processing alerts: {e}")
            
    def _cleanup_old_events(self) -> None:
        """Clean up old events and alerts."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.config.event_retention_hours)
        
        # Clean up events (deque handles this automatically with maxlen)
        
        # Clean up alerts
        self.alerts = [
            alert for alert in self.alerts
            if alert.triggered_at > cutoff_time or not alert.resolved
        ]
        
    def _get_event_by_id(self, event_id: str) -> Optional[SecurityEvent]:
        """Get event by ID."""
        for event in self.events:
            if event.event_id == event_id:
                return event
        return None
        
    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        import uuid
        return f"evt_{uuid.uuid4().hex[:8]}"
        
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID."""
        import uuid
        return f"alt_{uuid.uuid4().hex[:8]}"
        
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get security monitoring dashboard data."""
        now = datetime.now(timezone.utc)
        
        # Recent events summary
        recent_events = [
            event for event in self.events
            if (now - event.timestamp).total_seconds() < 3600  # Last hour
        ]
        
        events_by_type = defaultdict(int)
        events_by_severity = defaultdict(int)
        
        for event in recent_events:
            events_by_type[event.event_type] += 1
            events_by_severity[event.severity] += 1
            
        # Active alerts
        active_alerts = [
            alert for alert in self.alerts
            if not alert.resolved and (now - alert.triggered_at).total_seconds() < 86400  # Last 24 hours
        ]
        
        # Top suspicious IPs
        top_suspicious_ips = sorted(
            self.suspicious_ips.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # System status
        try:
            system_status = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': (psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100
            }
        except Exception:
            system_status = {'error': 'Unable to get system stats'}
            
        return {
            'monitoring_status': {
                'enabled': self.config.monitoring_enabled,
                'running': self.running,
                'event_count': len(self.events),
                'alert_count': len(self.alerts)
            },
            'recent_events': {
                'total': len(recent_events),
                'by_type': dict(events_by_type),
                'by_severity': dict(events_by_severity)
            },
            'active_alerts': len(active_alerts),
            'threat_indicators': {
                'total': len(self.threat_indicators),
                'hits': sum(indicator.hit_count for indicator in self.threat_indicators)
            },
            'blocked_ips': len(self.blocked_ips),
            'suspicious_ips': dict(top_suspicious_ips),
            'system_status': system_status,
            'last_updated': now.isoformat()
        }
        
    def acknowledge_alert(self, alert_id: str, user: str = "system") -> bool:
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                
                self.log_security_event(
                    "SYSTEM",
                    "INFO",
                    f"Alert acknowledged: {alert_id}",
                    f"Alert {alert_id} acknowledged by {user}",
                    user_id=user,
                    metadata={'alert_id': alert_id}
                )
                
                return True
                
        return False
        
    def resolve_alert(self, alert_id: str, user: str = "system") -> bool:
        """Resolve an alert."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.acknowledged = True
                
                self.log_security_event(
                    "SYSTEM",
                    "INFO",
                    f"Alert resolved: {alert_id}",
                    f"Alert {alert_id} resolved by {user}",
                    user_id=user,
                    metadata={'alert_id': alert_id}
                )
                
                return True
                
        return False
        
    def add_threat_indicator(self, 
                           indicator_type: str,
                           pattern: str,
                           severity: str,
                           description: str) -> ThreatIndicator:
        """Add new threat indicator."""
        indicator = ThreatIndicator(
            indicator_type=indicator_type,
            pattern=pattern,
            severity=severity,
            description=description
        )
        
        self.threat_indicators.append(indicator)
        
        self.log_security_event(
            "SYSTEM",
            "INFO",
            "Threat indicator added",
            f"New {indicator_type} threat indicator: {pattern}",
            metadata={
                'indicator_type': indicator_type,
                'pattern': pattern,
                'severity': severity
            }
        )
        
        return indicator


# Factory function
def create_security_monitor(config: Optional[MonitorConfig] = None) -> SecurityMonitor:
    """Create configured security monitor instance."""
    config = config or MonitorConfig()
    return SecurityMonitor(config)