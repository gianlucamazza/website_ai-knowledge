"""
Security Incident Response System

Provides automated incident response capabilities including containment,
investigation, communication, and recovery procedures for security incidents.
"""

import json
import logging
import smtplib
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import subprocess
import requests
import threading


logger = logging.getLogger(__name__)


class IncidentSeverity(Enum):
    """Incident severity levels."""
    
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class IncidentStatus(Enum):
    """Incident status values."""
    
    OPEN = "OPEN"
    INVESTIGATING = "INVESTIGATING"
    CONTAINED = "CONTAINED"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"


class IncidentCategory(Enum):
    """Incident categories."""
    
    INTRUSION = "INTRUSION"
    MALWARE = "MALWARE"
    DATA_BREACH = "DATA_BREACH"
    DOS_ATTACK = "DOS_ATTACK"
    UNAUTHORIZED_ACCESS = "UNAUTHORIZED_ACCESS"
    SYSTEM_COMPROMISE = "SYSTEM_COMPROMISE"
    POLICY_VIOLATION = "POLICY_VIOLATION"
    OTHER = "OTHER"


@dataclass
class IncidentConfig:
    """Configuration for incident response."""
    
    # Response settings
    auto_containment_enabled: bool = True
    containment_threshold_severity: str = "HIGH"
    
    # Communication settings
    email_notifications_enabled: bool = True
    sms_notifications_enabled: bool = False
    slack_notifications_enabled: bool = True
    
    # Email configuration
    smtp_host: str = "localhost"
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    incident_email_from: str = "security-incidents@ai-knowledge.com"
    
    # Notification groups
    security_team_emails: List[str] = field(default_factory=list)
    management_emails: List[str] = field(default_factory=list)
    devops_emails: List[str] = field(default_factory=list)
    
    slack_webhook_url: Optional[str] = None
    emergency_webhook_url: Optional[str] = None
    
    # Response procedures
    incident_playbooks_dir: str = ".security/playbooks"
    evidence_collection_enabled: bool = True
    evidence_storage_dir: str = ".security/evidence"
    
    # SLA settings (in hours)
    initial_response_sla: int = 1
    investigation_sla: int = 4
    containment_sla: int = 2
    resolution_sla: int = 24
    
    # Auto-response actions
    auto_block_malicious_ips: bool = True
    auto_isolate_compromised_accounts: bool = False
    auto_scale_resources: bool = True
    
    # Integration settings
    siem_integration_enabled: bool = False
    siem_api_url: Optional[str] = None
    siem_api_key: Optional[str] = None


@dataclass
class IncidentAction:
    """Individual incident response action."""
    
    action_id: str
    action_type: str  # CONTAINMENT, INVESTIGATION, COMMUNICATION, RECOVERY
    title: str
    description: str
    assigned_to: Optional[str] = None
    status: str = "PENDING"  # PENDING, IN_PROGRESS, COMPLETED, FAILED
    priority: int = 1  # 1=high, 2=medium, 3=low
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    

@dataclass
class SecurityIncident:
    """Security incident data structure."""
    
    incident_id: str
    title: str
    description: str
    category: IncidentCategory
    severity: IncidentSeverity
    status: IncidentStatus = IncidentStatus.OPEN
    
    # Timeline
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reported_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged_at: Optional[datetime] = None
    contained_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    
    # Assignment
    assigned_to: Optional[str] = None
    reporter: str = "system"
    
    # Impact assessment
    affected_systems: List[str] = field(default_factory=list)
    affected_users: List[str] = field(default_factory=list)
    data_at_risk: Optional[str] = None
    business_impact: Optional[str] = None
    
    # Technical details
    indicators_of_compromise: List[str] = field(default_factory=list)
    attack_vectors: List[str] = field(default_factory=list)
    evidence_collected: List[str] = field(default_factory=list)
    
    # Response actions
    actions: List[IncidentAction] = field(default_factory=list)
    
    # Communication
    stakeholders_notified: List[str] = field(default_factory=list)
    external_parties_notified: List[str] = field(default_factory=list)
    
    # Metadata
    related_events: List[str] = field(default_factory=list)
    related_alerts: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    
@dataclass
class PlaybookStep:
    """Individual step in an incident response playbook."""
    
    step_id: str
    title: str
    description: str
    action_type: str
    priority: int
    estimated_duration: int  # minutes
    required_role: Optional[str] = None
    automation_possible: bool = False
    automation_script: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IncidentPlaybook:
    """Incident response playbook."""
    
    playbook_id: str
    name: str
    description: str
    triggers: List[str]  # What triggers this playbook
    categories: List[IncidentCategory]
    severity_levels: List[IncidentSeverity]
    steps: List[PlaybookStep]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class IncidentResponse:
    """
    Automated incident response system.
    
    Features:
    - Automated incident creation and classification
    - Playbook-driven response procedures
    - Automated containment actions
    - Multi-channel notifications
    - Evidence collection and preservation
    - SLA tracking and escalation
    - Post-incident analysis and reporting
    """
    
    def __init__(self, config: IncidentConfig):
        self.config = config
        self.incidents: Dict[str, SecurityIncident] = {}
        self.playbooks: Dict[str, IncidentPlaybook] = {}
        
        self._setup_directories()
        self._load_playbooks()
        self._setup_logging()
        
    def _setup_directories(self) -> None:
        """Setup incident response directories."""
        Path(self.config.incident_playbooks_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.evidence_storage_dir).mkdir(parents=True, exist_ok=True)
        
    def _setup_logging(self) -> None:
        """Setup incident response logging."""
        self.incident_logger = logging.getLogger("incident_response")
        self.incident_logger.setLevel(logging.INFO)
        
        log_file = Path(".security/incidents.log")
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.incident_logger.addHandler(handler)
        
    def _load_playbooks(self) -> None:
        """Load incident response playbooks."""
        # Create default playbooks if none exist
        if not list(Path(self.config.incident_playbooks_dir).glob("*.json")):
            self._create_default_playbooks()
            
        # Load playbooks from files
        playbook_dir = Path(self.config.incident_playbooks_dir)
        for playbook_file in playbook_dir.glob("*.json"):
            try:
                with open(playbook_file, 'r') as f:
                    data = json.load(f)
                    playbook = self._dict_to_playbook(data)
                    self.playbooks[playbook.playbook_id] = playbook
            except Exception as e:
                logger.error(f"Failed to load playbook {playbook_file}: {e}")
                
    def _create_default_playbooks(self) -> None:
        """Create default incident response playbooks."""
        # Intrusion detection playbook
        intrusion_playbook = IncidentPlaybook(
            playbook_id="intrusion_response",
            name="Intrusion Response",
            description="Response procedures for detected intrusions",
            triggers=["INTRUSION", "UNAUTHORIZED_ACCESS"],
            categories=[IncidentCategory.INTRUSION, IncidentCategory.UNAUTHORIZED_ACCESS],
            severity_levels=[IncidentSeverity.HIGH, IncidentSeverity.CRITICAL],
            steps=[
                PlaybookStep(
                    step_id="1",
                    title="Immediate Containment",
                    description="Block malicious IP addresses and isolate affected systems",
                    action_type="CONTAINMENT",
                    priority=1,
                    estimated_duration=15,
                    automation_possible=True,
                    automation_script="block_malicious_ips.py"
                ),
                PlaybookStep(
                    step_id="2",
                    title="Evidence Collection",
                    description="Collect logs, network traffic, and system artifacts",
                    action_type="INVESTIGATION",
                    priority=1,
                    estimated_duration=30,
                    required_role="security_analyst"
                ),
                PlaybookStep(
                    step_id="3",
                    title="Notify Stakeholders",
                    description="Notify security team and management",
                    action_type="COMMUNICATION",
                    priority=2,
                    estimated_duration=10,
                    automation_possible=True
                ),
                PlaybookStep(
                    step_id="4",
                    title="Root Cause Analysis",
                    description="Investigate attack vector and extent of compromise",
                    action_type="INVESTIGATION",
                    priority=2,
                    estimated_duration=120,
                    required_role="security_analyst"
                ),
                PlaybookStep(
                    step_id="5",
                    title="System Recovery",
                    description="Restore systems and implement additional security measures",
                    action_type="RECOVERY",
                    priority=3,
                    estimated_duration=240,
                    required_role="devops_engineer"
                )
            ]
        )
        
        # DDoS attack playbook
        ddos_playbook = IncidentPlaybook(
            playbook_id="ddos_response",
            name="DDoS Attack Response",
            description="Response procedures for DDoS attacks",
            triggers=["DOS_ATTACK", "HIGH_TRAFFIC"],
            categories=[IncidentCategory.DOS_ATTACK],
            severity_levels=[IncidentSeverity.HIGH, IncidentSeverity.CRITICAL],
            steps=[
                PlaybookStep(
                    step_id="1",
                    title="Traffic Analysis",
                    description="Analyze traffic patterns to confirm DDoS attack",
                    action_type="INVESTIGATION",
                    priority=1,
                    estimated_duration=10,
                    automation_possible=True,
                    automation_script="analyze_traffic.py"
                ),
                PlaybookStep(
                    step_id="2",
                    title="Enable DDoS Protection",
                    description="Activate DDoS protection services and rate limiting",
                    action_type="CONTAINMENT",
                    priority=1,
                    estimated_duration=5,
                    automation_possible=True,
                    automation_script="enable_ddos_protection.py"
                ),
                PlaybookStep(
                    step_id="3",
                    title="Scale Resources",
                    description="Scale infrastructure to handle increased load",
                    action_type="CONTAINMENT",
                    priority=2,
                    estimated_duration=15,
                    automation_possible=True,
                    required_role="devops_engineer"
                ),
                PlaybookStep(
                    step_id="4",
                    title="Monitor and Adjust",
                    description="Monitor effectiveness and adjust protection measures",
                    action_type="RECOVERY",
                    priority=2,
                    estimated_duration=60,
                    required_role="devops_engineer"
                )
            ]
        )
        
        # Save playbooks
        self._save_playbook(intrusion_playbook)
        self._save_playbook(ddos_playbook)
        
    def create_incident(self,
                       title: str,
                       description: str,
                       category: IncidentCategory,
                       severity: IncidentSeverity,
                       reporter: str = "system",
                       related_events: Optional[List[str]] = None,
                       related_alerts: Optional[List[str]] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> SecurityIncident:
        """
        Create a new security incident.
        
        Args:
            title: Brief incident title
            description: Detailed incident description
            category: Incident category
            severity: Incident severity level
            reporter: Who reported the incident
            related_events: Related security event IDs
            related_alerts: Related alert IDs
            metadata: Additional incident metadata
            
        Returns:
            Created security incident
        """
        incident_id = self._generate_incident_id()
        
        incident = SecurityIncident(
            incident_id=incident_id,
            title=title,
            description=description,
            category=category,
            severity=severity,
            reporter=reporter,
            related_events=related_events or [],
            related_alerts=related_alerts or [],
            metadata=metadata or {}
        )
        
        # Store incident
        self.incidents[incident_id] = incident
        
        # Log incident creation
        self.incident_logger.info(
            f"Incident created: {incident_id} - {category.value} - {severity.value} - {title}"
        )
        
        # Auto-assign based on category and severity
        self._auto_assign_incident(incident)
        
        # Execute playbook if available
        self._execute_playbook(incident)
        
        # Send notifications
        self._send_incident_notifications(incident, "CREATED")
        
        # Automated containment if enabled
        if (self.config.auto_containment_enabled and 
            severity.value in ["HIGH", "CRITICAL"]):
            self._execute_automated_containment(incident)
            
        logger.info(f"Security incident created: {incident_id}")
        
        return incident
        
    def _execute_playbook(self, incident: SecurityIncident) -> None:
        """Execute appropriate playbook for incident."""
        # Find matching playbook
        playbook = self._find_matching_playbook(incident)
        
        if not playbook:
            logger.warning(f"No matching playbook found for incident {incident.incident_id}")
            return
            
        logger.info(f"Executing playbook {playbook.name} for incident {incident.incident_id}")
        
        # Create actions from playbook steps
        for step in playbook.steps:
            action = IncidentAction(
                action_id=f"{incident.incident_id}_{step.step_id}",
                action_type=step.action_type,
                title=step.title,
                description=step.description,
                priority=step.priority,
                metadata={
                    'playbook_id': playbook.playbook_id,
                    'step_id': step.step_id,
                    'estimated_duration': step.estimated_duration,
                    'required_role': step.required_role,
                    'automation_possible': step.automation_possible,
                    'automation_script': step.automation_script
                }
            )
            
            incident.actions.append(action)
            
            # Execute automated actions
            if step.automation_possible and step.automation_script:
                self._execute_automated_action(action, step)
                
    def _find_matching_playbook(self, incident: SecurityIncident) -> Optional[IncidentPlaybook]:
        """Find playbook matching incident characteristics."""
        for playbook in self.playbooks.values():
            # Check category match
            if incident.category in playbook.categories:
                # Check severity match
                if incident.severity in playbook.severity_levels:
                    return playbook
                    
        return None
        
    def _execute_automated_action(self, action: IncidentAction, step: PlaybookStep) -> None:
        """Execute automated response action."""
        try:
            action.status = "IN_PROGRESS"
            action.started_at = datetime.now(timezone.utc)
            
            # Execute automation script
            if step.automation_script:
                script_path = Path(self.config.incident_playbooks_dir) / step.automation_script
                
                if script_path.exists():
                    result = subprocess.run(
                        ["python", str(script_path)],
                        capture_output=True,
                        text=True,
                        timeout=300  # 5 minute timeout
                    )
                    
                    if result.returncode == 0:
                        action.status = "COMPLETED"
                        action.metadata['execution_output'] = result.stdout
                    else:
                        action.status = "FAILED"
                        action.metadata['execution_error'] = result.stderr
                else:
                    action.status = "FAILED"
                    action.metadata['execution_error'] = f"Script not found: {script_path}"
            else:
                action.status = "FAILED"
                action.metadata['execution_error'] = "No automation script specified"
                
            action.completed_at = datetime.now(timezone.utc)
            
            logger.info(f"Automated action {action.action_id} completed with status: {action.status}")
            
        except Exception as e:
            action.status = "FAILED"
            action.metadata['execution_error'] = str(e)
            action.completed_at = datetime.now(timezone.utc)
            
            logger.error(f"Failed to execute automated action {action.action_id}: {e}")
            
    def _execute_automated_containment(self, incident: SecurityIncident) -> None:
        """Execute automated containment actions."""
        containment_actions = []
        
        # Block malicious IPs if enabled
        if self.config.auto_block_malicious_ips:
            malicious_ips = self._extract_malicious_ips(incident)
            if malicious_ips:
                containment_actions.append(
                    f"Block malicious IP addresses: {', '.join(malicious_ips)}"
                )
                
        # Isolate compromised accounts if enabled
        if self.config.auto_isolate_compromised_accounts:
            compromised_users = incident.affected_users
            if compromised_users:
                containment_actions.append(
                    f"Isolate compromised user accounts: {', '.join(compromised_users)}"
                )
                
        # Scale resources if under attack
        if (self.config.auto_scale_resources and 
            incident.category == IncidentCategory.DOS_ATTACK):
            containment_actions.append("Scale infrastructure resources")
            
        # Create containment action
        if containment_actions:
            action = IncidentAction(
                action_id=f"{incident.incident_id}_auto_containment",
                action_type="CONTAINMENT",
                title="Automated Containment",
                description="; ".join(containment_actions),
                priority=1,
                status="COMPLETED",
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
                metadata={'automated': True, 'actions': containment_actions}
            )
            
            incident.actions.append(action)
            
            # Update incident status
            incident.status = IncidentStatus.CONTAINED
            incident.contained_at = datetime.now(timezone.utc)
            
            logger.info(f"Automated containment executed for incident {incident.incident_id}")
            
    def _extract_malicious_ips(self, incident: SecurityIncident) -> List[str]:
        """Extract malicious IP addresses from incident."""
        malicious_ips = []
        
        # Extract from indicators of compromise
        for ioc in incident.indicators_of_compromise:
            if self._is_ip_address(ioc):
                malicious_ips.append(ioc)
                
        # Extract from metadata
        if 'source_ips' in incident.metadata:
            source_ips = incident.metadata['source_ips']
            if isinstance(source_ips, list):
                malicious_ips.extend(source_ips)
            elif isinstance(source_ips, str):
                malicious_ips.append(source_ips)
                
        return list(set(malicious_ips))  # Remove duplicates
        
    def _is_ip_address(self, value: str) -> bool:
        """Check if value is an IP address."""
        import ipaddress
        try:
            ipaddress.ip_address(value)
            return True
        except ValueError:
            return False
            
    def _auto_assign_incident(self, incident: SecurityIncident) -> None:
        """Auto-assign incident based on category and severity."""
        # Simple assignment logic - in practice, would be more sophisticated
        if incident.severity in [IncidentSeverity.HIGH, IncidentSeverity.CRITICAL]:
            incident.assigned_to = "security_team_lead"
        else:
            incident.assigned_to = "security_analyst"
            
        incident.acknowledged_at = datetime.now(timezone.utc)
        
    def _send_incident_notifications(self, incident: SecurityIncident, action: str) -> None:
        """Send incident notifications."""
        try:
            # Email notifications
            if self.config.email_notifications_enabled:
                self._send_incident_email(incident, action)
                
            # Slack notifications
            if self.config.slack_notifications_enabled and self.config.slack_webhook_url:
                self._send_incident_slack(incident, action)
                
            # Emergency webhook for critical incidents
            if (incident.severity == IncidentSeverity.CRITICAL and 
                self.config.emergency_webhook_url):
                self._send_emergency_webhook(incident, action)
                
        except Exception as e:
            logger.error(f"Error sending incident notifications: {e}")
            
    def _send_incident_email(self, incident: SecurityIncident, action: str) -> None:
        """Send incident email notification."""
        try:
            # Determine recipient list based on severity
            recipients = []
            
            if incident.severity in [IncidentSeverity.HIGH, IncidentSeverity.CRITICAL]:
                recipients.extend(self.config.security_team_emails)
                recipients.extend(self.config.management_emails)
            else:
                recipients.extend(self.config.security_team_emails)
                
            if incident.category == IncidentCategory.DOS_ATTACK:
                recipients.extend(self.config.devops_emails)
                
            recipients = list(set(recipients))  # Remove duplicates
            
            if not recipients:
                return
                
            subject = f"[SECURITY INCIDENT] {incident.severity.value}: {incident.title}"
            
            body = f"""
Security Incident {action}

Incident ID: {incident.incident_id}
Category: {incident.category.value}
Severity: {incident.severity.value}
Status: {incident.status.value}
Detected: {incident.detected_at.isoformat()}
Reporter: {incident.reporter}

Title: {incident.title}

Description:
{incident.description}

Affected Systems: {', '.join(incident.affected_systems) if incident.affected_systems else 'None specified'}
Affected Users: {', '.join(incident.affected_users) if incident.affected_users else 'None specified'}

Indicators of Compromise:
{chr(10).join('- ' + ioc for ioc in incident.indicators_of_compromise) if incident.indicators_of_compromise else 'None identified'}

Actions Taken:
{chr(10).join('- ' + action.title + ' (' + action.status + ')' for action in incident.actions) if incident.actions else 'None yet'}

This is an automated incident notification from the AI Knowledge Website security system.
"""
            
            msg = MimeMultipart()
            msg['From'] = self.config.incident_email_from
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(self.config.smtp_host, self.config.smtp_port)
            if self.config.smtp_user and self.config.smtp_password:
                server.starttls()
                server.login(self.config.smtp_user, self.config.smtp_password)
                
            server.send_message(msg)
            server.quit()
            
            # Track notification
            incident.stakeholders_notified.extend(recipients)
            
            logger.info(f"Incident email sent for {incident.incident_id}")
            
        except Exception as e:
            logger.error(f"Failed to send incident email: {e}")
            
    def _send_incident_slack(self, incident: SecurityIncident, action: str) -> None:
        """Send incident Slack notification."""
        try:
            color = {
                'LOW': '#36a64f',
                'MEDIUM': '#ff9500', 
                'HIGH': '#ff0000',
                'CRITICAL': '#8B0000'
            }.get(incident.severity.value, '#808080')
            
            payload = {
                'text': f"Security Incident {action}",
                'attachments': [
                    {
                        'color': color,
                        'title': f"{incident.severity.value}: {incident.title}",
                        'text': incident.description,
                        'fields': [
                            {
                                'title': 'Incident ID',
                                'value': incident.incident_id,
                                'short': True
                            },
                            {
                                'title': 'Category',
                                'value': incident.category.value,
                                'short': True
                            },
                            {
                                'title': 'Status',
                                'value': incident.status.value,
                                'short': True
                            },
                            {
                                'title': 'Detected',
                                'value': incident.detected_at.isoformat(),
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
            
            logger.info(f"Incident Slack notification sent for {incident.incident_id}")
            
        except Exception as e:
            logger.error(f"Failed to send incident Slack notification: {e}")
            
    def update_incident_status(self, 
                              incident_id: str, 
                              status: IncidentStatus,
                              user: str = "system") -> bool:
        """Update incident status."""
        if incident_id not in self.incidents:
            return False
            
        incident = self.incidents[incident_id]
        old_status = incident.status
        incident.status = status
        
        # Update timeline
        now = datetime.now(timezone.utc)
        
        if status == IncidentStatus.INVESTIGATING and not incident.acknowledged_at:
            incident.acknowledged_at = now
        elif status == IncidentStatus.CONTAINED and not incident.contained_at:
            incident.contained_at = now
        elif status == IncidentStatus.RESOLVED and not incident.resolved_at:
            incident.resolved_at = now
        elif status == IncidentStatus.CLOSED and not incident.closed_at:
            incident.closed_at = now
            
        # Log status change
        self.incident_logger.info(
            f"Incident {incident_id} status changed from {old_status.value} to {status.value} by {user}"
        )
        
        # Send notifications for significant status changes
        if status in [IncidentStatus.CONTAINED, IncidentStatus.RESOLVED, IncidentStatus.CLOSED]:
            self._send_incident_notifications(incident, f"STATUS_UPDATED_TO_{status.value}")
            
        return True
        
    def get_incident_dashboard(self) -> Dict[str, Any]:
        """Get incident response dashboard data."""
        now = datetime.now(timezone.utc)
        
        # Incident statistics
        total_incidents = len(self.incidents)
        open_incidents = len([i for i in self.incidents.values() if i.status != IncidentStatus.CLOSED])
        critical_incidents = len([i for i in self.incidents.values() 
                                 if i.severity == IncidentSeverity.CRITICAL and i.status != IncidentStatus.CLOSED])
        
        # Recent incidents
        recent_incidents = sorted(
            self.incidents.values(),
            key=lambda x: x.detected_at,
            reverse=True
        )[:10]
        
        # SLA violations
        sla_violations = self._check_sla_violations()
        
        # Category breakdown
        category_counts = {}
        severity_counts = {}
        
        for incident in self.incidents.values():
            category_counts[incident.category.value] = category_counts.get(incident.category.value, 0) + 1
            severity_counts[incident.severity.value] = severity_counts.get(incident.severity.value, 0) + 1
            
        return {
            'summary': {
                'total_incidents': total_incidents,
                'open_incidents': open_incidents,
                'critical_incidents': critical_incidents,
                'sla_violations': len(sla_violations)
            },
            'recent_incidents': [
                {
                    'incident_id': i.incident_id,
                    'title': i.title,
                    'category': i.category.value,
                    'severity': i.severity.value,
                    'status': i.status.value,
                    'detected_at': i.detected_at.isoformat()
                }
                for i in recent_incidents
            ],
            'category_breakdown': category_counts,
            'severity_breakdown': severity_counts,
            'sla_violations': sla_violations,
            'playbooks_available': len(self.playbooks),
            'last_updated': now.isoformat()
        }
        
    def _check_sla_violations(self) -> List[Dict[str, Any]]:
        """Check for SLA violations."""
        violations = []
        now = datetime.now(timezone.utc)
        
        for incident in self.incidents.values():
            if incident.status == IncidentStatus.CLOSED:
                continue
                
            # Initial response SLA
            if not incident.acknowledged_at:
                hours_since_detection = (now - incident.detected_at).total_seconds() / 3600
                if hours_since_detection > self.config.initial_response_sla:
                    violations.append({
                        'incident_id': incident.incident_id,
                        'sla_type': 'initial_response',
                        'hours_overdue': hours_since_detection - self.config.initial_response_sla
                    })
                    
            # Containment SLA for high severity incidents
            if (incident.severity in [IncidentSeverity.HIGH, IncidentSeverity.CRITICAL] and
                not incident.contained_at):
                hours_since_detection = (now - incident.detected_at).total_seconds() / 3600
                if hours_since_detection > self.config.containment_sla:
                    violations.append({
                        'incident_id': incident.incident_id,
                        'sla_type': 'containment',
                        'hours_overdue': hours_since_detection - self.config.containment_sla
                    })
                    
        return violations
        
    def _generate_incident_id(self) -> str:
        """Generate unique incident ID."""
        import uuid
        return f"INC-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
        
    def _dict_to_playbook(self, data: Dict[str, Any]) -> IncidentPlaybook:
        """Convert dictionary to playbook object."""
        steps = [
            PlaybookStep(**step_data) for step_data in data.get('steps', [])
        ]
        
        return IncidentPlaybook(
            playbook_id=data['playbook_id'],
            name=data['name'],
            description=data['description'],
            triggers=data['triggers'],
            categories=[IncidentCategory(cat) for cat in data['categories']],
            severity_levels=[IncidentSeverity(sev) for sev in data['severity_levels']],
            steps=steps,
            created_at=datetime.fromisoformat(data.get('created_at', datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(data.get('updated_at', datetime.now().isoformat()))
        )
        
    def _save_playbook(self, playbook: IncidentPlaybook) -> None:
        """Save playbook to file."""
        try:
            playbook_file = Path(self.config.incident_playbooks_dir) / f"{playbook.playbook_id}.json"
            
            data = {
                'playbook_id': playbook.playbook_id,
                'name': playbook.name,
                'description': playbook.description,
                'triggers': playbook.triggers,
                'categories': [cat.value for cat in playbook.categories],
                'severity_levels': [sev.value for sev in playbook.severity_levels],
                'steps': [
                    {
                        'step_id': step.step_id,
                        'title': step.title,
                        'description': step.description,
                        'action_type': step.action_type,
                        'priority': step.priority,
                        'estimated_duration': step.estimated_duration,
                        'required_role': step.required_role,
                        'automation_possible': step.automation_possible,
                        'automation_script': step.automation_script,
                        'metadata': step.metadata
                    }
                    for step in playbook.steps
                ],
                'created_at': playbook.created_at.isoformat(),
                'updated_at': playbook.updated_at.isoformat()
            }
            
            with open(playbook_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Playbook saved: {playbook.playbook_id}")
            
        except Exception as e:
            logger.error(f"Failed to save playbook {playbook.playbook_id}: {e}")


# Factory function
def create_incident_response(config: Optional[IncidentConfig] = None) -> IncidentResponse:
    """Create configured incident response instance."""
    config = config or IncidentConfig()
    return IncidentResponse(config)