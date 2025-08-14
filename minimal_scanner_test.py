#!/usr/bin/env python3
"""
Minimal Scanner Test - Core Logic without External Dependencies
"""

import sqlite3
import json
import threading
import asyncio
import time
import uuid
import logging
import re
import os
import sys
import base64
import hashlib
import hmac
import urllib.request
import urllib.parse
import urllib.error
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class VulnerabilityResult:
    """Minimal vulnerability result"""
    vulnerability_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    vulnerability_type: str = ""
    target_url: str = ""
    parameter: str = ""
    payload: str = ""
    confidence: float = 0.0
    evidence: str = ""
    discovered_at: datetime = field(default_factory=datetime.now)

class MinimalPayloadDatabase:
    """Minimal payload database"""
    
    def __init__(self):
        self.payloads = {
            'sql_injection': [
                "1' OR '1'='1'-- ",
                "1' AND (SELECT sleep(5))-- ",
                "' UNION SELECT 1,2,3-- ",
                "1' AND extractvalue(rand(),concat(0x3a,version()))-- "
            ],
            'xss': [
                "<script>alert('XSS')</script>",
                "<img src=x onerror=alert('XSS')>",
                "<svg onload=alert('XSS')>",
                "javascript:alert('XSS')"
            ],
            'command_injection': [
                "; cat /etc/passwd",
                "| whoami",
                "&& ls -la",
                "`id`"
            ]
        }
    
    def get_payloads(self, vuln_type: str) -> List[str]:
        return self.payloads.get(vuln_type, [])

class MinimalDetectionEngine:
    """Minimal detection engine using only stdlib"""
    
    def __init__(self):
        self.payload_db = MinimalPayloadDatabase()
        self.error_patterns = [
            r'mysql_fetch_array\(\)',
            r'SQL syntax.*error',
            r'ORA-\d{5}',
            r'PostgreSQL.*ERROR'
        ]
    
    def test_sql_injection(self, target_url: str, parameter: str) -> List[VulnerabilityResult]:
        """Test SQL injection using urllib"""
        results = []
        payloads = self.payload_db.get_payloads('sql_injection')
        
        for payload in payloads:
            try:
                # Build test URL
                test_url = f"{target_url}?{parameter}={urllib.parse.quote(payload)}"
                
                # Send request
                start_time = time.time()
                response = self._send_request(test_url)
                response_time = time.time() - start_time
                
                if response:
                    # Check for SQL errors
                    for pattern in self.error_patterns:
                        if re.search(pattern, response, re.IGNORECASE):
                            result = VulnerabilityResult(
                                vulnerability_type="sql_injection",
                                target_url=target_url,
                                parameter=parameter,
                                payload=payload,
                                confidence=0.8,
                                evidence=f"SQL error pattern detected: {pattern}"
                            )
                            results.append(result)
                            break
                    
                    # Check for time-based
                    if 'sleep' in payload and response_time > 4.0:
                        result = VulnerabilityResult(
                            vulnerability_type="sql_injection",
                            target_url=target_url,
                            parameter=parameter,
                            payload=payload,
                            confidence=0.9,
                            evidence=f"Time-based SQL injection: {response_time:.2f}s delay"
                        )
                        results.append(result)
                
            except Exception as e:
                logger.debug(f"Request failed: {e}")
        
        return results
    
    def test_xss(self, target_url: str, parameter: str) -> List[VulnerabilityResult]:
        """Test XSS vulnerabilities"""
        results = []
        payloads = self.payload_db.get_payloads('xss')
        
        for payload in payloads:
            try:
                test_url = f"{target_url}?{parameter}={urllib.parse.quote(payload)}"
                response = self._send_request(test_url)
                
                if response and payload in response:
                    result = VulnerabilityResult(
                        vulnerability_type="xss",
                        target_url=target_url,
                        parameter=parameter,
                        payload=payload,
                        confidence=0.7,
                        evidence="Payload reflected in response"
                    )
                    results.append(result)
                
            except Exception as e:
                logger.debug(f"Request failed: {e}")
        
        return results
    
    def _send_request(self, url: str) -> Optional[str]:
        """Send HTTP request using urllib"""
        try:
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Enterprise-Scanner/2.0')
            
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.read().decode('utf-8', errors='ignore')
                
        except Exception:
            return None

class MinimalScanner:
    """Minimal working scanner"""
    
    def __init__(self):
        self.detection_engine = MinimalDetectionEngine()
        self.scan_results = {}
        logger.info("Minimal scanner initialized")
    
    def scan_target(self, target_url: str, parameters: List[str] = None) -> Dict:
        """Scan target for vulnerabilities"""
        if not parameters:
            parameters = ['id', 'page', 'search', 'q']
        
        scan_id = str(uuid.uuid4())
        scan_start = datetime.now()
        
        logger.info(f"Starting scan {scan_id} for {target_url}")
        
        all_vulnerabilities = []
        
        # Test SQL injection
        for param in parameters:
            sql_vulns = self.detection_engine.test_sql_injection(target_url, param)
            all_vulnerabilities.extend(sql_vulns)
        
        # Test XSS
        for param in parameters:
            xss_vulns = self.detection_engine.test_xss(target_url, param)
            all_vulnerabilities.extend(xss_vulns)
        
        scan_result = {
            'scan_id': scan_id,
            'target_url': target_url,
            'started_at': scan_start.isoformat(),
            'completed_at': datetime.now().isoformat(),
            'vulnerabilities_found': len(all_vulnerabilities),
            'vulnerabilities': [asdict(vuln) for vuln in all_vulnerabilities],
            'status': 'completed'
        }
        
        self.scan_results[scan_id] = scan_result
        
        logger.info(f"Scan {scan_id} completed. Found {len(all_vulnerabilities)} vulnerabilities")
        
        return scan_result

def test_minimal_scanner():
    """Test the minimal scanner"""
    print("🧪 Testing Minimal Scanner...")
    
    # Test with a safe test URL (httpbin.org)
    scanner = MinimalScanner()
    
    # Test basic functionality
    test_url = "https://httpbin.org/get"
    result = scanner.scan_target(test_url, ['param', 'test'])
    
    print(f"✅ Scan completed: {result['scan_id']}")
    print(f"📊 Vulnerabilities found: {result['vulnerabilities_found']}")
    
    # Test with intentionally vulnerable URL (for demo)
    print("\n🎯 Testing with demo vulnerable URL...")
    
    # Simulate vulnerable response
    class MockDetectionEngine(MinimalDetectionEngine):
        def _send_request(self, url):
            if 'sleep' in url:
                time.sleep(0.1)  # Simulate delay
                return "mysql_fetch_array() error"
            elif 'script' in url:
                return url  # Reflect the payload
            return "Normal response"
    
    scanner.detection_engine = MockDetectionEngine()
    demo_result = scanner.scan_target("http://demo.testsite.com/page.php", ['id'])
    
    print(f"✅ Demo scan completed: {demo_result['scan_id']}")
    print(f"📊 Demo vulnerabilities found: {demo_result['vulnerabilities_found']}")
    
    if demo_result['vulnerabilities']:
        print("🔍 Sample vulnerability:")
        vuln = demo_result['vulnerabilities'][0]
        print(f"   Type: {vuln['vulnerability_type']}")
        print(f"   Confidence: {vuln['confidence']}")
        print(f"   Evidence: {vuln['evidence']}")

if __name__ == '__main__':
    test_minimal_scanner()