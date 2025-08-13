import os
import json
import uuid
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

from flask import Flask, request, jsonify


# -----------------------------
# Configuration and Utilities
# -----------------------------

class AdvancedConfig:
    """Minimal configuration loader for the consolidated app."""

    def __init__(self) -> None:
        self.config: Dict[str, Dict[str, str]] = {
            "security": {
                "jwt_secret": os.getenv("JWT_SECRET_KEY", self._generate_random_secret()),
            }
        }

    def _generate_random_secret(self) -> str:
        return hashlib.sha256(f"{uuid.uuid4()}{time.time()}".encode()).hexdigest()


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """Hash a password using SHA-256 with a salt.

    Returns (salt, hash).
    """
    salt_to_use = salt or uuid.uuid4().hex
    hashed = hashlib.sha256(f"{salt_to_use}:{password}".encode()).hexdigest()
    return salt_to_use, hashed


# -----------------------------
# Data Models
# -----------------------------

@dataclass
class VulnerabilityResult:
    vuln_type: str
    severity: str
    confidence: float
    title: str
    description: str
    location: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


# -----------------------------
# Core Services
# -----------------------------

class EnterpriseSecurityScanner:
    """Minimal scanner implementation returning dummy results."""

    def __init__(self, config: AdvancedConfig) -> None:
        self.config = config

    def scan(self, target: str, scan_type: str, scan_options: Dict[str, object]) -> List[VulnerabilityResult]:
        # In a real implementation, this would perform complex scanning.
        # Here we return deterministic but realistic-looking results.
        baseline: List[VulnerabilityResult] = [
            VulnerabilityResult(
                vuln_type="WEB_APP",
                severity="HIGH",
                confidence=0.92,
                title="Outdated framework detected",
                description="The target appears to run an outdated web framework.",
                location=target,
            ),
            VulnerabilityResult(
                vuln_type="CONFIG",
                severity="MEDIUM",
                confidence=0.81,
                title="Missing security headers",
                description="Some recommended HTTP security headers are missing.",
                location=target,
            ),
        ]

        if scan_options.get("include_network"):
            baseline.append(
                VulnerabilityResult(
                    vuln_type="NETWORK",
                    severity="LOW",
                    confidence=0.75,
                    title="Open non-critical port",
                    description="A non-critical service port appears to be reachable.",
                    location=target,
                )
            )

        return baseline


class UserManager:
    """In-memory user manager for simplicity."""

    def __init__(self) -> None:
        # username -> record
        self._users: Dict[str, Dict[str, object]] = {}
        # Create a default admin user
        self.create_user(username="admin", email="admin@example.com", password="admin", role="admin")

    def _generate_api_key(self) -> str:
        return hashlib.sha256(f"{uuid.uuid4()}{time.time()}".encode()).hexdigest()

    def create_user(self, username: str, email: str, password: str, role: str = "user") -> Dict[str, object]:
        if username in self._users:
            raise ValueError("Username already exists")

        salt, password_hash = hash_password(password)
        api_key = self._generate_api_key()
        user_record = {
            "username": username,
            "email": email,
            "password_salt": salt,
            "password_hash": password_hash,
            "role": role,
            "api_key": api_key,
            "created_at": int(time.time()),
        }
        self._users[username] = user_record
        return {
            "username": username,
            "email": email,
            "role": role,
            "api_key": api_key,
        }

    def authenticate(self, username: str, password: str) -> Optional[Dict[str, object]]:
        user = self._users.get(username)
        if not user:
            return None
        salt = user["password_salt"]
        _, hashed = hash_password(password, salt)
        if hashed != user["password_hash"]:
            return None
        return {
            "username": user["username"],
            "email": user["email"],
            "role": user["role"],
            "api_key": user["api_key"],
        }

    def get_user_by_api_key(self, api_key: str) -> Optional[Dict[str, object]]:
        for user in self._users.values():
            if user["api_key"] == api_key:
                return {
                    "username": user["username"],
                    "email": user["email"],
                    "role": user["role"],
                    "api_key": user["api_key"],
                }
        return None


# -----------------------------
# App Factory
# -----------------------------

def create_app() -> Flask:
    app = Flask(__name__)

    config = AdvancedConfig()
    scanner = EnterpriseSecurityScanner(config)
    users = UserManager()

    # Basic in-memory storage for scans
    scans: Dict[str, Dict[str, object]] = {}

    def require_api_key() -> Optional[Dict[str, object]]:
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            return None
        return users.get_user_by_api_key(api_key)

    @app.get("/health")
    def health() -> Tuple[str, int]:
        return "ok", 200

    @app.get("/metrics")
    def metrics() -> Tuple[str, int, Dict[str, str]]:
        body = [
            "app_up 1",
            "dummy_counter_total 42",
        ]
        return "\n".join(body) + "\n", 200, {"Content-Type": "text/plain; version=0.0.4"}

    @app.post("/api/auth/login")
    def login():
        try:
            payload = request.get_json(force=True)
            username = str(payload.get("username", "")).strip()
            password = str(payload.get("password", "")).strip()
        except Exception:
            return jsonify({"error": "Invalid JSON"}), 400

        if not username or not password:
            return jsonify({"error": "username and password are required"}), 400

        auth = users.authenticate(username, password)
        if not auth:
            return jsonify({"error": "Invalid credentials"}), 401
        return jsonify({"user": auth, "token_type": "API_KEY", "expires_in": 0})

    @app.post("/api/users")
    def create_user():
        try:
            payload = request.get_json(force=True)
            username = str(payload.get("username", "")).strip()
            email = str(payload.get("email", "")).strip()
            password = str(payload.get("password", "")).strip()
            role = str(payload.get("role", "user")).strip()
        except Exception:
            return jsonify({"error": "Invalid JSON"}), 400

        if not username or not email or not password:
            return jsonify({"error": "username, email, and password are required"}), 400

        try:
            created = users.create_user(username=username, email=email, password=password, role=role)
            return jsonify({"user": created}), 201
        except ValueError as ve:
            return jsonify({"error": str(ve)}), 409
        except Exception as exc:
            return jsonify({"error": f"Failed to create user: {exc}"}), 500

    @app.post("/api/scans")
    def start_scan():
        user = require_api_key()
        if not user:
            return jsonify({"error": "Missing or invalid X-API-Key"}), 401

        try:
            payload = request.get_json(force=True)
        except Exception:
            return jsonify({"error": "Invalid JSON"}), 400

        scan_type = str(payload.get("scan_type", "web")).strip()
        target = str(payload.get("target", "")).strip()
        priority = str(payload.get("priority", "medium")).strip()
        async_scan = bool(payload.get("async_scan", False))
        scan_options = payload.get("scan_options", {})
        if not isinstance(scan_options, dict):
            scan_options = {}

        if not target:
            return jsonify({"error": "target is required"}), 400

        scan_id = uuid.uuid4().hex
        results = scanner.scan(target=target, scan_type=scan_type, scan_options=scan_options)

        scan_record = {
            "scan_id": scan_id,
            "requested_by": user["username"],
            "scan_type": scan_type,
            "target": target,
            "priority": priority,
            "async": async_scan,
            "results": [r.to_dict() for r in results],
            "created_at": int(time.time()),
        }
        scans[scan_id] = scan_record

        response = {
            "scan_id": scan_id,
            "status": "completed",
            "total_vulns": len(scan_record["results"]),
            "vulnerabilities": scan_record["results"],
            "risk_score": _calculate_risk_score(scan_record["results"]),
        }
        return jsonify(response), 200

    @app.get("/api/scans/<scan_id>")
    def get_scan(scan_id: str):
        user = require_api_key()
        if not user:
            return jsonify({"error": "Missing or invalid X-API-Key"}), 401
        data = scans.get(scan_id)
        if not data:
            return jsonify({"error": "scan not found"}), 404
        return jsonify(data), 200

    return app


# -----------------------------
# Helpers
# -----------------------------

def _calculate_risk_score(results: List[Dict[str, object]]) -> int:
    score_map = {"LOW": 1, "MEDIUM": 3, "HIGH": 6, "CRITICAL": 10}
    total = 0
    for item in results:
        sev = str(item.get("severity", "LOW")).upper()
        total += score_map.get(sev, 0)
    return total


# -----------------------------
# Entrypoint
# -----------------------------

if __name__ == "__main__":
    app = create_app()
    print("=" * 80)
    print("Enterprise Security Scanner - Consolidated Minimal App")
    print("API:")
    print("  POST /api/auth/login                -> returns API key")
    print("  POST /api/users                     -> create user")
    print("  POST /api/scans                     -> start scan (requires X-API-Key)")
    print("  GET  /api/scans/<scan_id>           -> get scan (requires X-API-Key)")
    print("  GET  /health                         -> health check")
    print("  GET  /metrics                        -> prometheus-style metrics")
    print("=" * 80)
    app.run(host="0.0.0.0", port=3000, debug=False)