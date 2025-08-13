# enterprise-security-scanner
Description: Advanced Security Scanner Code

## Run the consolidated app

- Create or use Python 3.10+.
- Install deps (may need system override):

```bash
python3 -m pip install --break-system-packages -r /workspace/requirements.txt
```

- Start the app:

```bash
python3 /workspace/final_app.py
```

- Basic usage:

```bash
# Health
curl -s http://localhost:3000/health

# Create user
curl -s -X POST http://localhost:3000/api/users \
  -H 'Content-Type: application/json' \
  -d '{"username":"u1","email":"u1@example.com","password":"pass"}'

# Start scan (replace API_KEY)
curl -s -X POST http://localhost:3000/api/scans \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: API_KEY' \
  -d '{"scan_type":"web","target":"example.com","scan_options":{"include_network":true}}'
```
