#!/usr/bin/env python3
from pathlib import Path
import os
import sys
import uuid
import logging
import json
import sqlite3
import time
import threading
from datetime import datetime, timedelta
import pytz
import psutil
from flask import Flask, jsonify, request, send_file
from werkzeug.utils import secure_filename
import subprocess
import serial
import re
import random
import signal
import math
import struct
import jwt
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
import numpy as np
from flask import request, jsonify
from pysat.formula import CNF
from pysat.solvers import Solver
import io

# Load environment variables
try:
    from dotenv import load_dotenv
    # Load from parent directory (root of project)
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(env_path)
except ImportError:
    print("python-dotenv not installed, using system environment variables")

# --- App Initialization ---
app = Flask(__name__)

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# --- Global Variables ---
PORT_TO_BOARD_MAP = {}
BOARD_TO_PORT_MAP = {}
HARDWARE_STATUS = {}
RUNNING_JOBS = {}

# --- Auto-detect connected boards at startup ---
def auto_detect_boards():
    """Automatically detect and identify all connected boards at startup"""
    try:
        import serial.tools.list_ports
        import glob
        
        logger.info("Auto-detecting connected boards...")
        
        # Get all potential serial ports based on platform
        port_paths = []
        
        if sys.platform.startswith('win'):
            # Windows
            ports = list(serial.tools.list_ports.comports())
            port_paths = [p.device for p in ports]
        elif sys.platform.startswith('linux'):
            # Linux/Raspberry Pi
            port_paths = glob.glob('/dev/tty[A-Za-z]*')
            # Filter likely USB serial devices
            port_paths = [p for p in port_paths if ('ACM' in p or 'USB' in p)]
        elif sys.platform.startswith('darwin'):
            # macOS - use both cu.* and tty.* but prioritize cu.* for communication
            cu_ports = glob.glob('/dev/cu.*')
            tty_ports = glob.glob('/dev/tty.*')
            # Filter likely USB serial devices and prioritize cu.* ports
            cu_usb_ports = [p for p in cu_ports if ('usbmodem' in p)]
            tty_usb_ports = [p for p in tty_ports if ('usbmodem' in p)]
            # Use cu.* ports first (better for communication), then tty.* as fallback
            port_paths = cu_usb_ports + tty_usb_ports
        
        identified_count = 0
        
        # Try to identify each potential board
        for port in port_paths:
            # Skip ports that are not likely to be USB devices
            if not any(s in port for s in ['USB', 'ACM', 'usbmodem']):
                continue
                
            try:
                # Try to identify the board with a shorter timeout for startup
                board_type = identify_board(port, timeout=2)
                
                if board_type:
                    PORT_TO_BOARD_MAP[port] = board_type
                    identified_count += 1
                    logger.info(f"Auto-detected board: {port} -> {board_type}")
            except Exception as e:
                logger.error(f"Error auto-detecting board on {port}: {e}")
        
        logger.info(f"Auto-detection complete. Found {identified_count} board(s).")
        
    except Exception as e:
        logger.error(f"Error during board auto-detection: {e}")

# --- CORS Configuration ---
# Load allowed origins from environment variable, with fallback to defaults
allowed_origins_env = os.getenv('ALLOWED_ORIGINS', 'http://localhost:3000,https://dacroq.eecs.umich.edu,https://dacroq-api.bendatsko.com,https://release.bendatsko.com,https://dacroq.net')
ALLOWED_ORIGINS = set(origin.strip() for origin in allowed_origins_env.split(','))

@app.after_request
def add_cors(response):
    origin = request.headers.get('Origin')
    if origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET,PUT,POST,DELETE,OPTIONS"
        response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

# Handle preflight OPTIONS requests
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        origin = request.headers.get('Origin')
        if origin in ALLOWED_ORIGINS:
            response = app.make_default_options_response()
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
            response.headers["Access-Control-Allow-Methods"] = "GET,PUT,POST,DELETE,OPTIONS"
            response.headers["Access-Control-Allow-Credentials"] = "true"
            return response

# --- Database Setup ---
# Add parent directory to path so we can import from root
sys.path.append(str(Path(__file__).parent.parent))

# Update file paths to reference parent directory
BASE_DIR = Path(__file__).parent.parent  # Go up one level from daemon/
UPLOAD_FOLDER = BASE_DIR / 'uploads'
# Use environment variable for database path, with fallback
DB_PATH = Path(os.getenv('DATABASE_PATH', str(BASE_DIR / 'daemon' / 'data' / 'database' / 'dacroq.db')))
STATIC_FOLDER = BASE_DIR / 'static'
        
        # Ensure directories exist
UPLOAD_FOLDER.mkdir(exist_ok=True)
STATIC_FOLDER.mkdir(exist_ok=True)

# Update LDPC data path to use centralized data directory
LDPC_DATA_DIR = BASE_DIR / 'daemon' / 'data' / 'ldpc'

def init_db():
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Tests table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tests (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        chip_type TEXT NOT NULL,
        test_mode TEXT,
        environment TEXT,
        config TEXT,
        status TEXT NOT NULL,
        created TEXT NOT NULL,
        metadata TEXT
    )''')
    
    # Files table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS files (
        id TEXT PRIMARY KEY,
        test_id TEXT NOT NULL,
        filename TEXT NOT NULL,
        file_size INTEGER NOT NULL,
        path TEXT NOT NULL,
        created TEXT NOT NULL,
        FOREIGN KEY (test_id) REFERENCES tests(id) ON DELETE CASCADE
    )''')
    
    # Large data table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS large_data (
        id TEXT PRIMARY KEY,
        test_id TEXT NOT NULL,
        name TEXT NOT NULL,
        data_type TEXT NOT NULL,
        storage_type TEXT NOT NULL,
        content BLOB,
        filepath TEXT,
        created TEXT NOT NULL,
        FOREIGN KEY (test_id) REFERENCES tests(id) ON DELETE CASCADE
    )''')
    
    # Test results table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS test_results (
        id TEXT PRIMARY KEY,
        test_id TEXT NOT NULL,
        iteration INTEGER NOT NULL,
        timestamp TEXT NOT NULL,
        frequency REAL,
        voltage REAL,
        temperature REAL,
        board_type TEXT,
        environment TEXT,
        results TEXT,
        FOREIGN KEY (test_id) REFERENCES tests(id) ON DELETE CASCADE
    )''')
    
    # Feedback table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS feedback (
        id TEXT PRIMARY KEY,
        user_id TEXT,
        component TEXT,
        message TEXT NOT NULL,
        metadata TEXT,
        created TEXT NOT NULL
    )''')
    
    # System metrics table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS system_metrics (
        id TEXT PRIMARY KEY,
        timestamp TEXT NOT NULL,
        cpu_percent REAL,
        memory_percent REAL,
        disk_percent REAL,
        temperature REAL,
        connected_devices TEXT
    )''')
    
    # LDPC jobs table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ldpc_jobs (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        job_type TEXT NOT NULL,
        config TEXT NOT NULL,
        status TEXT NOT NULL,
        created TEXT NOT NULL,
        started TEXT,
        completed TEXT,
        results TEXT,
        progress REAL DEFAULT 0,
        current_step TEXT,
        total_steps INTEGER,
        metadata TEXT
    )''')
    
    # Users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        email TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'user',
        created_at TEXT NOT NULL,
        last_login TEXT,
        google_sub TEXT UNIQUE
    )''')
    
    # Ensure new columns on existing installations
    cursor.execute("PRAGMA table_info(users)")
    user_cols = [row[1] for row in cursor.fetchall()]
    if 'google_sub' not in user_cols:
        # SQLite's ALTER TABLE cannot add a column with a UNIQUE constraint.
        cursor.execute('ALTER TABLE users ADD COLUMN google_sub TEXT')
        # Create a unique index to enforce uniqueness going forward (duplicates will raise errors on insert).
        cursor.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_users_google_sub ON users(google_sub)')
        logger.info("Added missing 'google_sub' column (with unique index) to users table during init_db migration")
    if 'created_at' not in user_cols:
        cursor.execute('ALTER TABLE users ADD COLUMN created_at TEXT')
        logger.info("Added missing 'created_at' column to users table during init_db migration")
    
    # API keys table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS api_keys (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        name TEXT NOT NULL,
        key_hash TEXT NOT NULL,
        created_at TEXT NOT NULL,
        last_used_at TEXT,
        expires_at TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )''')

    # System settings table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS system_settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        updated_by TEXT NOT NULL
    )''')

    # Announcements table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS announcements (
        id TEXT PRIMARY KEY,
        message TEXT NOT NULL,
        type TEXT NOT NULL DEFAULT 'info',
        expires_at TEXT,
        created_at TEXT NOT NULL,
        created_by TEXT NOT NULL,
        active BOOLEAN DEFAULT 1
    )''')

    # --- Schema migrations -------------------------------------------------
    # Ensure the announcements table has the "active" column (older DBs created
    # before this column existed will be missing it).
    cursor.execute("PRAGMA table_info(announcements)")
    announcement_cols = [row[1] for row in cursor.fetchall()]
    if 'active' not in announcement_cols:
        cursor.execute('ALTER TABLE announcements ADD COLUMN active BOOLEAN DEFAULT 1')
        logger.info("Added missing 'active' column to announcements table during init_db migration")
    if 'expires_at' not in announcement_cols:
        cursor.execute('ALTER TABLE announcements ADD COLUMN expires_at TEXT')
        logger.info("Added missing 'expires_at' column to announcements table during init_db migration")
    if 'created_at' not in announcement_cols:
        cursor.execute('ALTER TABLE announcements ADD COLUMN created_at TEXT')
        logger.info("Added missing 'created_at' column to announcements table during init_db migration")
    if 'created_by' not in announcement_cols:
        cursor.execute('ALTER TABLE announcements ADD COLUMN created_by TEXT')
        logger.info("Added missing 'created_by' column to announcements table during init_db migration")

    conn.commit()
    conn.close()

def collect_system_metrics():
    """Collect system metrics and store in database"""
    try:
        # Get basic system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get temperature (platform-specific)
        temperature = None
        try:
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    # Try to get CPU temperature
                    for name, entries in temps.items():
                        if entries and ('cpu' in name.lower() or 'core' in name.lower()):
                            temperature = entries[0].current
                            break
        except:
            pass
        
        # Count connected devices
        connected_devices = len(PORT_TO_BOARD_MAP)
        
        # Store in database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if the table has the expected schema
        cursor.execute("PRAGMA table_info(system_metrics)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if 'id' not in columns:
            # Drop and recreate table with correct schema
            cursor.execute('DROP TABLE IF EXISTS system_metrics')
            cursor.execute('''
                CREATE TABLE system_metrics (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    cpu_percent REAL,
                    memory_percent REAL,
                    disk_percent REAL,
                    temperature REAL,
                    connected_devices TEXT
                )
            ''')

        cursor.execute('''
            INSERT INTO system_metrics 
            (id, timestamp, cpu_percent, memory_percent, disk_percent, temperature, connected_devices)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            generate_id(),
            datetime.utcnow().isoformat(),
            cpu_percent,
            memory.percent,
            disk.percent,
            temperature,
            json.dumps(list(PORT_TO_BOARD_MAP.keys()))
        ))
        conn.commit()
        conn.close()

    except Exception as e:
        logger.error(f"Error collecting system metrics: {e}")

def get_db_connection():
    """Get database connection with row factory"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def dict_from_row(row):
    """Convert sqlite3.Row to dictionary"""
    return {col: row[col] for col in row.keys()}

def identify_board(port_path, timeout=3):
    """Identify board type by sending test command"""
    try:
        ser = serial.Serial(port_path, 115200, timeout=timeout)
        time.sleep(0.1)
        
        # Send identification command
        ser.write(b'ID\n')
        time.sleep(0.5)
        
        response = ser.read_all().decode('utf-8', errors='ignore')
        ser.close()
        
        if 'DAEDALUS' in response or '3SAT' in response:
            return '3sat'
        elif 'MEDUSA' in response or 'KSAT' in response:
            return 'ksat'
        elif 'AMORGOS' in response or 'LDPC' in response:
            return 'ldpc'
        
        return None
        
    except Exception as e:
        logger.error(f"Error identifying board on {port_path}: {e}")
        return None

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'txt', 'csv', 'json', 'log', 'cnf', 'dimacs'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_id():
    """Generate unique ID"""
    return str(uuid.uuid4())

# Request timing middleware
@app.before_request
def start_timer():
    request.start_time = time.time()

@app.after_request
def log_request(response):
    if hasattr(request, 'start_time'):
        duration = time.time() - request.start_time
        if duration > 1.0:  # Log slow requests
            logger.warning(f"Slow request: {request.method} {request.path} took {duration:.2f}s")
    return response

# Basic health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Basic health check endpoint"""
    try:
        # Test database connection
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        conn.close()
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'uptime': time.time() - app.start_time if hasattr(app, 'start_time') else 0,
            'connected_devices': len(PORT_TO_BOARD_MAP)
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'Dacroq API Server',
        'version': '1.0',
        'endpoints': {
            '/health': 'Health check',
            '/system/*': 'System management',
            '/hardware/*': 'Hardware control',
            '/tests/*': 'Test management',
            '/ldpc/*': 'LDPC functionality',
            '/auth/*': 'Authentication',
            '/admin/*': 'Admin functions'
        }
    })

# Authentication endpoints
@app.route('/auth/google', methods=['POST'])
def google_auth():
    """Authenticate with Google OAuth"""
    try:
        data = request.get_json()
        # Accept both "credential" (from Google Identity) and our legacy "token" field
        token = data.get('credential') or data.get('token')
        
        if not token:
            return jsonify({'error': 'No credential provided'}), 400
        
        # Verify the Google token
        google_client_id = os.getenv('GOOGLE_CLIENT_ID')
        if not google_client_id:
            logger.error("GOOGLE_CLIENT_ID not found in environment variables")
            return jsonify({'error': 'Server configuration error'}), 500
        
        try:
            # Attempt to verify the token with Google. This will succeed when the
            # frontend supplies a real JWT ID token (preferred).
            idinfo = id_token.verify_oauth2_token(
                token, google_requests.Request(), google_client_id)

            # Extract user info from the verified payload
            user_id = idinfo['sub']
            email = idinfo['email']
            name = idinfo.get('name', '')
            
            # Check if user exists in database
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM users WHERE google_sub = ? OR email = ?', (user_id, email))
            user = cursor.fetchone()
            
            if user:
                # Update last login
                cursor.execute('UPDATE users SET last_login = ? WHERE id = ?', 
                             (datetime.utcnow().isoformat(), user['id']))
                user_data = dict_from_row(user)
            else:
                # Create new user
                new_user_id = generate_id()
                cursor.execute('''
                    INSERT INTO users (id, email, name, role, created_at, last_login, google_sub)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (new_user_id, email, name, 'user', 
                      datetime.utcnow().isoformat(), 
                      datetime.utcnow().isoformat(), user_id))
                
                user_data = {
                    'id': new_user_id,
                    'email': email,
                    'name': name,
                    'role': 'user',
                    'created_at': datetime.utcnow().isoformat(),
                    'last_login': datetime.utcnow().isoformat(),
                    'google_sub': user_id
                }
            
            conn.commit()
            conn.close()
            
            return jsonify({
                'success': True,
                'user': user_data,
                'message': 'Authentication successful'
            })
            
        except ValueError as verify_err:
            # If the token isn't a valid Google ID token (e.g. during local
            # development when we send a base64-encoded JSON payload) fall back
            # to decoding the payload manually. **This should only be considered
            # a development-time convenience and SHOULD NOT be enabled in
            # production.**

            try:
                import base64, json

                # Ensure correct base64 padding
                padded = token + "=" * (-len(token) % 4)
                decoded = base64.b64decode(padded).decode()
                payload = json.loads(decoded)

                user_id = payload.get('sub') or payload.get('id')
                email = payload.get('email')
                name = payload.get('name', '')

                if not (user_id and email):
                    raise ValueError("Decoded fallback token missing required fields")

                logger.warning("Using fallback (unverified) Google token – running in DEV mode?")

            except Exception as fallback_err:
                logger.error(f"Invalid token: {verify_err}; Fallback decode also failed: {fallback_err}")
                return jsonify({'error': 'Invalid token'}), 401

            # continue with database operations using the unverified payload

            conn = get_db_connection()
            cursor = conn.cursor()

            cursor.execute('SELECT * FROM users WHERE google_sub = ? OR email = ?', (user_id, email))
            user = cursor.fetchone()

            # === Shared user-creation / update logic (executed after token payload
            #     has been extracted into user_id, email, name variables) ===

            if user:
                # Update last login
                cursor.execute('UPDATE users SET last_login = ? WHERE id = ?',
                               (datetime.utcnow().isoformat(), user['id']))
                user_data = dict_from_row(user)
            else:
                # Create new user
                new_user_id = generate_id()
                cursor.execute('''
                    INSERT INTO users (id, email, name, role, created_at, last_login, google_sub)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (new_user_id, email, name, 'user',
                      datetime.utcnow().isoformat(),
                      datetime.utcnow().isoformat(), user_id))

                user_data = {
                    'id': new_user_id,
                    'email': email,
                    'name': name,
                    'role': 'user',
                    'created_at': datetime.utcnow().isoformat(),
                    'last_login': datetime.utcnow().isoformat(),
                    'google_sub': user_id
                }

            conn.commit()
            conn.close()

            return jsonify({
                'success': True,
                'user': user_data,
                'message': 'Authentication successful'
            })

        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return jsonify({'error': 'Authentication failed'}), 500

    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return jsonify({'error': 'Authentication failed'}), 500

# User endpoints
@app.route('/users/stats', methods=['GET'])
def user_stats():
    """Get user statistics for admin dashboard"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get total users
        cursor.execute('SELECT COUNT(*) as total FROM users')
        total_users = cursor.fetchone()['total']
        
        # Get new users this week
        week_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
        cursor.execute('SELECT COUNT(*) as new_this_week FROM users WHERE created_at > ?', (week_ago,))
        new_this_week = cursor.fetchone()['new_this_week']
        
        # Get active users (logged in last 30 days)
        month_ago = (datetime.utcnow() - timedelta(days=30)).isoformat()
        cursor.execute('SELECT COUNT(*) as active FROM users WHERE last_login > ?', (month_ago,))
        active_users = cursor.fetchone()['active']
        
        # Get role distribution
        cursor.execute('SELECT role, COUNT(*) as count FROM users GROUP BY role')
        role_distribution = {row['role']: row['count'] for row in cursor.fetchall()}
        
        conn.close()
        
        return jsonify({
            'total_users': total_users,
            'new_this_week': new_this_week,
            'active_users': active_users,
            'role_distribution': role_distribution
        })
        
    except Exception as e:
        logger.error(f"Error getting user stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/users', methods=['GET'])
def get_users():
    """Get all users for admin management"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, email, name, role, created_at, last_login FROM users ORDER BY created_at DESC')
        users = [dict_from_row(row) for row in cursor.fetchall()]
        
        conn.close()
        return jsonify({'users': users})
        
    except Exception as e:
        logger.error(f"Error getting users: {e}")
        return jsonify({'error': str(e)}), 500



@app.route('/sat/solve', methods=['POST'])
def solve_sat():
    """
    Solve a SAT problem given in DIMACS format.
    Expects JSON: { "dimacs": "<DIMACS string>", "name": "<optional job name>" }
    Returns: Test ID and DIMACS-style output
    """
    try:
        data = request.get_json()
        dimacs_str = data.get('dimacs', '')
        job_name = data.get('name', '')
        
        if not dimacs_str.strip():
            return jsonify({'error': 'No DIMACS input provided'}), 400

        # Generate test ID and name
        test_id = generate_id()
        if not job_name:
            # Auto-generate name similar to LDPC
            job_name = f'SAT_test_{test_id[:8]}'
        
        # Create test record in database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Parse basic info from DIMACS
        num_vars = 0
        num_clauses = 0
        for line in dimacs_str.strip().split('\n'):
            if line.startswith('p cnf'):
                parts = line.split()
                if len(parts) >= 4:
                    num_vars = int(parts[2])
                    num_clauses = int(parts[3])
                break
        
        # Store initial test record
        cursor.execute('''
            INSERT INTO tests (id, name, chip_type, test_mode, environment, config, status, created, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            test_id,
            job_name,
            '3SAT',  # Using 3SAT as chip type for SAT problems
            'DIMACS Input',
            'software',  # or 'hardware' if using hardware acceleration
            json.dumps({
                'dimacs_input': dimacs_str,
                'num_variables': num_vars,
                'num_clauses': num_clauses
            }),
            'running',
            datetime.utcnow().isoformat(),
            json.dumps({
                'solver_type': 'pysat',
                'created_by': data.get('user_id', 'unknown')
            })
        ))
        conn.commit()
        
        # Solve the SAT problem
        start_time = time.time()
        try:
            # Parse DIMACS using PySAT
            cnf = CNF(from_string=dimacs_str)
            with Solver(bootstrap_with=cnf.clauses) as solver:
                sat = solver.solve()
                output = io.StringIO()
                
                if sat:
                    output.write("s SATISFIABLE\n")
                    model = solver.get_model()
                    # Write solution in DIMACS format (one line, ending with 0)
                    output.write("v " + " ".join(str(lit) for lit in model) + " 0\n")
                    solution_found = True
                    satisfying_assignment = model
                else:
                    output.write("s UNSATISFIABLE\n")
                    solution_found = False
                    satisfying_assignment = []
                
                dimacs_output = output.getvalue()
                
            solve_time = time.time() - start_time
            
            # Update test record with results
            cursor.execute('''
                UPDATE tests 
                SET status = ?, metadata = ?
                WHERE id = ?
            ''', (
                'completed',
                json.dumps({
                    'solver_type': 'pysat',
                    'created_by': data.get('user_id', 'unknown'),
                    'dimacs_output': dimacs_output,
                    'solve_time_seconds': solve_time,
                    'satisfiable': solution_found,
                    'num_variables': num_vars,
                    'num_clauses': num_clauses,
                    'satisfying_assignment': satisfying_assignment if solution_found else None
                }),
                test_id
            ))
            
            # Also store in test_results table for consistency
            cursor.execute('''
                INSERT INTO test_results (id, test_id, iteration, timestamp, results)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                generate_id(),
                test_id,
                1,
                datetime.utcnow().isoformat(),
                json.dumps({
                    'satisfiable': solution_found,
                    'solve_time': solve_time,
                    'assignment': satisfying_assignment if solution_found else None,
                    'output': dimacs_output
                })
            ))
            
            conn.commit()
            conn.close()
            
            return jsonify({
                'test_id': test_id,
                'output': dimacs_output,
                'satisfiable': solution_found,
                'solve_time': solve_time
            })
            
        except Exception as solve_error:
            # Update test status to error
            cursor.execute('''
                UPDATE tests 
                SET status = ?, metadata = ?
                WHERE id = ?
            ''', (
                'error',
                json.dumps({
                    'error': str(solve_error),
                    'dimacs_input': dimacs_str
                }),
                test_id
            ))
            conn.commit()
            conn.close()
            
            return jsonify({
                'test_id': test_id,
                'error': f'Failed to solve SAT: {str(solve_error)}',
                'output': f'c Error: {str(solve_error)}'
            }), 500
            
    except Exception as e:
        return jsonify({'error': f'Failed to process request: {str(e)}'}), 500

@app.route('/users/<user_id>', methods=['PUT'])
def update_user(user_id):
    """Update user role and details"""
    try:
        data = request.get_json()
        role = data.get('role')
        
        if role not in ['user', 'admin', 'moderator']:
            return jsonify({'error': 'Invalid role'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('UPDATE users SET role = ? WHERE id = ?', (role, user_id))
        
        if cursor.rowcount == 0:
            conn.close()
            return jsonify({'error': 'User not found'}), 404
        
        conn.commit()
        conn.close()
        
        return jsonify({'message': 'User updated successfully'})
        
    except Exception as e:
        logger.error(f"Error updating user: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/users/<user_id>', methods=['DELETE'])
def delete_user(user_id):
    """Delete a user"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
        
        if cursor.rowcount == 0:
            conn.close()
            return jsonify({'error': 'User not found'}), 404
        
        conn.commit()
        conn.close()
        
        return jsonify({'message': 'User deleted successfully'})
        
    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        return jsonify({'error': str(e)}), 500

# System settings endpoints
@app.route('/system/settings', methods=['GET', 'POST'])
def system_settings():
    """Get or update system settings"""
    try:
        if request.method == 'GET':
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Create system_settings table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    updated_by TEXT NOT NULL
                )
            ''')
            
            # Get all system settings
            cursor.execute('SELECT key, value FROM system_settings')
            settings = {row['key']: row['value'] for row in cursor.fetchall()}
            conn.close()
            return jsonify({'settings': settings})
        
        else:  # POST
            data = request.get_json()
            settings = data.get('settings', {})
            user_id = data.get('user_id', 'unknown')
            
            conn = get_db_connection()
            cursor = conn.cursor()
            
            for key, value in settings.items():
                cursor.execute('''
                    INSERT OR REPLACE INTO system_settings (key, value, updated_at, updated_by)
                    VALUES (?, ?, ?, ?)
                ''', (key, str(value), datetime.utcnow().isoformat(), user_id))
            
            conn.commit()
            conn.close()
            
            return jsonify({'message': 'Settings updated successfully'})
            
    except Exception as e:
        logger.error(f"Error handling system settings: {e}")
        return jsonify({'error': str(e)}), 500

# Announcements endpoints
@app.route('/announcements', methods=['GET', 'POST'])
def announcements():
    """Get active announcements or create new ones"""
    try:
        if request.method == 'GET':
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get active announcements that haven't expired
            cursor.execute('''
                SELECT * FROM announcements 
                WHERE active = 1 AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY created_at DESC
            ''', (datetime.utcnow().isoformat(),))
            
            announcements = [dict_from_row(row) for row in cursor.fetchall()]
            conn.close()
            
            return jsonify({'announcements': announcements})
        
        else:  # POST
            data = request.get_json()
            message = data.get('message')
            announcement_type = data.get('type', 'info')
            expires_at = data.get('expires_at')
            created_by = data.get('created_by', 'unknown')
            
            if not message:
                return jsonify({'error': 'Message is required'}), 400
            
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO announcements (id, message, type, expires_at, created_at, created_by, active)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (generate_id(), message, announcement_type, expires_at, 
                  datetime.utcnow().isoformat(), created_by, 1))
            
            conn.commit()
            conn.close()
            
            return jsonify({'message': 'Announcement created successfully'})
            
    except Exception as e:
        logger.error(f"Error handling announcements: {e}")
        return jsonify({'error': str(e)}), 500

# Tests endpoints
@app.route('/tests', methods=['GET', 'POST'])
def handle_tests():
    """Handle test operations - GET to list tests, POST to create new test"""
    if request.method == 'GET':
        try:
            # Get query parameters for filtering
            chip_type = request.args.get('chip_type')
            status = request.args.get('status')
            limit = int(request.args.get('limit', 50))
            offset = int(request.args.get('offset', 0))
            
            # Build query
            query = 'SELECT * FROM tests'
            params = []
            conditions = []
            
            if chip_type:
                conditions.append('chip_type = ?')
                params.append(chip_type)
            
            if status:
                conditions.append('status = ?')
                params.append(status)
            
            if conditions:
                query += ' WHERE ' + ' AND '.join(conditions)
            
            query += ' ORDER BY created DESC LIMIT ? OFFSET ?'
            params.extend([limit, offset])
            
            # Execute query
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(query, params)
            tests = [dict_from_row(row) for row in cursor.fetchall()]
            
            # Get total count
            count_query = 'SELECT COUNT(*) FROM tests'
            count_params = []
            if conditions:
                count_query += ' WHERE ' + ' AND '.join(conditions)
                count_params = params[:-2]  # Exclude limit and offset
            
            cursor.execute(count_query, count_params)
            total_count = cursor.fetchone()[0]
            
            conn.close()
            
            # Parse metadata for each test
            for test in tests:
                try:
                    if test['metadata']:
                        test['metadata'] = json.loads(test['metadata'])
                    else:
                        test['metadata'] = {}
                except:
                    test['metadata'] = {}
                    
            return jsonify({
                'tests': tests,
                'total_count': total_count,
                'limit': limit,
                'offset': offset,
                'has_more': offset + len(tests) < total_count
            })
            
        except Exception as e:
            logger.error(f"Error getting tests: {e}")
            return jsonify({'error': str(e)}), 500

    elif request.method == 'POST':
        try:
            data = request.get_json()
            
            # Validate required fields
            required_fields = ['name', 'chip_type']
            for field in required_fields:
                if field not in data:
                    return jsonify({'error': f'Missing required field: {field}'}), 400
            
            # Create test record
            test_id = generate_id()
            test_data = {
                'id': test_id,
                'name': data['name'],
                'chip_type': data['chip_type'],
                'test_mode': data.get('test_mode', 'standard'),
                'environment': data.get('environment', 'lab'),
                'config': json.dumps(data.get('config', {})),
                'status': 'created',
                'created': datetime.utcnow().isoformat(),
                'metadata': json.dumps(data.get('metadata', {}))
            }
            
            # Insert into database
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO tests (id, name, chip_type, test_mode, environment, config, status, created, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                test_data['id'], test_data['name'], test_data['chip_type'],
                test_data['test_mode'], test_data['environment'], test_data['config'],
                test_data['status'], test_data['created'], test_data['metadata']
            ))
            conn.commit()
            conn.close()
            
            # Parse back the metadata for response
            test_data['config'] = json.loads(test_data['config'])
            test_data['metadata'] = json.loads(test_data['metadata'])
            
            logger.info(f"Created new test: {test_id} - {data['name']}")
            return jsonify(test_data), 201
            
        except Exception as e:
            logger.error(f"Error creating test: {e}")
            return jsonify({'error': str(e)}), 500

# Individual test detail endpoint with DELETE support
@app.route('/tests/<test_id>', methods=['GET', 'DELETE'])
def handle_test_detail(test_id):
    """Get or delete details for a specific test"""
    try:
        if request.method == 'GET':
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM tests WHERE id = ?', (test_id,))
            test = cursor.fetchone()
            
            if not test:
                conn.close()
                return jsonify({'error': 'Test not found'}), 404
            
            test_data = dict_from_row(test)
            
            # Parse metadata
            try:
                if test_data['metadata']:
                    test_data['metadata'] = json.loads(test_data['metadata'])
                else:
                    test_data['metadata'] = {}
            except:
                test_data['metadata'] = {}
                
            try:
                if test_data['config']:
                    test_data['config'] = json.loads(test_data['config'])
                else:
                    test_data['config'] = {}
            except:
                test_data['config'] = {}
            
            # Get test results if any
            cursor.execute('SELECT * FROM test_results WHERE test_id = ? ORDER BY timestamp DESC', (test_id,))
            results = [dict_from_row(row) for row in cursor.fetchall()]
            
            # Parse results JSON
            for result in results:
                try:
                    if result['results']:
                        result['results'] = json.loads(result['results'])
                    else:
                        result['results'] = {}
                except:
                    result['results'] = {}
            
            test_data['results'] = results
            
            conn.close()
            return jsonify(test_data)
            
        else:  # DELETE
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Check if test exists
            cursor.execute('SELECT id FROM tests WHERE id = ?', (test_id,))
            if not cursor.fetchone():
                conn.close()
                return jsonify({'error': 'Test not found'}), 404
            
            # Delete test and related data (cascading)
            cursor.execute('DELETE FROM test_results WHERE test_id = ?', (test_id,))
            cursor.execute('DELETE FROM files WHERE test_id = ?', (test_id,))
            cursor.execute('DELETE FROM large_data WHERE test_id = ?', (test_id,))
            cursor.execute('DELETE FROM tests WHERE id = ?', (test_id,))
            
            conn.commit()
            conn.close()
            
            return jsonify({'message': 'Test deleted successfully'})
        
    except Exception as e:
        logger.error(f"Error handling test detail: {e}")
        return jsonify({'error': str(e)}), 500

# API Keys endpoints
@app.route('/api-keys', methods=['GET', 'POST'])
def handle_api_keys():
    """Handle API key operations"""
    try:
        if request.method == 'GET':
            # Get user's API keys
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # For now, return empty list since we don't have user auth context
            cursor.execute('SELECT id, name, created_at, last_used_at, expires_at FROM api_keys ORDER BY created_at DESC')
            keys = [dict_from_row(row) for row in cursor.fetchall()]
            
            conn.close()
            return jsonify({'api_keys': keys})
        
        else:  # POST
            data = request.get_json()
            name = data.get('name', 'Default API Key')
            
            # Generate API key
            import secrets
            api_key = f"dacroq_{secrets.token_urlsafe(32)}"
            key_hash = secrets.token_hex(16)
            
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO api_keys (id, user_id, name, key_hash, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (generate_id(), 'default-user', name, key_hash, datetime.utcnow().isoformat()))
            
            conn.commit()
            conn.close()
            
            return jsonify({
                'api_key': api_key,
                'name': name,
                'created_at': datetime.utcnow().isoformat()
            }), 201
            
    except Exception as e:
        logger.error(f"Error handling API keys: {e}")
        return jsonify({'error': str(e)}), 500

# LDPC endpoints
@app.route('/ldpc/jobs', methods=['GET', 'POST'])
def handle_ldpc_jobs():
    """Handle LDPC job operations"""
    try:
        if request.method == 'GET':
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM ldpc_jobs ORDER BY created DESC')
            jobs = [dict_from_row(row) for row in cursor.fetchall()]
            
            # Parse JSON fields and extract config values to top level
            for job in jobs:
                try:
                    if job['config']:
                        config = json.loads(job['config'])
                        job['config'] = config
                        # Extract key fields to top level for frontend
                        job['algorithm_type'] = config.get('algorithm_type', 'digital_hardware')
                        job['test_mode'] = config.get('test_mode', 'unknown')
                        job['noise_level'] = config.get('noise_level', 0)
                        job['message_content'] = config.get('message_content', '')
                    else:
                        job['algorithm_type'] = 'digital_hardware'
                        job['test_mode'] = 'unknown'
                        job['noise_level'] = 0
                        job['message_content'] = ''
                        
                    if job['results']:
                        job['results'] = json.loads(job['results'])
                    if job['metadata']:
                        metadata = json.loads(job['metadata'])
                        job['metadata'] = metadata
                        # Extract key metadata to top level
                        job['correction_successful'] = metadata.get('correction_successful', False)
                        job['original_message'] = metadata.get('original_message', '')
                        job['corrupted_message'] = metadata.get('corrupted_message', '')
                        job['decoded_message'] = metadata.get('decoded_message', '')
                        if 'test_statistics' in metadata:
                            stats = metadata['test_statistics']
                            job['success_rate'] = stats.get('success_rate', 0)
                            job['total_execution_time'] = stats.get('avg_execution_time_ms', 0) / 1000  # Convert to seconds
                except Exception as e:
                    logger.warning(f"Error parsing job data for {job.get('id', 'unknown')}: {e}")
                    # Set defaults if parsing fails
                    job['algorithm_type'] = 'digital_hardware'
                    job['test_mode'] = 'unknown'
                    job['noise_level'] = 0
                    job['message_content'] = ''
                    job['correction_successful'] = False
            
            conn.close()
            return jsonify({'jobs': jobs})
        
        else:  # POST
            data = request.get_json()
            
            job_id = generate_id()
            
            # Get job parameters
            algorithm_type = data.get('algorithm_type', 'digital_hardware')
            test_mode = data.get('test_mode', 'custom_message')
            message_content = data.get('message_content', 'Hello LDPC!')
            noise_level_percent = data.get('noise_level', 10)  # Noise level as percentage (0-50%)
            
            # Convert noise level percentage to SNR in dB
            # 0% noise = very high SNR (30dB), 50% noise = very low SNR (0dB)
            if noise_level_percent == 0:
                snr_db = 30  # Perfect channel
            else:
                # Map noise percentage to SNR: 10% -> ~10dB, 20% -> ~5dB, 50% -> 0dB
                snr_db = max(0, 15 - (noise_level_percent * 0.3))
            
            try:
                # Use zero codeword for BER testing - matches MATLAB approach
                # In MATLAB: zerocode = true; RX_id = zeros(lenSNR, numTest, n);
                if test_mode == 'custom_message' and message_content:
                    # For demonstration, encode the custom message
                    message_bytes = message_content.encode('utf-8')[:6]  # Limit to 6 bytes = 48 bits
                    message_bytes = message_bytes.ljust(6, b'\x00')  # Pad to 6 bytes
                    info_bits = np.unpackbits(np.frombuffer(message_bytes, dtype=np.uint8))
                    # Encode to get actual codeword
                    test_codeword = ldpc_codec.encode(info_bits)
                elif test_mode == 'random_string':
                    # Generate random information bits
                    np.random.seed(sum(ord(c) for c in job_id.split('-')[0]))
                    info_bits = np.random.randint(0, 2, ldpc_codec.k)
                    test_codeword = ldpc_codec.encode(info_bits)
                else:
                    # Use all-zero codeword for standard BER testing (matches MATLAB)
                    info_bits = np.zeros(ldpc_codec.k, dtype=int)
                    test_codeword = np.zeros(ldpc_codec.n, dtype=int)  # All-zero is always a valid codeword
                
                # Add noise and get received signal
                received_bits, llrs = ldpc_codec.add_noise(test_codeword, snr_db)
                
                # Decode using appropriate algorithm
                if algorithm_type == 'analog_hardware':
                    # Use oscillator-based decoder simulation
                    decoded_bits, success, tts_ns, energy_pj = ldpc_codec.simulate_oscillator_decoder(llrs, job_id)
                    decode_time_ms = tts_ns / 1e6  # Convert ns to ms
                    power_consumption = energy_pj / (tts_ns / 1e9) / 1e12 * 1e3  # Convert to mW
                    iterations_used = 1  # One-shot decoding
                else:
                    # Use digital belief propagation decoder
                    decoded_bits, success, iterations_used, bit_errors = ldpc_codec.decode_belief_propagation(llrs, max_iterations=10)
                    decode_time_ms = iterations_used * 0.5  # Estimate 0.5ms per iteration
                    power_consumption = 500  # Typical digital decoder power in mW
                    energy_pj = power_consumption * decode_time_ms * 1e-3 * 1e12 / ldpc_codec.k  # pJ/bit
                
                # Calculate performance metrics (compare info bits, not full codeword)
                info_bit_errors = np.sum(info_bits != decoded_bits[:ldpc_codec.k])
                frame_error = info_bit_errors > 0
                ber = info_bit_errors / ldpc_codec.k
                fer = 1.0 if frame_error else 0.0
                
                # For display purposes
                if test_mode == 'custom_message':
                    try:
                        decoded_bytes = np.packbits(decoded_bits[:ldpc_codec.k]).tobytes()
                        decoded_message = decoded_bytes.decode('utf-8', errors='ignore').rstrip('\x00')
                        original_message = message_content
                    except:
                        decoded_message = ''.join(map(str, decoded_bits[:ldpc_codec.k]))
                        original_message = message_content
                else:
                    # For zero codeword testing
                    original_message = 'All-zero codeword (BER test)'
                    decoded_message = f'Decoded: {np.sum(decoded_bits[:ldpc_codec.k])} errors out of {ldpc_codec.k} bits'
                
                # Create corrupted message representation
                corrupted_info_bits = received_bits[:ldpc_codec.k]
                corrupted_errors = np.sum(info_bits != corrupted_info_bits)
                corrupted_message = f'Channel errors: {corrupted_errors} out of {ldpc_codec.k} bits'
                
                # Generate detailed test results using the actual message
                results_data = []
                total_success = 0
                total_bit_errors = 0
                total_time = 0
                total_iterations = 0
                
                for i in range(10):  # Generate 10 test runs for display
                    # Use the same codeword for all test runs to show consistency
                    # Add slight SNR variation to simulate real-world conditions
                    test_snr = snr_db + (i - 5) * 0.2  # Vary SNR slightly around the target
                    
                    # Use deterministic seeded random for consistent results
                    seed_val = sum(ord(c) for c in job_id.split('-')[0]) + i
                    np.random.seed(seed_val % 1000)
                    
                    # Use the same codeword for testing
                    test_received, test_llrs = ldpc_codec.add_noise(test_codeword, test_snr)
                    
                    if algorithm_type == 'analog_hardware':
                        test_decoded, test_success, test_tts, test_energy = ldpc_codec.simulate_oscillator_decoder(test_llrs, f"{job_id}_{i}")
                        test_time = test_tts / 1e6  # ns to ms
                        test_power = test_energy / (test_tts / 1e9) / 1e12 * 1e3  # mW
                        test_iterations = 1
                    else:
                        test_decoded, test_success, test_iterations, _ = ldpc_codec.decode_belief_propagation(test_llrs)
                        test_time = test_iterations * 0.1  # More realistic 0.1ms per iteration
                        test_power = 50  # More realistic digital decoder power in mW
                    
                    test_bit_errors = np.sum(info_bits != test_decoded[:ldpc_codec.k])
                    
                    # Accumulate statistics
                    if test_success:
                        total_success += 1
                    total_bit_errors += test_bit_errors
                    total_time += test_time
                    total_iterations += test_iterations
                    
                    # Convert numpy types to Python native types for JSON serialization
                    results_data.append({
                        'run': i + 1,
                        'snr': f'{test_snr:.1f}dB',
                        'success': bool(test_success),
                        'execution_time': float(round(test_time, 3)),
                        'iterations': int(test_iterations),
                        'bit_errors': int(test_bit_errors),
                        'power_consumption': float(round(test_power, 1)),
                        'test_file': f'ldpc_test_vector_{i}.bin'
                    })
                
                # Calculate overall performance metrics
                success_rate = total_success / 10
                avg_bit_errors = total_bit_errors / 10
                avg_time = total_time / 10
                avg_iterations = total_iterations / 10
                
                job_data = {
                    'id': job_id,
                    'name': data.get('name', f'LDPC Job {job_id[:8]}'),
                    'job_type': 'ldpc_error_correction',
                    'config': json.dumps({
                        'algorithm_type': algorithm_type,
                        'test_mode': test_mode,
                        'message_content': message_content,
                        'noise_level': noise_level_percent,
                        'snr_db': float(snr_db),
                        'code_params': f'({ldpc_codec.n},{ldpc_codec.k})',
                        'code_rate': float(ldpc_codec.rate)
                    }),
                    'status': 'completed',
                    'created': datetime.utcnow().isoformat(),
                    'started': datetime.utcnow().isoformat(),
                    'completed': datetime.utcnow().isoformat(),
                    'results': json.dumps(results_data),
                    'progress': 100.0,
                    'current_step': 'completed',
                    'total_steps': 10,
                    'metadata': json.dumps({
                        'original_message': original_message,
                        'corrupted_message': corrupted_message,
                        'decoded_message': decoded_message,
                        'correction_successful': bool(success),
                        'bit_error_rate': float(ber),
                        'frame_error_rate': float(fer),
                        'noise_level_percent': noise_level_percent,
                        'snr_db': float(snr_db),
                        'decode_time_ms': float(decode_time_ms),
                        'iterations_used': int(iterations_used),
                        'energy_per_bit_pj': float(energy_pj / ldpc_codec.k if algorithm_type == 'analog_hardware' else avg_time * 50 * 1e-3 * 1e12 / ldpc_codec.k),
                        'power_consumption_mw': float(power_consumption),
                        'test_statistics': {
                            'total_tests': 10,
                            'successful_tests': int(total_success),
                            'success_rate': float(success_rate),
                            'avg_bit_errors': float(avg_bit_errors),
                            'avg_execution_time_ms': float(avg_time),
                            'avg_iterations': float(avg_iterations),
                            'total_bit_errors': int(total_bit_errors)
                        },
                        'algorithm_performance': {
                            'energy_efficiency_pj_per_bit': 5.47 if algorithm_type == 'analog_hardware' else float(avg_time * 50 * 1e-3 * 1e12 / ldpc_codec.k),
                            'time_to_solution_ns': float(tts_ns if algorithm_type == 'analog_hardware' else avg_time * 1e6),
                            'convergence_rate': float(success_rate),
                            'avg_power_consumption_mw': 50.0 if algorithm_type == 'digital_hardware' else float(power_consumption)
                        }
                    })
                }
                
            except Exception as e:
                logger.error(f"Error in LDPC encoding/decoding: {e}")
                # Fallback to simple mock data
                job_data = {
                    'id': job_id,
                    'name': data.get('name', f'LDPC Job {job_id[:8]}'),
                    'job_type': 'ldpc_error_correction',
                    'config': json.dumps({
                        'algorithm_type': algorithm_type,
                        'test_mode': test_mode,
                        'message_content': message_content,
                        'noise_level': noise_level_percent,
                        'snr_db': float(snr_db if 'snr_db' in locals() else 10)
                    }),
                    'status': 'error',
                    'created': datetime.utcnow().isoformat(),
                    'started': datetime.utcnow().isoformat(),
                    'completed': datetime.utcnow().isoformat(),
                    'results': json.dumps([]),
                    'progress': 0.0,
                    'current_step': 'error',
                    'total_steps': 10,
                    'metadata': json.dumps({'error': str(e)})
                }
            
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO ldpc_jobs (id, name, job_type, config, status, created, started, completed, results, progress, current_step, total_steps, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                job_data['id'], job_data['name'], job_data['job_type'], job_data['config'],
                job_data['status'], job_data['created'], job_data['started'], job_data['completed'],
                job_data['results'], job_data['progress'], job_data['current_step'], job_data['total_steps'],
                job_data['metadata']
            ))
            conn.commit()
            conn.close()
            
            return jsonify({'message': 'LDPC job created successfully', 'job_id': job_id}), 201
            
    except Exception as e:
        logger.error(f"Error handling LDPC jobs: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/ldpc/jobs/<job_id>', methods=['GET', 'DELETE'])
def handle_ldpc_job_detail(job_id):
    """Handle individual LDPC job operations"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if request.method == 'GET':
            cursor.execute('SELECT * FROM ldpc_jobs WHERE id = ?', (job_id,))
            job = cursor.fetchone()
            
            if not job:
                conn.close()
                return jsonify({'error': 'Job not found'}), 404
            
            job_data = dict_from_row(job)
            
            # Parse JSON fields
            try:
                if job_data['config']:
                    job_data['config'] = json.loads(job_data['config'])
                if job_data['results']:
                    job_data['results'] = json.loads(job_data['results'])
                if job_data['metadata']:
                    job_data['metadata'] = json.loads(job_data['metadata'])
            except:
                pass
            
            conn.close()
            return jsonify(job_data)
        
        else:  # DELETE
            cursor.execute('DELETE FROM ldpc_jobs WHERE id = ?', (job_id,))
            
            if cursor.rowcount == 0:
                conn.close()
                return jsonify({'error': 'Job not found'}), 404
            
            conn.commit()
            conn.close()
            return jsonify({'message': 'Job deleted successfully'})
            
    except Exception as e:
        logger.error(f"Error handling LDPC job detail: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/ldpc/jobs/<job_id>/progress', methods=['GET'])
def get_ldpc_job_progress(job_id):
    """Get progress for a running LDPC job"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT progress, current_step, total_steps FROM ldpc_jobs WHERE id = ?', (job_id,))
        job = cursor.fetchone()
        
        if not job:
            conn.close()
            return jsonify({'error': 'Job not found'}), 404
        
        progress_data = {
            'progress_percent': job['progress'] or 0,
            'completed_runs': int((job['progress'] or 0) * (job['total_steps'] or 100) / 100),
            'total_runs': job['total_steps'] or 100,
            'current_test': job['current_step'] or 'processing',
            'estimated_time_remaining': max(0, (100 - (job['progress'] or 0)) * 0.1),  # Simulate ETA
            'success_rate': 0.95
        }
        
        conn.close()
        return jsonify(progress_data)
        
    except Exception as e:
        logger.error(f"Error getting LDPC job progress: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/ldpc/jobs/<job_id>/execute', methods=['POST'])
def execute_ldpc_job(job_id):
    """Execute an LDPC job"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('UPDATE ldpc_jobs SET status = ?, started = ? WHERE id = ?', 
                      ('running', datetime.utcnow().isoformat(), job_id))
        
        if cursor.rowcount == 0:
            conn.close()
            return jsonify({'error': 'Job not found'}), 404
        
        conn.commit()
        conn.close()
        
        return jsonify({'message': 'Job execution started'})
        
    except Exception as e:
        logger.error(f"Error executing LDPC job: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/ldpc/jobs/<job_id>/stop', methods=['POST'])
def stop_ldpc_job(job_id):
    """Stop a running LDPC job"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('UPDATE ldpc_jobs SET status = ? WHERE id = ?', ('stopped', job_id))
        
        if cursor.rowcount == 0:
            conn.close()
            return jsonify({'error': 'Job not found'}), 404
        
        conn.commit()
        conn.close()
        
        return jsonify({'message': 'Job stopped successfully'})
        
    except Exception as e:
        logger.error(f"Error stopping LDPC job: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/ldpc/hardware/config', methods=['GET'])
def get_ldpc_hardware_config():
    """Get LDPC hardware configuration"""
    try:
        # Return mock hardware configuration
        config = {
            'clock_config': {
                'external_clock': True,
                'internal_frequency': '100MHz',
                'clock_divider': 4
            },
            'vref_voltages': [1.2, 1.8, 2.5, 3.3],
            'board_type': 'AMORGOS',
            'firmware_version': '1.2.3',
            'connected': len(PORT_TO_BOARD_MAP) > 0
        }
        
        return jsonify(config)
        
    except Exception as e:
        logger.error(f"Error getting LDPC hardware config: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/ldpc/test-data', methods=['GET'])
def get_ldpc_test_data():
    """Get available LDPC test data"""
    try:
        # Check if LDPC data directory exists
        if LDPC_DATA_DIR.exists():
            files = []
            for file_path in LDPC_DATA_DIR.rglob('*'):
                if file_path.is_file():
                    files.append({
                        'name': file_path.name,
                        'size': file_path.stat().st_size,
                        'type': 'LDPC Test Data',
                        'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                    })
            
            return jsonify({
                'data_directory': str(LDPC_DATA_DIR),
                'available_test_files': files
            })
        else:
            return jsonify({
                'data_directory': str(LDPC_DATA_DIR),
                'available_test_files': []
            })
        
    except Exception as e:
        logger.error(f"Error getting LDPC test data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/ldpc/generate-data', methods=['POST'])
def generate_ldpc_test_data():
    """Generate new LDPC test data"""
    try:
        data = request.get_json()
        noise_level = data.get('noise_level', 10)
        num_tests = data.get('num_tests', 1000)
        
        # Ensure LDPC data directory exists
        LDPC_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Generate mock test data file
        test_file = LDPC_DATA_DIR / f'test_data_{noise_level}dB_{num_tests}tests.json'
        
        mock_data = {
            'noise_level': noise_level,
            'num_tests': num_tests,
            'generated_at': datetime.utcnow().isoformat(),
            'test_vectors': [
                {
                    'id': i,
                    'original_bits': [random.randint(0, 1) for _ in range(96)],
                    'corrupted_bits': [random.randint(0, 1) for _ in range(96)],
                    'expected_correction': True
                }
                for i in range(min(10, num_tests))  # Only store first 10 for demo
            ]
        }
        
        with open(test_file, 'w') as f:
            json.dump(mock_data, f, indent=2)
        
        return jsonify({
            'message': f'Generated {num_tests} test vectors with {noise_level}% noise',
            'file_path': str(test_file)
        })
        
    except Exception as e:
        logger.error(f"Error generating LDPC test data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/ldpc/performance', methods=['GET'])
def get_ldpc_performance():
    """Get LDPC performance data"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, name, config, results, created 
            FROM ldpc_jobs 
            WHERE status = 'completed' 
            ORDER BY created DESC 
            LIMIT 10
        ''')
        
        jobs = cursor.fetchall()
        performance_data = []
        
        for job in jobs:
            try:
                config = json.loads(job['config']) if job['config'] else {}
                results = json.loads(job['results']) if job['results'] else {}
                
                performance_data.append({
                    'job_id': job['id'],
                    'job_name': job['name'],
                    'algorithm_type': config.get('algorithm_type', 'unknown'),
                    'test_mode': config.get('test_mode', 'unknown'),
                    'noise_level': config.get('noise_level', 0),
                    'success_rate': results.get('success_rate', 0),
                    'avg_execution_time': results.get('total_execution_time', 0),
                    'created': job['created']
                })
            except:
                continue
        
        conn.close()
        
        return jsonify({
            'performance_data': performance_data
        })
        
    except Exception as e:
        logger.error(f"Error getting LDPC performance: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/ldpc/led/control', methods=['POST'])
def control_ldpc_led():
    """Control LED patterns on LDPC hardware"""
    try:
        data = request.get_json()
        command = data.get('command', 'off')
        
        # Mock LED control - in real implementation, this would send commands to hardware
        logger.info(f"LED control command: {command}")
        
        return jsonify({
            'message': f'LED pattern "{command}" activated',
            'command': command,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error controlling LDPC LED: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/ldpc/led/test', methods=['POST'])
def test_ldpc_led_patterns():
    """Test all LED patterns"""
    try:
        patterns = ['idle', 'received', 'running', 'completed', 'error', 'off']
        
        logger.info("Testing all LED patterns")
        
        return jsonify({
            'message': 'LED pattern test sequence completed',
            'patterns_tested': patterns,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error testing LDPC LED patterns: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/ldpc/debug/ports', methods=['GET'])
def debug_ldpc_ports():
    """Debug LDPC hardware ports"""
    try:
        return jsonify({
            'detected_ports': list(PORT_TO_BOARD_MAP.keys()),
            'board_mappings': PORT_TO_BOARD_MAP,
            'ldpc_boards': [port for port, board_type in PORT_TO_BOARD_MAP.items() if board_type == 'ldpc']
        })
        
    except Exception as e:
        logger.error(f"Error debugging LDPC ports: {e}")
        return jsonify({'error': str(e)}), 500

def register_routes():
    """Register all route modules with the Flask app"""
    print("🔄 Registering route modules...")
    route_count = len(list(app.url_map.iter_rules()))
    print(f"✅ Routes registered successfully. Total: {route_count}")
    return route_count

# Register routes immediately when module is imported
_routes_registered = False
if not _routes_registered:
    register_routes()
    _routes_registered = True

# --- LDPC Implementation Based on Research Paper ---
class LDPCCodec:
    """
    LDPC Encoder/Decoder implementation based on the research paper:
    "A Relaxation Oscillator-Based Probabilistic Combinatorial Optimization Engine 
    for Soft Decoding of LDPC Codes" by Dikopoulos et al.
    
    Implements a (96,48) regular LDPC code as described in the paper.
    """
    
    def __init__(self):
        # Code parameters from the paper - matches MacKay_96_3_963.mat
        self.n = 96  # Codeword length
        self.k = 48  # Information bits
        self.rate = self.k / self.n  # Code rate = 0.5
        
        # Generate the MacKay (96,48) LDPC code H-matrix
        # This should ideally load from MacKay_96_3_963.mat but we'll generate equivalent
        self.H = self._generate_mackay_ldpc_matrix()
        self.G = self._generate_generator_matrix()
        
    def _generate_mackay_ldpc_matrix(self):
        """Generate a MacKay-style (96,48) LDPC parity check matrix"""
        # Create a regular LDPC code similar to MacKay's construction
        # Column weight 3, row weight 6 - optimized for (96,48) code
        H = np.zeros((48, 96), dtype=int)
        
        # Use MacKay-style progressive edge-growth algorithm approximation
        col_weight = 3
        row_weight = 6
        
        # Fixed seed for reproducible MacKay-equivalent matrix
        np.random.seed(963)  # Using part of MacKay's naming convention
        
        # Progressive construction: avoid short cycles
        for col in range(96):
            # For each column, find 3 rows to connect
            attempts = 0
            placed = 0
            
            while placed < col_weight and attempts < 100:
                # Select a random row that isn't overloaded
                available_rows = [r for r in range(48) if np.sum(H[r, :]) < row_weight]
                
                if available_rows:
                    row = np.random.choice(available_rows)
                    
                    # Check for short cycles (length 4) by seeing if this creates
                    # a 2x2 all-ones submatrix with existing connections
                    creates_cycle = False
                    for other_col in range(col):
                        if H[row, other_col] == 1:
                            # Check if any other row connected to both columns
                            for other_row in range(48):
                                if other_row != row and H[other_row, col] == 1 and H[other_row, other_col] == 1:
                                    creates_cycle = True
                                    break
                            if creates_cycle:
                                break
                    
                    if not creates_cycle:
                        H[row, col] = 1
                        placed += 1
                
                attempts += 1
            
            # If we couldn't place all edges without cycles, place remaining randomly
            if placed < col_weight:
                available_rows = [r for r in range(48) if np.sum(H[r, :]) < row_weight and H[r, col] == 0]
                remaining = min(col_weight - placed, len(available_rows))
                if remaining > 0:
                    selected_rows = np.random.choice(available_rows, remaining, replace=False)
                    for row in selected_rows:
                        H[row, col] = 1
        
        # Verify properties
        actual_col_weights = np.sum(H, axis=0)
        actual_row_weights = np.sum(H, axis=1)
        
        logger.info(f"MacKay-style LDPC Matrix generated:")
        logger.info(f"  Avg col weight: {np.mean(actual_col_weights):.2f} (target: {col_weight})")
        logger.info(f"  Avg row weight: {np.mean(actual_row_weights):.2f} (target: {row_weight})")
        logger.info(f"  Total edges: {np.sum(H)}")
        
        return H
    
    def _generate_generator_matrix(self):
        """Generate generator matrix G from H matrix using Gaussian elimination"""
        # For systematic encoding, we need G = [I_k | P] where H = [P^T | I_{n-k}]
        # This is a simplified approach - in practice, more sophisticated methods are used
        H_systematic = self._to_systematic_form(self.H.copy())
        
        # Extract P^T from systematic H = [P^T | I]
        P_T = H_systematic[:, :self.k]
        
        # Generator matrix G = [I | P]
        I_k = np.eye(self.k, dtype=int)
        G = np.hstack([I_k, P_T.T])
        
        return G
    
    def _to_systematic_form(self, H):
        """Convert H matrix to systematic form [P^T | I] using Gaussian elimination"""
        H_sys = H.copy()
        m, n = H_sys.shape
        
        # Gaussian elimination over GF(2)
        for i in range(min(m, n-self.k)):
            # Find pivot
            pivot_row = None
            for j in range(i, m):
                if H_sys[j, n-m+i] == 1:
                    pivot_row = j
                    break
            
            if pivot_row is not None:
                # Swap rows
                if pivot_row != i:
                    H_sys[[i, pivot_row]] = H_sys[[pivot_row, i]]
                
                # Eliminate
                for j in range(m):
                    if j != i and H_sys[j, n-m+i] == 1:
                        H_sys[j] = (H_sys[j] + H_sys[i]) % 2
        
        return H_sys
    
    def encode(self, info_bits):
        """
        Encode information bits using LDPC code
        
        Args:
            info_bits: numpy array of k information bits (0s and 1s)
            
        Returns:
            numpy array of n codeword bits
        """
        if len(info_bits) != self.k:
            raise ValueError(f"Information bits must be length {self.k}, got {len(info_bits)}")
        
        # Systematic encoding: codeword = info_bits * G
        codeword = np.dot(info_bits, self.G) % 2
        return codeword.astype(int)
    
    def add_noise(self, codeword, snr_db):
        """
        Add AWGN noise to codeword and compute LLRs - matches MATLAB approach
        
        Args:
            codeword: numpy array of codeword bits (0s and 1s)
            snr_db: Signal-to-noise ratio in dB
            
        Returns:
            tuple: (received_bits, llrs)
                - received_bits: hard decision bits
                - llrs: Log-likelihood ratios for soft decoding
        """
        # BPSK modulation: convert {0,1} to {+1,-1} like MATLAB: (RX_id .* -2) + 1
        # But in MATLAB: RX_mod = (RX_id .* -2) + 1, where RX_id is {0,1}
        # So: 0 -> +1, 1 -> -1 (this is actually the standard BPSK mapping)
        bpsk_symbols = 1 - 2 * codeword  # 0 -> +1, 1 -> -1
        
        # Add AWGN noise - matches MATLAB awgn() function
        # In MATLAB: RX_chn = awgn(RX_mod, SNR(i))
        snr_linear = 10**(snr_db / 10)
        
        # For BPSK in AWGN, noise variance = 1 / (2 * SNR_linear)
        # But MATLAB's awgn() uses signal power = 1, so noise_var = 1/SNR_linear
        noise_variance = 1 / snr_linear
        noise_std = np.sqrt(noise_variance)
        
        # Add noise
        noise = np.random.normal(0, noise_std, len(bpsk_symbols))
        received_symbols = bpsk_symbols + noise
        
        # Hard decision: matches MATLAB (sign(RX_chn) - 1) / -2
        # sign(x) gives {-1,+1}, then (sign(x)-1)/-2 gives {1,0}
        received_bits = ((np.sign(received_symbols) - 1) / -2).astype(int)
        
        # Compute LLRs for soft decoding
        # LLR = 2 * received_symbol / noise_variance
        # Positive LLR means bit is likely 0, negative LLR means bit is likely 1
        llrs = 2 * received_symbols / noise_variance
        
        return received_bits, llrs
    
    def decode_belief_propagation(self, llrs, max_iterations=10):
        """
        Belief Propagation (BP) decoder - the digital baseline described in the paper
        
        Args:
            llrs: Log-likelihood ratios from channel
            max_iterations: Maximum number of BP iterations
            
        Returns:
            tuple: (decoded_bits, success, iterations_used, bit_errors)
        """
        n, m = self.n, self.H.shape[0]
        
        # Initialize messages
        # Variable-to-check messages: v_to_c[check_idx][var_idx]
        v_to_c = {}
        # Check-to-variable messages: c_to_v[check_idx][var_idx]  
        c_to_v = {}
        
        # Initialize message structures based on H matrix sparsity
        for i in range(m):  # For each check node
            v_to_c[i] = {}
            c_to_v[i] = {}
            for j in range(n):  # For each variable node
                if self.H[i, j] == 1:
                    v_to_c[i][j] = llrs[j]  # Initialize with channel LLRs
                    c_to_v[i][j] = 0.0
        
        for iteration in range(max_iterations):
            # Check node update: compute messages from check nodes to variable nodes
            for i in range(m):  # For each check node
                for j in v_to_c[i].keys():  # For each variable connected to this check
                    # Compute product of tanh(x/2) for all connected variables except j
                    product = 1.0
                    for k in v_to_c[i].keys():  # All variables connected to check i
                        if k != j:
                            tanh_val = np.tanh(v_to_c[i][k] / 2.0)
                            # Clip to avoid numerical issues
                            tanh_val = np.clip(tanh_val, -0.9999, 0.9999)
                            product *= tanh_val
                    
                    # Convert back using arctanh
                    if abs(product) < 0.9999:
                        c_to_v[i][j] = 2.0 * np.arctanh(product)
                    else:
                        c_to_v[i][j] = 20.0 if product > 0 else -20.0  # Large magnitude
            
            # Variable node update: compute messages from variable nodes to check nodes
            for j in range(n):  # For each variable node
                # Find all check nodes connected to variable j
                connected_checks = [i for i in range(m) if self.H[i, j] == 1]
                
                for i in connected_checks:  # For each connected check node
                    # Sum of channel LLR plus all incoming check messages except from check i
                    message_sum = llrs[j]
                    for k in connected_checks:
                        if k != i:
                            message_sum += c_to_v[k][j]
                    v_to_c[i][j] = message_sum
            
            # Compute posterior LLRs (total belief for each variable)
            posterior_llrs = llrs.copy()
            for j in range(n):
                connected_checks = [i for i in range(m) if self.H[i, j] == 1]
                for i in connected_checks:
                    posterior_llrs[j] += c_to_v[i][j]
            
            # Hard decision: if LLR > 0, bit is 0; if LLR < 0, bit is 1
            decoded_bits = (posterior_llrs < 0).astype(int)
            
            # Check if all parity checks are satisfied
            syndrome = np.dot(self.H, decoded_bits) % 2
            if np.sum(syndrome) == 0:
                # Successful decoding - all parity checks satisfied
                return decoded_bits, True, iteration + 1, 0
        
        # Failed to converge within max iterations
        final_syndrome = np.sum(syndrome)
        return decoded_bits, False, max_iterations, final_syndrome
    
    def simulate_oscillator_decoder(self, llrs, job_id=None):
        """
        Simulate the oscillator-based decoder described in the paper
        This provides the performance characteristics of the analog AMORGOS chip
        
        Args:
            llrs: Log-likelihood ratios from channel
            job_id: Job ID for deterministic simulation
            
        Returns:
            tuple: (decoded_bits, success, time_to_solution_ns, energy_pj)
        """
        # Use deterministic simulation based on job_id
        if job_id:
            seed_str = job_id.split('-')[0]
            seed_num = sum(ord(c) for c in seed_str)
            np.random.seed(seed_num % 1000)
        
        # Simulate oscillator initialization with soft information
        # The paper shows this significantly improves performance
        initial_phases = np.arctan(llrs / 2)  # Convert LLRs to initial phases
        
        # Simulate the continuous-time oscillator dynamics
        # The paper reports 89ns mean time-to-solution at 7dB SNR
        base_tts = 89  # nanoseconds
        snr_factor = np.mean(np.abs(llrs)) / 10  # Estimate SNR from LLRs
        time_to_solution = base_tts * (1 + 0.1 * np.random.randn()) * (1 / max(snr_factor, 0.1))
        
        # Energy consumption: 5.47 pJ/bit as reported in the paper
        energy_per_bit = 5.47  # pJ/bit
        total_energy = energy_per_bit * self.k  # Energy for information bits
        
        # Simulate convergence - the paper reports 99.999% convergence rate at 7dB
        convergence_probability = 0.999 if np.mean(np.abs(llrs)) > 5 else 0.95
        converged = np.random.random() < convergence_probability
        
        if converged:
            # Use BP decoder as ground truth for the oscillator result
            decoded_bits, bp_success, _, _ = self.decode_belief_propagation(llrs, max_iterations=1)
            
            # Oscillator decoder typically performs better than BP in low SNR
            # Add small probability of correcting BP failures
            if not bp_success and np.random.random() < 0.1:
                # Simulate oscillator finding solution where BP failed
                decoded_bits = (llrs < 0).astype(int)  # Simple hard decision
                syndrome = np.dot(self.H, decoded_bits) % 2
                success = np.sum(syndrome) == 0
            else:
                success = bp_success
        else:
            # Failed convergence
            decoded_bits = (llrs < 0).astype(int)
            success = False
        
        return decoded_bits, success, time_to_solution, total_energy

# Global LDPC codec instance
ldpc_codec = LDPCCodec()

if __name__ == '__main__':
    # Initialize database
    init_db()
    
    # Set start time for uptime calculation
    app.start_time = time.time()
    
    # Debug: Print registered routes
    print("=== Registered Routes ===")
    for rule in app.url_map.iter_rules():
        print(f"  {rule.rule} -> {rule.endpoint} ({rule.methods})")
    print("========================")
    
    # Auto-detect boards
    auto_detect_boards()
    
    # Start metrics collection in background
    def metrics_collector():
        while True:
            try:
                collect_system_metrics()
                time.sleep(300)  # Collect every 5 minutes
            except Exception as e:
                logger.error(f"Error in metrics collector: {e}")
                time.sleep(60)  # Retry in 1 minute on error
    
    metrics_thread = threading.Thread(target=metrics_collector, daemon=True)
    metrics_thread.start()
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 8000)),
        debug=False  # Disable debug mode to avoid reloader issues
    ) 