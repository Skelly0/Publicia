#!/usr/bin/env python3
"""
Watchdog script for Publicia Discord bot
This script launches the bot and automatically restarts it if it crashes.
"""

import subprocess
import time
import sys
import logging
import os
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("watchdog.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("watchdog")

# Configuration
MAX_RESTARTS = 5  # Maximum number of restarts in a time period
RESTART_PERIOD = 3600  # Time period in seconds (1 hour)
COOLDOWN_TIME = 10  # Time to wait between restarts in seconds
CRASH_LOG_DIR = "crash_logs"  # Directory to store crash logs

def ensure_crash_log_dir():
    """Ensure the crash log directory exists."""
    if not os.path.exists(CRASH_LOG_DIR):
        os.makedirs(CRASH_LOG_DIR)
        logger.info(f"Created crash log directory: {CRASH_LOG_DIR}")

def save_crash_log(log_content):
    """Save the crash log to a file."""
    ensure_crash_log_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(CRASH_LOG_DIR, f"crash_{timestamp}.log")
    
    with open(log_file, "w") as f:
        f.write(log_content)
    
    logger.info(f"Saved crash log to {log_file}")

def run_bot():
    """Run the bot as a subprocess and return its exit code and stdout/stderr."""
    logger.info("Starting bot process...")
    
    # Assuming the bot script is named PubliciaV5.py
    # Capture output to save as crash log if needed
    process = subprocess.Popen(
        [sys.executable, "PubliciaV6.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1  # Line buffered
    )
    
    # Store the last 100 lines of output for crash logs
    output_lines = []
    max_lines = 100
    
    # Read output until process exits
    for line in iter(process.stdout.readline, ''):
        print(line, end='')  # Echo to console
        
        # Store limited output history
        output_lines.append(line)
        if len(output_lines) > max_lines:
            output_lines.pop(0)
    
    # Wait for process to exit and get return code
    return_code = process.wait()
    
    # Prepare crash log if crashed
    if return_code != 0:
        crash_log = f"Bot crashed with exit code {return_code}\n"
        crash_log += f"Timestamp: {datetime.now()}\n\n"
        crash_log += "Last output lines:\n"
        crash_log += "".join(output_lines)
        save_crash_log(crash_log)
    
    logger.info(f"Bot process exited with code {return_code}")
    return return_code

def main():
    """Main watchdog function that monitors and restarts the bot."""
    restarts = 0
    restart_times = []
    
    while True:
        # Run the bot
        exit_code = run_bot()
        
        # If clean exit (code 0), don't restart
        if exit_code == 0:
            logger.info("Bot exited cleanly. Not restarting.")
            break
        
        # Track restart times
        current_time = time.time()
        restart_times.append(current_time)
        
        # Remove restart times older than RESTART_PERIOD
        restart_times = [t for t in restart_times if current_time - t < RESTART_PERIOD]
        
        # Check if we've hit the maximum number of restarts
        if len(restart_times) >= MAX_RESTARTS:
            logger.error(f"Bot restarted {len(restart_times)} times in {RESTART_PERIOD/3600:.1f} hours. Stopping watchdog.")
            break
        
        # Wait for cooldown before restarting
        logger.info(f"Waiting {COOLDOWN_TIME} seconds before restarting...")
        time.sleep(COOLDOWN_TIME)
        
        logger.info(f"Restarting bot (restart {len(restart_times)} of {MAX_RESTARTS} allowed in period)...")

if __name__ == "__main__":
    logger.info("Bot watchdog started")
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Watchdog terminated by user")
    except Exception as e:
        logger.critical(f"Watchdog encountered fatal error: {e}", exc_info=True)
    
    logger.info("Bot watchdog stopped")
