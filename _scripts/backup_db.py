#!/usr/bin/env python3
"""
Database backup script - creates timestamped pg_dump backups.
Run this BEFORE any risky operation.
"""
import subprocess
import os
from datetime import datetime
from pathlib import Path

BACKUP_DIR = Path("/tmp/backups")
BACKUP_DIR.mkdir(exist_ok=True)

def backup():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = BACKUP_DIR / f"earthlink_{timestamp}.sql"
    
    print(f"Creating backup: {backup_file}")
    
    # Run pg_dump from the db container
    result = subprocess.run([
        "docker", "exec", "earthlink-db",
        "pg_dump", "-U", "earthlink", "-d", "earthlink"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"BACKUP FAILED: {result.stderr}")
        return None
    
    # Write output to file
    backup_file.write_text(result.stdout)
    
    # Get file size
    size_mb = backup_file.stat().st_size / (1024 * 1024)
    print(f"Backup complete: {backup_file} ({size_mb:.1f} MB)")
    
    # List all backups
    backups = sorted(BACKUP_DIR.glob("earthlink_*.sql"))
    print(f"\nAvailable backups ({len(backups)}):")
    for b in backups[-5:]:  # Show last 5
        size = b.stat().st_size / (1024 * 1024)
        print(f"  {b.name} ({size:.1f} MB)")
    
    return backup_file

if __name__ == "__main__":
    backup()
