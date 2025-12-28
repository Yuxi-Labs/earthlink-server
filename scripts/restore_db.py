#!/usr/bin/env python3
"""
Database restore script - restores from a backup file.
"""
import subprocess
import os
import sys
from pathlib import Path

BACKUP_DIR = Path("/tmp/backups")

def list_backups():
    backups = sorted(BACKUP_DIR.glob("earthlink_*.sql"))
    print(f"Available backups ({len(backups)}):")
    for i, b in enumerate(backups):
        size = b.stat().st_size / (1024 * 1024)
        print(f"  [{i}] {b.name} ({size:.1f} MB)")
    return backups

def restore(backup_file: Path):
    print(f"Restoring from: {backup_file}")
    
    db_password = os.getenv("DB_PASSWORD", "earthlink")
    result = subprocess.run([
        "psql",
        "-h", "db",
        "-U", "earthlink",
        "-d", "earthlink",
        "-f", str(backup_file)
    ], env={**os.environ, "PGPASSWORD": db_password}, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"RESTORE FAILED: {result.stderr}")
        return False
    
    print("Restore complete!")
    return True

if __name__ == "__main__":
    backups = list_backups()
    
    if len(sys.argv) > 1:
        # Use provided index or filename
        arg = sys.argv[1]
        if arg.isdigit():
            backup_file = backups[int(arg)]
        else:
            backup_file = BACKUP_DIR / arg
    elif backups:
        # Use latest
        backup_file = backups[-1]
        print(f"Using latest backup: {backup_file.name}")
    else:
        print("No backups found!")
        sys.exit(1)
    
    restore(backup_file)
