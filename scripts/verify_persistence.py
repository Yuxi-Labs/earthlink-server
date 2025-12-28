#!/usr/bin/env python3
"""
Verify database persistence WITHOUT destroying data.
Tests that data survives container restart (NOT docker compose down).
"""
import subprocess
import os
import sys
import time

def run_sql(query: str) -> str:
    """Run SQL and return output."""
    db_password = os.getenv("DB_PASSWORD", "earthlink")
    result = subprocess.run(
        ["psql", "-h", "db", "-U", "earthlink", "-d", "earthlink", "-t", "-c", query],
        env={**os.environ, "PGPASSWORD": db_password},
        capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else f"ERROR: {result.stderr}"

def get_counts() -> dict:
    """Get all table counts."""
    tables = ["buildings", "roads", "places", "pois", "water_features", "land_cover", "boundaries"]
    counts = {}
    for table in tables:
        result = run_sql(f"SELECT COUNT(*) FROM {table}")
        try:
            counts[table] = int(result)
        except:
            counts[table] = 0
    return counts

def main():
    print("=" * 60)
    print("PERSISTENCE VERIFICATION (SAFE - restart only)")
    print("=" * 60)
    
    print("\n[1/3] Getting current counts...")
    before = get_counts()
    before_total = sum(before.values())
    
    if before_total == 0:
        print("❌ No data to verify! Load data first.")
        sys.exit(1)
    
    print(f"  Total before: {before_total:,}")
    for table, count in before.items():
        print(f"    {table}: {count:,}")
    
    print("\n[2/3] Restarting database container...")
    # This is SAFE - restart doesn't destroy volumes
    subprocess.run(["docker", "restart", "earthlink-db"], capture_output=True)
    
    print("  Waiting for database to be ready...")
    for i in range(30):
        time.sleep(1)
        result = subprocess.run(
            ["docker", "exec", "earthlink-db", "pg_isready", "-U", "earthlink"],
            capture_output=True
        )
        if result.returncode == 0:
            print(f"  Database ready after {i+1}s")
            break
    else:
        print("❌ Database didn't come back up!")
        sys.exit(1)
    
    time.sleep(2)  # Extra buffer
    
    print("\n[3/3] Verifying counts after restart...")
    after = get_counts()
    after_total = sum(after.values())
    
    print(f"  Total after: {after_total:,}")
    
    # Compare
    all_match = True
    for table in before:
        if before[table] != after[table]:
            print(f"  ❌ {table}: {before[table]:,} → {after[table]:,} (MISMATCH!)")
            all_match = False
        else:
            print(f"  ✓ {table}: {after[table]:,}")
    
    print()
    if all_match and before_total == after_total:
        print("=" * 60)
        print("✅ PERSISTENCE VERIFIED: All data survived restart!")
        print("=" * 60)
    else:
        print("=" * 60)
        print("❌ PERSISTENCE FAILED: Data was lost!")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()
