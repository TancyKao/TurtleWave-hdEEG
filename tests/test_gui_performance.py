#!/usr/bin/env python3
"""
Test script to verify GUI database performance optimizations
"""

import sqlite3
import time
import os
import tempfile
import random

def create_test_database(db_path, num_rows=500000):
    """Create a test database with the specified number of rows"""
    print(f"Creating test database with {num_rows:,} rows...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create events table (same structure as in TurtleWave)
    cursor.execute("""
        CREATE TABLE events (
            id INTEGER PRIMARY KEY,
            event_type TEXT,
            method TEXT,
            freq_lower REAL,
            freq_upper REAL,
            stage TEXT,
            channel TEXT,
            start_time REAL,
            duration REAL
        )
    """)
    
    # Generate test data
    event_types = ['slow_wave', 'spindle']
    methods = ['Massimini2004', 'Moelle2011', 'Ferrarelli2007', 'Ngo2015']
    stages = ['NREM1', 'NREM2', 'NREM3', 'REM', 'Wake']
    channels = [f'E{i}' for i in range(1, 257)]  # 256 channels
    
    # Insert test data in batches for better performance
    batch_size = 10000
    for batch_start in range(0, num_rows, batch_size):
        batch_data = []
        for i in range(batch_start, min(batch_start + batch_size, num_rows)):
            event_type = random.choice(event_types)
            method = random.choice(methods)
            stage = random.choice(stages)
            channel = random.choice(channels)
            
            if event_type == 'slow_wave':
                freq_lower = random.uniform(0.1, 1.0)
                freq_upper = random.uniform(1.0, 4.0)
            else:  # spindle
                freq_lower = random.uniform(9.0, 12.0)
                freq_upper = random.uniform(12.0, 16.0)
            
            batch_data.append((
                event_type, method, freq_lower, freq_upper, stage, channel,
                random.uniform(0, 3600), random.uniform(0.1, 3.0)
            ))
        
        cursor.executemany("""
            INSERT INTO events (event_type, method, freq_lower, freq_upper, stage, channel, start_time, duration)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, batch_data)
        
        if batch_start % 50000 == 0:
            print(f"  Inserted {batch_start + len(batch_data):,} rows...")
    
    conn.commit()
    conn.close()
    print(f"Test database created successfully with {num_rows:,} rows")

def ensure_database_indexes(cursor):
    """Create the same indexes as in the optimized GUI code"""
    print("Creating database indexes...")
    
    # Create composite index for event_type + method + freq + stage queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_events_composite 
        ON events(event_type, method, freq_lower, freq_upper, stage)
    """)
    
    # Create index for event_type filtering
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_events_type 
        ON events(event_type)
    """)
    
    # Create index for method filtering
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_events_method 
        ON events(method)
    """)
    
    # Create index for stage filtering
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_events_stage 
        ON events(stage)
    """)
    
    # Create index for channel queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_events_channel 
        ON events(channel)
    """)
    
    # Create composite index for PAC channel queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_events_pac_channels 
        ON events(event_type, method, stage, channel)
    """)
    
    print("Database indexes created")

def test_query_performance(db_path):
    """Test the performance of the optimized queries"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("\n=== Testing Query Performance ===")
    
    # Test 1: Original separate queries (slow)
    print("\n1. Testing ORIGINAL separate queries (without indexes):")
    
    start_time = time.time()
    cursor.execute("""
        SELECT method, freq_lower, freq_upper, stage, COUNT(*) as event_count
        FROM events 
        WHERE event_type = 'slow_wave'
        GROUP BY method, freq_lower, freq_upper, stage
        ORDER BY method, freq_lower, freq_upper, stage
    """)
    sw_results = cursor.fetchall()
    
    cursor.execute("""
        SELECT method, freq_lower, freq_upper, stage, COUNT(*) as event_count
        FROM events 
        WHERE event_type = 'spindle'
        GROUP BY method, freq_lower, freq_upper, stage
        ORDER BY method, freq_lower, freq_upper, stage
    """)
    spindle_results = cursor.fetchall()
    
    original_time = time.time() - start_time
    print(f"   Original queries time: {original_time:.3f} seconds")
    print(f"   Found {len(sw_results)} slow wave methods, {len(spindle_results)} spindle methods")
    
    # Test 2: Add indexes and test optimized query
    print("\n2. Creating indexes and testing OPTIMIZED query:")
    
    start_time = time.time()
    ensure_database_indexes(cursor)
    index_time = time.time() - start_time
    print(f"   Index creation time: {index_time:.3f} seconds")
    
    # Test optimized single query
    start_time = time.time()
    cursor.execute("""
        SELECT event_type, method, freq_lower, freq_upper, stage, COUNT(*) as event_count
        FROM events 
        WHERE event_type IN ('slow_wave', 'spindle')
        GROUP BY event_type, method, freq_lower, freq_upper, stage
        ORDER BY event_type, method, freq_lower, freq_upper, stage
    """)
    all_results = cursor.fetchall()
    
    # Separate results by event type (same as in GUI)
    sw_results_opt = [(method, freq_lower, freq_upper, stage, count) 
                     for event_type, method, freq_lower, freq_upper, stage, count in all_results 
                     if event_type == 'slow_wave']
    
    spindle_results_opt = [(method, freq_lower, freq_upper, stage, count) 
                          for event_type, method, freq_lower, freq_upper, stage, count in all_results 
                          if event_type == 'spindle']
    
    optimized_time = time.time() - start_time
    print(f"   Optimized query time: {optimized_time:.3f} seconds")
    print(f"   Found {len(sw_results_opt)} slow wave methods, {len(spindle_results_opt)} spindle methods")
    
    # Test 3: Channel query performance
    print("\n3. Testing channel query performance:")
    
    start_time = time.time()
    cursor.execute("SELECT DISTINCT channel FROM events ORDER BY channel")
    channels = cursor.fetchall()
    channel_time = time.time() - start_time
    print(f"   Channel query time: {channel_time:.3f} seconds")
    print(f"   Found {len(channels)} unique channels")
    
    # Performance summary
    print(f"\n=== Performance Summary ===")
    print(f"Original queries time:  {original_time:.3f} seconds")
    print(f"Optimized query time:   {optimized_time:.3f} seconds")
    print(f"Performance improvement: {original_time/optimized_time:.1f}x faster")
    print(f"Time saved:             {original_time - optimized_time:.3f} seconds")
    
    if original_time > 1.0 and optimized_time < 0.5:
        print("✅ SIGNIFICANT performance improvement achieved!")
    elif optimized_time < original_time:
        print("✅ Performance improvement achieved")
    else:
        print("⚠️  No significant performance improvement")
    
    conn.close()

def main():
    """Main test function"""
    print("TurtleWave GUI Database Performance Test")
    print("=" * 50)
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name
    
    try:
        # Create test database with 500k rows (similar to user's issue)
        create_test_database(db_path, num_rows=500000)
        
        # Test query performance
        test_query_performance(db_path)
        
    finally:
        # Clean up
        if os.path.exists(db_path):
            os.unlink(db_path)
            print(f"\nCleaned up test database: {db_path}")

if __name__ == "__main__":
    main()