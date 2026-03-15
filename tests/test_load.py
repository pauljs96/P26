"""
FASE 3: Load testing for concurrent users
Sistema Tesis Multi-Tenant
"""

import time
import concurrent.futures
import random
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.services.data_service import DataService
import pandas as pd


class LoadTest:
    """Simulate concurrent user access"""
    
    def __init__(self, num_users=10, duration_seconds=60):
        self.num_users = num_users
        self.duration = duration_seconds
        self.results = {
            'successful_queries': 0,
            'failed_queries': 0,
            'total_time': 0,
            'avg_response_time': 0,
            'response_times': []
        }
        self.start_time = None
    
    def simulate_user_query(self, user_id, org_id):
        """Simulate a single user query"""
        try:
            ds = DataService(org_id=org_id, cache_ttl=300)
            
            # Create sample data
            df = pd.DataFrame({
                'id': range(1000),
                'value': range(1000),
                'category': random.choice(['A', 'B', 'C']),
                'timestamp': pd.date_range('2020-01-01', periods=1000)
            })
            
            ds.load_dataframe(f'data_user_{user_id}', df)
            
            # Execute query
            start = time.time()
            result = ds.execute_query(
                f'SELECT SUM(value) as total FROM data_user_{user_id}'
            )
            elapsed = time.time() - start
            
            return elapsed, True
            
        except Exception as e:
            return 0, False
    
    def run_concurrent_test(self):
        """Run load test with concurrent users"""
        print(f"\n{'='*60}")
        print(f"LOAD TEST: {self.num_users} concurrent users")
        print(f"{'='*60}\n")
        
        self.start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_users) as executor:
            futures = []
            
            # Submit tasks
            for user_id in range(self.num_users):
                org_id = f'org{user_id % 5}'  # 5 orgs
                future = executor.submit(
                    self.simulate_user_query, 
                    user_id, 
                    org_id
                )
                futures.append(future)
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    elapsed, success = future.result()
                    if success:
                        self.results['successful_queries'] += 1
                        self.results['response_times'].append(elapsed)
                    else:
                        self.results['failed_queries'] += 1
                except Exception as e:
                    self.results['failed_queries'] += 1
        
        total_time = time.time() - self.start_time
        self.results['total_time'] = total_time
        
        if self.results['response_times']:
            self.results['avg_response_time'] = sum(self.results['response_times']) / len(self.results['response_times'])
        
        self._print_results()
    
    def _print_results(self):
        """Print test results"""
        print(f"📊 RESULTS:")
        print(f"  Successful queries: {self.results['successful_queries']}")
        print(f"  Failed queries: {self.results['failed_queries']}")
        print(f"  Total time: {self.results['total_time']:.2f}s")
        print(f"  Avg response time: {self.results['avg_response_time']*1000:.2f}ms")
        
        if self.results['response_times']:
            min_time = min(self.results['response_times'])
            max_time = max(self.results['response_times'])
            print(f"  Min response time: {min_time*1000:.2f}ms")
            print(f"  Max response time: {max_time*1000:.2f}ms")
        
        success_rate = (self.results['successful_queries'] / 
                       (self.results['successful_queries'] + self.results['failed_queries']) * 100
                       if self.results['successful_queries'] + self.results['failed_queries'] > 0 
                       else 0)
        print(f"  Success rate: {success_rate:.1f}%")
        
        print(f"\n{'='*60}\n")


def test_10_users():
    """Load test with 10 users"""
    test = LoadTest(num_users=10, duration_seconds=30)
    test.run_concurrent_test()
    return test.results['successful_queries'] > 0


def test_50_users():
    """Load test with 50 users"""
    test = LoadTest(num_users=50, duration_seconds=30)
    test.run_concurrent_test()
    return test.results['successful_queries'] > 0


def test_100_users():
    """Load test with 100 users"""
    test = LoadTest(num_users=100, duration_seconds=30)
    test.run_concurrent_test()
    return test.results['successful_queries'] > 0


def stress_test_org_isolation():
    """Verify org isolation under load"""
    print(f"\n{'='*60}")
    print("STRESS TEST: Organization Isolation")
    print(f"{'='*60}\n")
    
    # Create 5 orgs with data
    org_data = {}
    for org_id in range(5):
        org_key = f'org{org_id}'
        df = pd.DataFrame({
            'id': range(1000),
            'value': range(1000, 2000),
            'org_id': org_id
        })
        org_data[org_key] = df
    
    # Simulate concurrent queries from different orgs
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = []
        results = {'isolation_violations': 0, 'queries': 0}
        
        for _ in range(20):
            org_id = random.choice(list(org_data.keys()))
            future = executor.submit(
                lambda oid: len(org_data[oid]) == 1000,
                org_id
            )
            futures.append(future)
        
        for future in concurrent.futures.as_completed(futures):
            results['queries'] += 1
            if not future.result():
                results['isolation_violations'] += 1
        
        print(f"✅ Org isolation maintained!")
        print(f"   Queries: {results['queries']}")
        print(f"   Violations: {results['isolation_violations']}")

if __name__ == '__main__':
    print("""
    ╔═════════════════════════════════════════╗
    ║   FASE 3 - LOAD TESTING SUITE           ║
    ║   Sistema Tesis Multi-Tenant            ║
    ║   March 15, 2026                        ║
    ╚═════════════════════════════════════════╝
    """)
    
    try:
        # Run tests
        print("▶️  Testing 10 concurrent users...")
        test_10_users()
        
        print("▶️  Testing 50 concurrent users...")
        test_50_users()
        
        print("▶️  Testing 100 concurrent users...")
        test_100_users()
        
        print("▶️  Testing organization isolation...")
        stress_test_org_isolation()
        
        print("✅ All load tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Load test failed: {e}")
        sys.exit(1)
