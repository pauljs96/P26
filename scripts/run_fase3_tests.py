#!/usr/bin/env python3
"""
FASE 3: Complete Testing & Deployment Validation
Sistema Tesis Multi-Tenant
"""

import subprocess
import sys
import os
import json
from datetime import datetime
from pathlib import Path

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header():
    """Print test suite header"""
    print(f"""
    {Colors.BOLD}{Colors.OKBLUE}═══════════════════════════════════════════════════════{Colors.ENDC}
    {Colors.BOLD}{Colors.OKBLUE}   FASE 3: TESTING & DEPLOYMENT VALIDATION            {Colors.ENDC}
    {Colors.BOLD}{Colors.OKBLUE}   Sistema Tesis Multi-Tenant                         {Colors.ENDC}
    {Colors.BOLD}{Colors.OKBLUE}   March 15, 2026                                     {Colors.ENDC}
    {Colors.BOLD}{Colors.OKBLUE}═══════════════════════════════════════════════════════{Colors.ENDC}
    """)


def run_command(cmd, description):
    """Run a shell command and report results"""
    print(f"\n{Colors.OKCYAN}▶️  {description}{Colors.ENDC}")
    print(f"   Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=False)
    
    if result.returncode == 0:
        print(f"{Colors.OKGREEN}✅ {description} - PASSED{Colors.ENDC}\n")
        return True
    else:
        print(f"{Colors.FAIL}❌ {description} - FAILED{Colors.ENDC}\n")
        return False


def run_unit_tests():
    """Run unit tests"""
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKCYAN}1. UNIT TESTS{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKCYAN}{'='*60}{Colors.ENDC}")
    
    success = run_command(
        "pytest tests/test_services.py -v -m 'not slow'",
        "Running unit tests for core services"
    )
    
    return success


def run_dashboard_tests():
    """Run dashboard integration tests"""
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKCYAN}2. DASHBOARD TESTS{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKCYAN}{'='*60}{Colors.ENDC}")
    
    success = run_command(
        "pytest tests/test_dashboard.py -v",
        "Running dashboard integration tests"
    )
    
    return success


def run_load_tests():
    """Run load tests"""
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKCYAN}3. LOAD TESTS{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKCYAN}{'='*60}{Colors.ENDC}")
    
    success = run_command(
        "python tests/test_load.py",
        "Running load tests (10, 50, 100 concurrent users)"
    )
    
    return success


def run_security_tests():
    """Run security tests"""
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKCYAN}4. SECURITY TESTS{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKCYAN}{'='*60}{Colors.ENDC}")
    
    success = run_command(
        "pytest tests/test_services.py::TestRBACMiddleware -v",
        "Running RBAC & security tests"
    )
    
    return success


def run_performance_tests():
    """Run performance benchmarks"""
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKCYAN}5. PERFORMANCE BENCHMARKS{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKCYAN}{'='*60}{Colors.ENDC}")
    
    success = run_command(
        "pytest tests/test_services.py::TestPerformance -v",
        "Running performance benchmarks"
    )
    
    return success


def run_all_tests():
    """Run all tests"""
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKCYAN}6. COMPREHENSIVE TEST SUITE{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKCYAN}{'='*60}{Colors.ENDC}")
    
    success = run_command(
        "pytest tests/ -v --tb=short",
        "Running complete test suite"
    )
    
    return success


def check_requirements():
    """Verify all dependencies are installed"""
    print(f"\n{Colors.OKCYAN}▶️  Checking dependencies...{Colors.ENDC}")
    
    required_packages = [
        'pytest',
        'pandas',
        'duckdb',
        'duckdb-stubs',
        'polars',
        'pyarrow',
        'boto3',
        'supabase',
        'streamlit',
        'psutil',
        'requests'
    ]
    
    missing = []
    for package in required_packages:
        result = subprocess.run(
            f"pip show {package}",
            shell=True,
            capture_output=True
        )
        if result.returncode != 0:
            missing.append(package)
    
    if missing:
        print(f"{Colors.WARNING}⚠️  Missing packages: {', '.join(missing)}{Colors.ENDC}")
        print(f"{Colors.OKCYAN}▶️  Installing missing packages...{Colors.ENDC}\n")
        subprocess.run(f"pip install {' '.join(missing)}", shell=True)
        print()
    
    print(f"{Colors.OKGREEN}✅ All dependencies verified{Colors.ENDC}\n")


def generate_report(results):
    """Generate test report"""
    print(f"\n{Colors.BOLD}{Colors.OKBLUE}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKBLUE}TEST RESULTS SUMMARY{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKBLUE}{'='*60}{Colors.ENDC}\n")
    
    test_names = [
        "Unit Tests",
        "Dashboard Tests",
        "Load Tests",
        "Security Tests",
        "Performance Tests",
        "Complete Suite"
    ]
    
    passed = sum(1 for r in results if r)
    total = len(results)
    
    for name, result in zip(test_names, results):
        status = f"{Colors.OKGREEN}✅ PASSED{Colors.ENDC}" if result else f"{Colors.FAIL}❌ FAILED{Colors.ENDC}"
        print(f"  {name:<30} {status}")
    
    print(f"\n{Colors.BOLD}Overall: {passed}/{total} test groups passed{Colors.ENDC}")
    
    if passed == total:
        print(f"{Colors.OKGREEN}{Colors.BOLD}✅ READY FOR DEPLOYMENT{Colors.ENDC}\n")
        return True
    else:
        print(f"{Colors.FAIL}{Colors.BOLD}❌ SOME TESTS FAILED{Colors.ENDC}\n")
        return False


def main():
    """Main test orchestrator"""
    print_header()
    
    # Check environment
    check_requirements()
    
    # Store CWD
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run test suites
    results = []
    
    try:
        results.append(run_unit_tests())
        results.append(run_dashboard_tests())
        results.append(run_load_tests())
        results.append(run_security_tests())
        results.append(run_performance_tests())
        results.append(run_all_tests())
        
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}⚠️  Tests interrupted by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.FAIL}❌ Error during testing: {e}{Colors.ENDC}")
        sys.exit(1)
    
    # Generate report
    all_passed = generate_report(results)
    
    # Print deployment recommendations
    print(f"{Colors.BOLD}{Colors.OKCYAN}DEPLOYMENT INSTRUCTIONS:{Colors.ENDC}")
    print(f"""
    1. {Colors.OKCYAN}Setup Streamlit Cloud:{Colors.ENDC}
       git remote add streamlit https://github.com/pauljs96/P26.git
       
    2. {Colors.OKCYAN}Create streamlit/secrets.toml in repo:{Colors.ENDC}
       SUPABASE_URL = "your-supabase-url"
       SUPABASE_KEY = "your-supabase-key"
       AWS_ACCESS_KEY_ID = "your-aws-key"
       AWS_SECRET_ACCESS_KEY = "your-aws-secret"
       
    3. {Colors.OKCYAN}Deploy to Streamlit Cloud:{Colors.ENDC}
       https://share.streamlit.io → Deploy from GitHub
       
    4. {Colors.OKCYAN}Test production deployment:{Colors.ENDC}
       Visit: https://share.streamlit.io/pauljs96/P26/main/main.py
       Login with demo credentials (see PHASE2_README.md)
    """)
    
    # Cleanup
    print(f"\n{Colors.BOLD}{Colors.OKBLUE}{'='*60}{Colors.ENDC}")
    print(f"{Colors.OKGREEN}{Colors.BOLD}✅ FASE 3 TESTING COMPLETE{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKBLUE}{'='*60}{Colors.ENDC}\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
