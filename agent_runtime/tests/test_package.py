"""Public module boundary: no dashboard or Streamlit is needed by the core."""
from pathlib import Path
import subprocess
import sys


def test_core_imports_in_fresh_process_without_ui_modules():
    repository = Path(__file__).resolve().parents[2]
    source = '''
import sys
sys.path.insert(0, sys.argv[1])
from agent_runtime import run_turn, AgentTools, ReportService
assert callable(run_turn)
assert 'streamlit' not in sys.modules
assert 'dashboard_app' not in sys.modules
assert 'dashboard_data' not in sys.modules
assert 'agent_core' not in sys.modules
assert 'agent_reports' not in sys.modules
'''
    completed = subprocess.run([sys.executable, '-c', source, str(repository)],
                               capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stderr
