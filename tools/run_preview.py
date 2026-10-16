"""Start a local review server. No files or data are uploaded by this launcher."""
from pathlib import Path
import argparse
import os
import subprocess
import sys
import toml

repo = Path(__file__).resolve().parents[1]
workspace = repo.parent
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=8503)
parser.add_argument('--enable-openai', action='store_true')
args = parser.parse_args()
env = os.environ.copy()
cache = workspace / '.cache' / 'preview-pycache'
cache.mkdir(parents=True, exist_ok=True)
env['PYTHONPYCACHEPREFIX'] = str(cache)
env['PYTHONIOENCODING'] = 'utf-8'
if args.enable_openai:
    secrets = workspace / '.streamlit' / 'secrets.toml'
    if secrets.exists():
        settings = toml.loads(secrets.read_text(encoding='utf-8-sig'))
        for name in ('OPENAI_API_KEY', 'OPENAI_MODEL'):
            if isinstance(settings.get(name), str):
                env[name] = settings[name]
else:
    env.pop('OPENAI_API_KEY', None)
subprocess.run([sys.executable, '-m', 'streamlit', 'run', str(repo/'dashboard_app.py'),
                '--server.address', '127.0.0.1', '--server.port', str(args.port),
                '--server.headless', 'true', '--browser.gatherUsageStats', 'false'],
               cwd=repo, env=env, check=True)
