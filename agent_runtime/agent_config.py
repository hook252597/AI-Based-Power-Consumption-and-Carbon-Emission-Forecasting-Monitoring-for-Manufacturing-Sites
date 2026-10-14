"""Portable package settings; importing never reads or logs credentials."""
from pathlib import Path
from dataclasses import dataclass, field
import os
import toml

APP = Path(__file__).resolve().parent
REPO = APP.parent
TODAY = '2025-05-31'


def artifact_root():
    configured = os.environ.get('MARGARET_AI_ROOT')
    if configured:
        return Path(configured).expanduser().resolve()
    for candidate in (REPO / 'ml', REPO, REPO.parent):
        if (candidate / 'outputs/agent_data/manifest.json').is_file():
            return candidate
    return REPO / 'ml'


PROJECT = artifact_root()
DEFAULT_THRESHOLDS = {'phase_voltage': 210.0, 'line_voltage': 360.0, 'power_factor': 85.0}


@dataclass
class Settings:
    api_key: str = field(default='', repr=False)
    model: str = 'gpt-5-mini'
    note: str = ''


def load_settings(project=None):
    # Explicit configuration reads only that directory; it never falls back to
    # another project's secrets. Later default files override earlier files.
    paths = ([Path(project)] if project is not None else [PROJECT, REPO, APP])
    paths = list(dict.fromkeys(paths))
    values = {}
    note = ''
    try:
        for folder in paths:
            path = folder / '.streamlit/secrets.toml'
            if path.is_file():
                values.update(toml.loads(path.read_text(encoding='utf-8-sig')))
    except (OSError, ValueError):
        note = 'API 설정 파일 형식을 확인해 주세요. 키 값은 따옴표 안에 입력합니다.'
    return Settings(api_key=str(os.environ.get('OPENAI_API_KEY') or values.get('OPENAI_API_KEY', '')).strip(),
                    model=str(os.environ.get('OPENAI_MODEL') or values.get('OPENAI_MODEL', 'gpt-5-mini')).strip(),
                    note=note)
