"""Call these functions from the team's existing app; importing starts nothing."""
from __future__ import annotations

from agent_runtime import AgentTools, run_turn


def ask_agent(question, data, reports, state, *, api_key=None, model="gpt-5-mini"):
    """Framework-independent turn; retain the same state mapping between calls."""
    return run_turn(question, state, AgentTools(data, reports, state),
                    api_key=api_key, model=model)


def render_existing_agent_tab(data, reports, state, *, legacy_calendar=False):
    """Optional fragment inside a host Streamlit tab, without app navigation."""
    from agent_runtime.agent_ui import render_agent_tab

    render_agent_tab(data=data, reports=reports, state=state,
                     legacy_calendar=legacy_calendar)
