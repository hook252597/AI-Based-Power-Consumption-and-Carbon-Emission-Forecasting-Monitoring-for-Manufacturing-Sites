"""Shared, read-only monitoring calculations for dashboard and report agent."""
from .alert_service import get_alert_summary
__all__ = ["get_alert_summary"]
