from typing import Dict, Any
from graph_abstract.trace_parser import Invocation, Span

class TraceVisualizer:
    def render_timeline_html(self, invocation: Invocation) -> str:
        total_sec = (invocation.end_time - invocation.start_time).total_seconds() if invocation.end_time and invocation.start_time else 0.0
        if total_sec <= 0.0:
            total_sec = 0.001
            
        rows_html = []
        for span in invocation.spans:
            left_pct = 0.0
            width_pct = 100.0
            
            if span.start_time and invocation.start_time:
                left_sec = (span.start_time - invocation.start_time).total_seconds()
                left_pct = min(100.0, max(0.0, (left_sec / total_sec) * 100.0))
                
            if span.start_time and span.end_time:
                span_sec = (span.end_time - span.start_time).total_seconds()
                width_pct = min(100.0 - left_pct, max(0.0, (span_sec / total_sec) * 100.0))
            elif span.start_time:
                width_pct = 0.5
                
            bar_color = "linear-gradient(90deg, #6366F1 0%, #4F46E5 100%)"
            bar_border = "#4338CA"
            badge_bg = "#EEF2F6"
            badge_fg = "#475569"
            
            if span.status == "error":
                bar_color = "linear-gradient(90deg, #EF4444 0%, #DC2626 100%)"
                bar_border = "#B91C1C"
                badge_bg = "#FEE2E2"
                badge_fg = "#991B1B"
            elif span.type == "llm":
                bar_color = "linear-gradient(90deg, #14B8A6 0%, #0D9488 100%)"
                bar_border = "#0F766E"
                badge_bg = "#CCFBF1"
                badge_fg = "#115E59"
            elif span.type == "tool":
                bar_color = "linear-gradient(90deg, #F59E0B 0%, #D97706 100%)"
                bar_border = "#B45309"
                badge_bg = "#FEF3C7"
                badge_fg = "#92400E"
            elif span.type == "chain":
                bar_color = "linear-gradient(90deg, #6366F1 0%, #4F46E5 100%)"
                bar_border = "#4338CA"
                badge_bg = "#E0E7FF"
                badge_fg = "#3730A3"
                
            depth_pad = span.depth * 16
            name_color = "#EF4444" if span.status == "error" else "#1E293B"
            status_indicator = " 🔴" if span.status == "error" else ""
            
            row = f"""
            <div style="display: flex; align-items: center; border-bottom: 1px solid #F1F5F9; min-height: 38px; padding: 4px 0;">
                <div style="flex: 0 0 35%; display: flex; align-items: center; gap: 8px; padding-left: {depth_pad}px; overflow: hidden; white-space: nowrap; text-overflow: ellipsis;">
                    <span style="font-size: 13px; font-weight: 600; color: {name_color};">{span.name}{status_indicator}</span>
                    <span style="font-size: 10px; font-weight: 700; color: {badge_fg}; background-color: {badge_bg}; padding: 2px 6px; border-radius: 4px; text-transform: uppercase;">{span.type}</span>
                </div>
                <div style="flex: 0 0 65%; display: flex; align-items: center; height: 100%; background-image: linear-gradient(to right, #E2E8F0 1px, transparent 1px); background-size: 25% 100%; padding: 0 8px; min-height: 20px;">
                    <div style="flex: 0 0 {left_pct}%;"></div>
                    <div style="flex: 0 0 {width_pct}%; min-width: 6px; background: {bar_color}; border: 1px solid {bar_border}; height: 12px; border-radius: 6px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);" title="Duration: {span.duration_ms:.2f}ms"></div>
                </div>
            </div>
            """
            rows_html.append(row)
            
        joined_rows = "\n".join(rows_html)
        
        return f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; font-size: 13px; color: #1E293B; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); margin-bottom: 20px;">
            <div style="display: flex; border-bottom: 1px solid #E2E8F0; padding-bottom: 8px; margin-bottom: 8px; font-weight: 700; color: #475569;">
                <div style="flex: 0 0 35%;">Span / Component</div>
                <div style="flex: 0 0 65%; text-align: right; font-size: 11px; color: #94A3B8; padding-right: 8px;">Timeline ({invocation.duration_ms:.2f}ms)</div>
            </div>
            {joined_rows}
        </div>
        """
