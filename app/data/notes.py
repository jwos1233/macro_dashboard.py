"""
Daily Notes Generator for Epoch Macro Dashboard

Generates AI-powered daily macro notes using Anthropic's Claude.
Notes are generated once per day on first page visit and cached.
"""

import json
import os
from pathlib import Path
from datetime import datetime, date
from typing import Optional, List, Dict

DATA_DIR = Path(__file__).parent
NOTES_FILE = DATA_DIR / "daily_notes.json"

# Anthropic client (lazy loaded)
_anthropic_client = None


def get_anthropic_client():
    """Get or create Anthropic client"""
    global _anthropic_client
    if _anthropic_client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        try:
            import anthropic
            _anthropic_client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            print("Anthropic package not installed")
            return None
    return _anthropic_client


def load_notes() -> List[Dict]:
    """Load all notes from storage"""
    if not NOTES_FILE.exists():
        return []
    try:
        with open(NOTES_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading notes: {e}")
        return []


def save_notes(notes: List[Dict]) -> bool:
    """Save notes to storage"""
    try:
        with open(NOTES_FILE, 'w') as f:
            json.dump(notes, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving notes: {e}")
        return False


def get_todays_note() -> Optional[Dict]:
    """Get today's note if it exists"""
    today = date.today().isoformat()
    notes = load_notes()
    for note in notes:
        if note.get('date') == today:
            return note
    return None


def has_todays_note() -> bool:
    """Check if today's note already exists"""
    return get_todays_note() is not None


def generate_note_content(signals: dict) -> Optional[Dict]:
    """
    Generate daily note content using Anthropic API

    Args:
        signals: Current trading signals including regime, weights, scores

    Returns:
        Dict with title and content, or None if generation fails
    """
    client = get_anthropic_client()
    if client is None:
        print("Anthropic client not available - check ANTHROPIC_API_KEY")
        return None

    # Build context from signals
    top_quads = signals.get('top_quadrants', ('Q1', 'Q2'))
    regime = signals.get('current_regime', 'Unknown')
    scores = signals.get('quadrant_scores', {})
    weights = signals.get('target_weights', {})
    leverage = signals.get('total_leverage', 1.0)

    # Get top positions
    top_positions = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]
    positions_str = ", ".join([f"{t} ({w*100:.1f}%)" for t, w in top_positions])

    # Format scores
    scores_str = ", ".join([f"{q}: {s:.1f}%" for q, s in scores.items()])

    prompt = f"""You are a macro strategist writing a daily market note for a quantitative macro rotation strategy called "Epoch Macro".

Current Market Regime: {regime}
Quadrant Momentum Scores: {scores_str}
Top 5 Positions: {positions_str}
Total Portfolio Leverage: {leverage:.2f}x

Quadrant Definitions:
- Q1 (Goldilocks): Growth accelerating, inflation falling - favor growth/tech
- Q2 (Reflation): Growth accelerating, inflation rising - favor commodities/energy
- Q3 (Stagflation): Growth slowing, inflation rising - favor real assets/gold
- Q4 (Deflation): Growth slowing, inflation falling - favor bonds/defensives

Write a concise daily macro note (2-3 paragraphs) that:
1. Summarizes the current regime and what's driving it
2. Explains the portfolio positioning and key trades
3. Highlights any notable changes or risks to monitor

Keep the tone professional and analytical. Be specific about the data driving decisions.
Do NOT use phrases like "I think" or "In my opinion" - be declarative.
Do NOT include a title - just the note content.
Keep it under 200 words."""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        content = message.content[0].text.strip()

        # Generate a title based on regime
        title = generate_title(regime, scores)

        return {
            "title": title,
            "content": content
        }

    except Exception as e:
        print(f"Error generating note: {e}")
        return None


def generate_title(regime: str, scores: dict) -> str:
    """Generate a title based on current regime"""
    # Find dominant trend
    if scores:
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_quad = sorted_scores[0][0]
        top_score = sorted_scores[0][1]

        # Determine trend direction
        if top_score > 5:
            strength = "Strong"
        elif top_score > 2:
            strength = "Moderate"
        else:
            strength = "Emerging"

        quad_names = {
            'Q1': 'Goldilocks',
            'Q2': 'Reflation',
            'Q3': 'Stagflation',
            'Q4': 'Deflation'
        }

        return f"{strength} {quad_names.get(top_quad, top_quad)} Regime Continues"

    return f"Daily Macro Update: {regime}"


def create_daily_note(signals: dict) -> Optional[Dict]:
    """
    Create and save today's daily note

    Args:
        signals: Current trading signals

    Returns:
        The created note, or None if creation failed
    """
    # Check if already exists
    existing = get_todays_note()
    if existing:
        print(f"Today's note already exists: {existing.get('title')}")
        return existing

    print("Generating today's daily note...", flush=True)

    # Generate content
    generated = generate_note_content(signals)
    if generated is None:
        print("Failed to generate note content")
        return None

    # Create note object
    today = date.today().isoformat()
    regime = signals.get('current_regime', 'Unknown')

    note = {
        "date": today,
        "title": generated["title"],
        "content": generated["content"],
        "regime": regime,
        "generated_at": datetime.now().isoformat()
    }

    # Load existing notes and prepend new one
    notes = load_notes()
    notes.insert(0, note)

    # Keep only last 30 days of notes
    notes = notes[:30]

    # Save
    if save_notes(notes):
        print(f"Daily note created: {note['title']}")
        return note

    return None


def get_recent_notes(limit: int = 10) -> List[Dict]:
    """Get most recent notes"""
    notes = load_notes()
    return notes[:limit]


def ensure_todays_note(signals: dict) -> List[Dict]:
    """
    Ensure today's note exists, creating it if needed.
    Returns list of recent notes for display.

    Args:
        signals: Current trading signals (used if note needs to be generated)

    Returns:
        List of recent notes including today's
    """
    if not has_todays_note():
        create_daily_note(signals)

    return get_recent_notes()
