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


def categorize_holdings(weights: dict) -> dict:
    """Map ticker holdings to categories and calculate aggregate exposures"""
    categories = {
        'Growth Equities': ['QQQ', 'ARKK', 'IWM', 'XLC', 'XLY'],
        'Cyclical Equities': ['XLF', 'XLI', 'XLB', 'VTV', 'IWD'],
        'Defensive Equities': ['XLU', 'XLP', 'XLV'],
        'Energy': ['XLE', 'XOP', 'FCG', 'USO'],
        'Commodities (Broad)': ['DBC', 'GCC'],
        'Commodities (Metals)': ['LIT', 'AA', 'PALL', 'REMX', 'GLD'],
        'Commodities (Ags)': ['DBA'],
        'Commodities (Uranium)': ['URA'],
        'Duration (Long)': ['TLT', 'VGLT', 'IEF'],
        'Duration (Short/TIPS)': ['TIP', 'VTIP', 'VALT'],
        'Credit (IG)': ['LQD', 'MUB'],
        'Real Assets': ['VNQ', 'PAVE'],
        'Crypto': ['IBIT', 'ETHA']
    }

    # Reverse lookup
    ticker_to_category = {}
    for cat, tickers in categories.items():
        for ticker in tickers:
            ticker_to_category[ticker] = cat

    # Aggregate by category
    category_weights = {}
    for ticker, weight in weights.items():
        cat = ticker_to_category.get(ticker, 'Other')
        category_weights[cat] = category_weights.get(cat, 0) + weight

    return category_weights


def format_holdings_table(weights: dict) -> str:
    """Format top holdings as a table string"""
    sorted_holdings = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10]
    lines = ["| Ticker | Weight |", "|--------|--------|"]
    for ticker, weight in sorted_holdings:
        lines.append(f"| {ticker} | {weight*100:.1f}% |")
    return "\n".join(lines)


def format_category_summary(weights: dict) -> str:
    """Format category exposure summary"""
    cat_weights = categorize_holdings(weights)
    sorted_cats = sorted(cat_weights.items(), key=lambda x: x[1], reverse=True)
    parts = [f"{w*100:.0f}% {cat}" for cat, w in sorted_cats if w > 0.01]
    return ", ".join(parts)


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

    # Format data for prompt
    today_str = date.today().strftime("%B %d, %Y")
    holdings_table = format_holdings_table(weights)
    category_summary = format_category_summary(weights)
    scores_str = ", ".join([f"{q}: {s:.1f}%" for q, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)])

    prompt = f"""Daily Trading Morning Note Generator with Portfolio Context

Create a concise morning note for traders based on the systematic portfolio positioning analysis below. Focus on actionable themes and how current model positioning relates to broader market dynamics.

Today's Date: {today_str}

Portfolio Context:
- Dominant Quadrant: {top_quads[0]}
- Secondary Quadrant: {top_quads[1]}
- Quadrant Momentum Scores: {scores_str}
- Net Leverage: {leverage:.2f}x

Current Portfolio Holdings (Top 10):
{holdings_table}

Category Exposure Summary: {category_summary}

Category Mapping Reference:
- Growth Equities: QQQ, ARKK, IWM, XLC, XLY
- Cyclical Equities: XLF, XLI, XLB, VTV, IWD
- Defensive Equities: XLU, XLP, XLV
- Energy: XLE, XOP, FCG, USO
- Commodities (Broad): DBC, GCC
- Commodities (Metals): LIT, AA, PALL, REMX, GLD
- Commodities (Ags): DBA
- Commodities (Uranium): URA
- Duration (Long): TLT, VGLT, IEF
- Duration (Short/TIPS): TIP, VTIP, VALT
- Credit (IG): LQD, MUB
- Real Assets: VNQ, PAVE
- Crypto: IBIT, ETHA

Quadrant Definitions:
- Q1 (Goldilocks): Growth↑, Inflation↓ – favor growth assets, duration, crypto
- Q2 (Reflation): Growth↑, Inflation↑ – favor commodities, cyclicals, real assets, energy
- Q3 (Stagflation): Growth↓, Inflation↑ – favor energy, commodities, TIPS, defensives, crypto
- Q4 (Deflation): Growth↓, Inflation↓ – favor long duration, IG credit, defensives, USD

Please generate the following sections:

**Quick Note (Marketing Summary)**
A short-form summary (100-150 words max) containing:
- Opening Theme (2-3 sentences): The single most important market narrative for the current regime
- Portfolio Snapshot (1-2 sentences): Aggregate category exposures + net leverage
- Top 3 Holdings Overview (1 bullet each): For three largest positions, provide ticker, thesis, and key catalyst/risk

**Regime Context**
- Current regime characteristics and what this dual-quad environment typically favors
- Cross-quad dynamics: tensions or synergies between dominant and secondary positioning

**Position Analysis**
For the top 5 holdings, write a brief paragraph covering:
1. The macro thesis supporting the position
2. Whether the position aligns with the current quadrant regime
3. Key catalyst or risk to monitor

**Risk Considerations**
- Catalysts that could challenge current quadrant regime
- What would trigger rotation to a different quad
- Specific risks to concentrated positions

Guidelines:
- Length: 500-700 words total
- Tone: Professional, direct, written for experienced traders; newsletter style (refer to "the portfolio" not "your portfolio")
- Focus: Prioritize what's most relevant to current positioning
- Do NOT include section headers in the output - write as flowing prose with clear paragraph breaks
- Do NOT use phrases like "I think" - be declarative
- Start directly with the Quick Note content"""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
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
