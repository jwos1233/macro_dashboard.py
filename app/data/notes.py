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
        'Growth Equities': ['QQQ', 'ARKK', 'IWM', 'XLC', 'XLY', 'ARKX', 'BOTZ', 'EEM'],
        'Cyclical Equities': ['XLF', 'XLI', 'XLB', 'VTV', 'IWD', 'AA'],
        'Defensive Equities': ['XLU', 'XLP', 'XLV'],
        'Energy': ['XLE', 'XOP', 'FCG', 'USO'],
        'Commodities (Broad)': ['DBC', 'GCC'],
        'Commodities (Metals)': ['LIT', 'PALL', 'REMX', 'GLD'],
        'Commodities (Ags)': ['DBA'],
        'Commodities (Uranium)': ['URA'],
        'Duration (Long)': ['TLT', 'VGLT', 'IEF'],
        'Duration (Short/TIPS)': ['TIP', 'VTIP', 'VALT'],
        'Credit (IG)': ['LQD', 'MUB'],
        'Real Assets': ['VNQ', 'PAVE'],
        'Crypto': ['BTC-USD']
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
    scores_raw = signals.get('quadrant_scores', {})
    weights_raw = signals.get('target_weights', {})
    leverage = signals.get('total_leverage', 1.0)

    # Convert pandas Series to dict if needed
    scores = dict(scores_raw) if hasattr(scores_raw, 'items') else scores_raw
    weights = dict(weights_raw) if hasattr(weights_raw, 'items') else weights_raw

    # Format data for prompt
    today_str = date.today().strftime("%B %d, %Y")
    holdings_table = format_holdings_table(weights)
    category_summary = format_category_summary(weights)
    scores_str = ", ".join([f"{q}: {s:.1f}%" for q, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)])

    prompt = f"""Please create a concise morning note for traders by researching the latest publications from major sell-side firms and funds, enhanced with systematic portfolio positioning analysis. Focus on actionable themes and how current model positioning relates to consensus views.

Today's Date: {today_str}

Research Sources to Check:
Priority Sources:
- ING FX Daily (currency analysis and cross-asset themes)
- Danske Bank Daily (Nordic perspective and broader market themes)
- Goldman Sachs morning research/Market Know-How
- J.P. Morgan Weekly Market Recap and daily insights
- Bank of America Global Research weekly
- BlackRock Investment Institute weekly commentary
- Charles Schwab Market Open Update and Options Market Update
- Edward Jones Daily Market Recap
- Morningstar market analysis

Additional Sources as Available:
- Morgan Stanley morning notes
- Citi daily research
- UBS morning commentary
- Barclays daily research
- Deutsche Bank morning notes

Portfolio Context:
- Dominant Quadrant: {top_quads[0]}
- Secondary Quadrant: {top_quads[1]}
- Quadrant Momentum Scores: {scores_str}
- Net Leverage: {leverage:.2f}x

Current Portfolio Holdings (Top 10):
{holdings_table}

Category Exposure Summary: {category_summary}

Category Mapping Reference:
- Growth Equities: QQQ, ARKK, IWM, XLC, XLY, ARKX, BOTZ, EEM
- Cyclical Equities: XLF, XLI, XLB, VTV, IWD
- Defensive Equities: XLU, XLP, XLV
- Energy: XLE, XOP, FCG, USO
- Commodities (Broad): DBC, GCC
- Commodities (Metals): LIT, PALL, REMX, GLD
- Commodities (Ags): DBA
- Commodities (Uranium): URA
- Duration (Long): TLT, VGLT, IEF
- Duration (Short/TIPS): TIP, VTIP, VALT
- Credit (IG): LQD, MUB
- Real Assets: VNQ, PAVE
- Crypto: BTC-USD

Quadrant Definitions:
- Q1 (Goldilocks): Growth↑, Inflation↓ – favor growth assets, duration, crypto
- Q2 (Reflation): Growth↑, Inflation↑ – favor commodities, cyclicals, real assets, energy
- Q3 (Stagflation): Growth↓, Inflation↑ – favor energy, commodities, TIPS, defensives
- Q4 (Deflation): Growth↓, Inflation↓ – favor long duration, IG credit, defensives, USD

Content Structure:

**Quick Note (1-Minute Marketing Summary)**
A short-form summary for marketing purposes containing:
- Opening Theme (2-3 sentences): The single most important market narrative driving cross-asset moves right now
- Portfolio Snapshot (1-2 sentences): Aggregate category exposures as percentages + net leverage figure
- Top 3 Holdings Overview (1 bullet each): For the three largest positions, provide ticker, consensus view, and key catalyst/risk in one concise sentence each
Format: Keep to ~100-150 words maximum. Use bullet points for the top 3 holdings. This should be scannable and punchy.

**Opening Theme (1-2 sentences)**
Identify the single most important market narrative driving cross-asset moves right now. Frame with today's date.

**Quadrant Regime Context**
- Current Regime: State the dominant and secondary quadrants
- Regime Characteristics: Brief description of what this dual-quad environment typically favors
- Cross-Quad Dynamics: Note any tensions or synergies between dominant and secondary quad positioning

**Key Market Drivers (3-4 bullet points)**
- Focus on themes affecting multiple asset classes
- Include any significant events from the past week that still matter
- Highlight what's driving correlations/divergences between assets
- Connect drivers to quadrant regime where relevant

**Asset Class Outlook (thematic, not siloed)**
Weave together insights on:
- Rates/Fed Policy – What's priced in vs reality, curve positioning ideas
- USD – Key drivers, FX crosses with best risk/reward
- Equities – Sector rotation themes, growth vs value vs cyclicals vs defensives
- Commodities – Energy, metals, ags dynamics; real rates impact
- Credit – IG vs HY, spread dynamics
- Crypto – Regime context, correlation dynamics

**Portfolio Positioning vs. Consensus**
Analyze the portfolio against sell-side themes:

Current Portfolio Tilt Summary:
- Aggregate the top 10 holdings into category exposures
- Present as a clean table showing Category, Weight, and Holdings
- State the portfolio's net leverage

Position-Level Analysis:
For each holding (or grouped by theme where appropriate), write a short paragraph that covers: (1) the consensus view from sell-side research, (2) whether the portfolio is aligned, overweight, underweight, or contrarian relative to consensus, and (3) the key catalyst or risk for the position. Do not repeat the weighting percentage as this has already been stated in the summary table.
Use bracketed citation numbers [1], [2], [3] etc. to reference sources within the text.

**Aligned Sell-Side Trade Ideas (2-3 specific actionable suggestions)**
Review all researched publications and summarise the specific trade ideas pitched by sell-side analysts, selecting only those that align with the current quadrant framework. For each idea, write a short paragraph that covers: the specific trade, the sell-side rationale (noting if multiple publications recommend the same trade), how it fits the current quadrant regime, entry/target/stop levels where provided, and how it would impact existing portfolio tilts.
Use bracketed citation numbers [1], [2], [3] etc. to reference sources within the text.
If no suitable sell-side ideas fit the quadrant framework, explicitly state this and explain why available recommendations don't align with current regime positioning.

**Emerging Sell-Side Trade Ideas (2-3 specific actionable suggestions)**
Review ALL researched publications and summarise the most interesting and actionable trade ideas pitched by sell-side analysts that were not touched on above. The scope should be broader—include any compelling trades flagged in research, whether or not they align with current portfolio holdings or the quadrant framework.

**Risk Management Considerations**
- Catalysts that could challenge current quadrant regime
- What would trigger a rotation from dominant to secondary quad (or new quad entirely)
- Specific risks to concentrated positions in the portfolio

**Week Ahead Catalysts**
Key events/data that could shift the narrative or trigger regime changes, with specific relevance to current holdings noted (table format preferred)

**Sources**
List all referenced sources with their corresponding citation numbers.

Guidelines:
- Length: Keep total note to 600-800 words maximum (excluding sources section)
- Tone: Professional but direct, written for experienced traders; newsletter style (refer to "the portfolio" not "your portfolio")
- Objectivity: Distinguish between what the portfolio is doing vs. what consensus thinks
- Focus: Prioritize what's most relevant today – not every asset class needs coverage every day
- Actionability: Every insight should suggest either a trade idea or risk management consideration
- Synthesis: Connect dots across publications and portfolio positioning to identify emerging themes or risks
- Timing: Consider both London and NY session implications
- Honesty on Divergences: When the portfolio diverges from consensus, present both cases fairly – don't assume the portfolio is right
- Leverage Clarity: When presenting category weights that sum above 100%, reference the net leverage figure in the Portfolio Tilt Summary
- Paragraph Format: Use flowing paragraphs (not bullet points) for Position-Level Analysis and Sell-Side Trade Ideas sections
- Citations: Use bracketed numbers [1], [2], [3] in-text and compile all sources in the Sources section at the end

FORMATTING:
- Use markdown formatting for structure: **bold** for emphasis, ## for section headers
- CRITICAL: Each ## header MUST be on its own line with a blank line before it and content starting on the NEXT line
- Example of CORRECT header format:

## Quick Note

Opening Theme: Markets enter...

- Example of WRONG format: ## Quick Note **Opening Theme:** (don't put content on same line as header)
- Use bullet points (-) for lists where appropriate
- Use tables where specified (Week Ahead Catalysts, Portfolio Tilt Summary)

Create the morning note now based on today's date, current market conditions, latest available research, and the portfolio context provided."""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            messages=[
                {"role": "user", "content": prompt}
            ],
            tools=[{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 10  # Allow up to 10 searches for comprehensive research
            }]
        )

        # Extract text content from response - handle multi-block responses from web search
        # The response may contain tool_use, tool_result, and text blocks
        # We want the final/main text block with the actual note content
        content = None
        for block in message.content:
            if hasattr(block, 'text'):
                # Get the longest text block (the actual note, not preamble)
                if content is None or len(block.text) > len(content):
                    content = block.text.strip()

        if not content:
            print("No text content found in response")
            return None

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

    # Keep all notes (no limit - save in perpetuity)
    # notes = notes[:30]  # Removed limit

    # Save
    if save_notes(notes):
        print(f"Daily note created: {note['title']}")
        return note

    return None


def get_recent_notes(limit: int = 10) -> List[Dict]:
    """Get most recent notes"""
    notes = load_notes()
    return notes[:limit]


def ensure_todays_note(signals: dict, display_limit: int = 4) -> List[Dict]:
    """
    Ensure today's note exists, creating it if needed.
    Returns list of recent notes for display.

    Args:
        signals: Current trading signals (used if note needs to be generated)
        display_limit: Number of notes to return for display (default 4)

    Returns:
        List of recent notes including today's
    """
    if not has_todays_note():
        create_daily_note(signals)

    return get_recent_notes(limit=display_limit)
