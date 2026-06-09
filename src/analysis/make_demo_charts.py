#!/usr/bin/env python3
"""
Generate demo charts for the methods presentation.
Creates clean, minimal figures showing preliminary framing tension results.
"""
import pandas as pd
import numpy as np
import re
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures" / "methods"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Style — clean, academic
mpl.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

# Color palette — terracotta/sage theme
PRIMARY = '#B85042'      # terracotta
SECONDARY = '#A7BEAE'    # sage
ACCENT = '#E7E8D1'       # sand
DARK = '#36454F'         # charcoal

# ══════════════════════════════════════════════════════════════════
# Load data + lexicons
# ══════════════════════════════════════════════════════════════════
print("Loading data...")
df = pd.read_parquet(DATA_DIR / "iran_articles_full.parquet")
has_ft = df['fulltext'].notna() & (df['fulltext'] != '')
df_ft = df[has_ft].copy()

THREAT_WORDS = {
    'threat', 'threaten', 'threatens', 'threatening', 'threatened',
    'attack', 'attacks', 'strike', 'strikes', 'bomb', 'bombs', 'bombing',
    'missile', 'missiles', 'weapon', 'weapons', 'nuclear', 'enrichment',
    'uranium', 'centrifuge', 'warhead',
    'terror', 'terrorism', 'terrorist', 'terrorists',
    'war', 'warfare', 'military', 'militia', 'armed',
    'danger', 'dangerous', 'hostile', 'hostility',
    'aggression', 'aggressive', 'provocative', 'provocation',
    'destabilize', 'destabilizing', 'rogue',
    'sanctions', 'sanction', 'embargo', 'punish', 'punishment',
    'crisis', 'conflict', 'confrontation', 'escalation', 'escalate',
    'kill', 'killed', 'killing', 'death', 'deaths', 'deadly',
    'execute', 'execution', 'executed',
    'seize', 'seized', 'capture', 'captured',
    'proxy', 'proxies', 'retaliation', 'retaliate',
}
DIPLOMACY_WORDS = {
    'diplomacy', 'diplomatic', 'diplomat', 'diplomats',
    'negotiate', 'negotiation', 'negotiations', 'negotiating', 'negotiator',
    'agreement', 'accord', 'deal', 'treaty', 'pact',
    'talk', 'talks', 'dialogue', 'discussion',
    'peace', 'peaceful', 'peacemaker',
    'cooperate', 'cooperation', 'cooperative',
    'compromise', 'concession', 'concessions',
    'resolve', 'resolution',
    'engage', 'engagement', 'outreach',
    'moderate', 'moderation', 'moderates', 'reform', 'reformist', 'reformers',
    'rapprochement', 'detente', 'thaw',
}
HUMANIZING_WORDS = {
    'people', 'family', 'families', 'children', 'woman', 'women', 'girl', 'girls',
    'student', 'students', 'young', 'youth',
    'artist', 'writer', 'poet', 'filmmaker', 'musician', 'director',
    'ordinary', 'everyday', 'daily', 'life', 'lives',
    'hope', 'dream', 'love', 'joy', 'happy', 'happiness',
    'culture', 'cultural', 'art', 'arts', 'film', 'cinema', 'music', 'poetry',
    'beauty', 'beautiful', 'tradition', 'traditions',
    'freedom', 'liberty', 'rights', 'dignity',
    'protest', 'protesters', 'protester', 'demonstrators', 'movement',
    'community', 'neighborhood', 'street', 'home',
}

def tokenize(t):
    return re.findall(r'\b[a-z]+\b', str(t).lower()) if isinstance(t, str) else []

def density(text, lex):
    toks = tokenize(text)
    return sum(1 for t in toks if t in lex) / len(toks) * 1000 if toks else 0.0

print("Computing lexicon densities for each component...")
for comp in ['headline', 'abstract', 'lead_paragraph', 'fulltext']:
    df_ft[f'{comp}_threat'] = df_ft[comp].apply(lambda t: density(t, THREAT_WORDS))
    df_ft[f'{comp}_diplo'] = df_ft[comp].apply(lambda t: density(t, DIPLOMACY_WORDS))
    df_ft[f'{comp}_human'] = df_ft[comp].apply(lambda t: density(t, HUMANIZING_WORDS))

df_ft['decade'] = (df_ft['year'] // 10) * 10

# ══════════════════════════════════════════════════════════════════
# Figure 1: Component-level lexicon densities (headline vs body)
# ══════════════════════════════════════════════════════════════════
print("Figure 1: headline vs body...")
fig, ax = plt.subplots(figsize=(8, 4.5))
components = ['headline', 'abstract', 'lead_paragraph', 'fulltext']
labels = ['Headline', 'Abstract', 'Lead', 'Full text']
threat_vals = [df_ft[f'{c}_threat'].mean() for c in components]
diplo_vals = [df_ft[f'{c}_diplo'].mean() for c in components]
human_vals = [df_ft[f'{c}_human'].mean() for c in components]

x = np.arange(len(labels))
width = 0.27
ax.bar(x - width, threat_vals, width, label='Threat', color=PRIMARY)
ax.bar(x, diplo_vals, width, label='Diplomacy', color=SECONDARY)
ax.bar(x + width, human_vals, width, label='Humanizing', color=DARK)
ax.set_ylabel('Density (per 1,000 tokens)')
ax.set_title('Framing density across article components', loc='left', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(frameon=False, loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig1_components.png', dpi=180, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════
# Figure 2: Temporal trend — % headlines more threat-dense than body
# ══════════════════════════════════════════════════════════════════
print("Figure 2: temporal trend...")
df_ft['hl_body_gap'] = df_ft['headline_threat'] - df_ft['fulltext_threat']
decade_pct = df_ft.groupby('decade').apply(
    lambda g: (g['hl_body_gap'] > 0).mean() * 100
)

fig, ax = plt.subplots(figsize=(8, 4.5))
decades = decade_pct.index
pcts = decade_pct.values
ax.plot(decades, pcts, marker='o', color=PRIMARY, linewidth=2.5, markersize=10)
ax.fill_between(decades, pcts, alpha=0.15, color=PRIMARY)
for x, y in zip(decades, pcts):
    ax.annotate(f'{y:.0f}%', (x, y), textcoords='offset points',
                xytext=(0, 10), ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('Share of articles (%)')
ax.set_xlabel('Decade')
ax.set_title('Headlines more threat-dense than their own body', loc='left', fontweight='bold')
ax.set_xticks(decades)
ax.set_xticklabels([f'{int(d)}s' for d in decades])
ax.set_ylim(0, 55)
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig2_temporal.png', dpi=180, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════
# Figure 3: Genre tension — humanizing/threat ratio
# ══════════════════════════════════════════════════════════════════
print("Figure 3: genre tension...")
genre_map = {
    'News': 'News', 'An Analysis': 'News Analysis',
    'Editorial': 'Editorial', 'Op-Ed': 'Op-Ed',
    'Review': 'Review', 'Letter': 'Letter',
}
df_ft['genre'] = df_ft['type_of_material'].map(genre_map)
df_genre = df_ft[df_ft['genre'].notna()]
genre_stats = df_genre.groupby('genre').agg(
    threat=('fulltext_threat', 'mean'),
    human=('fulltext_human', 'mean'),
    n=('fulltext_threat', 'size')
).reset_index()
genre_stats['ratio'] = genre_stats['human'] / genre_stats['threat']
genre_stats = genre_stats[genre_stats['n'] >= 50].sort_values('ratio')

fig, ax = plt.subplots(figsize=(8, 4.5))
colors = [PRIMARY if r < 1 else SECONDARY for r in genre_stats['ratio']]
bars = ax.barh(genre_stats['genre'], genre_stats['ratio'], color=colors)
ax.axvline(x=1, color=DARK, linestyle='--', alpha=0.5, linewidth=1)
ax.text(1.02, -0.5, 'balanced', color=DARK, fontsize=9, alpha=0.7)
for bar, val, n in zip(bars, genre_stats['ratio'], genre_stats['n']):
    ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
            f'{val:.2f}  (n={n:,})', va='center', fontsize=10)
ax.set_xlabel('Humanizing / Threat ratio')
ax.set_title('Genre produces different Irans within one newspaper', loc='left', fontweight='bold')
ax.set_xlim(0, max(genre_stats['ratio']) * 1.25)
ax.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig3_genre.png', dpi=180, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════
# Figure 4: Visual-textual tension
# ══════════════════════════════════════════════════════════════════
print("Figure 4: visual-textual tension...")
# Imaged vs non-imaged threat density
imaged = df_ft[df_ft['has_image']]
not_imaged = df_ft[~df_ft['has_image']]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
# Panel 1: threat density
labels1 = ['With image', 'No image']
vals1 = [imaged['fulltext_threat'].mean(), not_imaged['fulltext_threat'].mean()]
bars = ax1.bar(labels1, vals1, color=[PRIMARY, SECONDARY], width=0.55)
for bar, val in zip(bars, vals1):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.3, f'{val:.1f}',
             ha='center', fontweight='bold', fontsize=12)
ax1.set_ylabel('Threat density (per 1,000 tokens)')
ax1.set_title('Imaged articles more threat-dense', loc='left', fontweight='bold', fontsize=12)
ax1.set_ylim(0, max(vals1) * 1.2)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Panel 2: diplomacy-security co-occurrence
def parse_kw(j):
    if pd.isna(j): return set()
    try:
        kws = json.loads(j) if isinstance(j, str) else j
        return {k.get('value', '') for k in kws}
    except:
        return set()

SEC_KW = {'Nuclear Weapons', 'ATOMIC WEAPONS', 'Terrorism',
    'ARMAMENT, DEFENSE AND MILITARY FORCES', 'MILITARY ACTION',
    'ARMS SALES ABROAD', 'Wars and Revolutions', 'Embargoes and Sanctions'}
DIP_KW = {'International Relations', 'United States International Relations',
    'Treaties', 'Diplomacy and Diplomats', 'United Nations',
    'Nuclear Nonproliferation Treaty'}

df_ft['kw_set'] = df_ft['keywords_json'].apply(parse_kw)
df_ft['has_dip'] = df_ft['kw_set'].apply(lambda s: bool(s & DIP_KW))
df_ft['has_sec'] = df_ft['kw_set'].apply(lambda s: bool(s & SEC_KW))

dip_only = df_ft[df_ft['has_dip']]['has_sec'].mean() * 100
overall_sec = df_ft['has_sec'].mean() * 100

vals2 = [dip_only, overall_sec]
labels2 = ['Diplomacy-\nkeyword\narticles', 'All\narticles']
bars = ax2.bar(labels2, vals2, color=[PRIMARY, SECONDARY], width=0.55)
for bar, val in zip(bars, vals2):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 1.5, f'{val:.0f}%',
             ha='center', fontweight='bold', fontsize=12)
ax2.set_ylabel('% with security keywords')
ax2.set_title('Diplomacy coverage co-framed with threat', loc='left', fontweight='bold', fontsize=12)
ax2.set_ylim(0, 75)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig4_visual.png', dpi=180, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════
# Figure 5: War/Peace journalism framework applied
# ══════════════════════════════════════════════════════════════════
print("Figure 5: war/peace framework...")
WAR_LANG = {'destitute','pathetic','tragic','helpless','desperate','suffering',
            'victim','victims','refugee','refugees','fled','displaced',
            'vicious','barbaric','terrorist','terrorists','fanatic','fanatics',
            'fundamentalist','fundamentalists','extremist','extremists','radical',
            'radicals','rogue','tyrant','dictator','evil','brutal','ruthless',
            'genocide','massacre','slaughter','atrocity','atrocities','carnage',
            'devastating','catastrophic','horrific','outrage','outrageous','fury',
            'president','minister','official','officials','government','administration',
            'commander','general','leader','supreme','ally','allies','enemy','enemies',
            'axis','regime','defeat','victory','win','lose','surrender','capitulate',
            'dominate','prevail','triumph','conquer'}
PEACE_LANG = {'people','families','children','women','civilians','ordinary','community',
              'neighborhood','daily','everyday','citizen','citizens',
              'peace','solution','resolution','reconciliation','dialogue',
              'compromise','agreement','cooperation','reform','progress',
              'because','caused','underlying','root','consequence','impact',
              'result','context','history','historical'}

df_ft['war_hl'] = df_ft['headline'].apply(lambda t: density(t, WAR_LANG))
df_ft['peace_hl'] = df_ft['headline'].apply(lambda t: density(t, PEACE_LANG))
df_ft['war_ft'] = df_ft['fulltext'].apply(lambda t: density(t, WAR_LANG))
df_ft['peace_ft'] = df_ft['fulltext'].apply(lambda t: density(t, PEACE_LANG))

decade_wp = df_ft.groupby('decade').agg(
    war=('war_ft', 'mean'),
    peace=('peace_ft', 'mean')
).reset_index()
decade_wp['ratio'] = decade_wp['war'] / decade_wp['peace']

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(decade_wp['decade'], decade_wp['ratio'], marker='o',
        color=PRIMARY, linewidth=2.5, markersize=10)
ax.fill_between(decade_wp['decade'], decade_wp['ratio'], alpha=0.15, color=PRIMARY)
for x, y in zip(decade_wp['decade'], decade_wp['ratio']):
    ax.annotate(f'{y:.2f}', (x, y), textcoords='offset points',
                xytext=(0, 10), ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('War / Peace journalism ratio')
ax.set_xlabel('Decade')
ax.set_title('War-framing dominance declining over time',
             loc='left', fontweight='bold')
ax.set_xticks(decade_wp['decade'])
ax.set_xticklabels([f'{int(d)}s' for d in decade_wp['decade']])
ax.axhline(y=1, color=DARK, linestyle='--', alpha=0.4, linewidth=1)
ax.text(decade_wp['decade'].max(), 1.05, 'balanced', color=DARK, fontsize=9, alpha=0.7)
ax.set_ylim(0, max(decade_wp['ratio']) * 1.2)
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig5_warpeace.png', dpi=180, bbox_inches='tight')
plt.close()

print(f"\nAll figures saved to: {FIG_DIR}")
print("Key stats for slides:")
print(f"  Headline threat: {df_ft['headline_threat'].mean():.1f} vs fulltext: {df_ft['fulltext_threat'].mean():.1f}")
print(f"  Ratio: {df_ft['headline_threat'].mean() / df_ft['fulltext_threat'].mean():.1f}x")
print(f"  1970s → 2020s: {decade_pct.iloc[0]:.0f}% → {decade_pct.iloc[-1]:.0f}%")
print(f"  Imaged threat: {imaged['fulltext_threat'].mean():.1f} vs non-imaged: {not_imaged['fulltext_threat'].mean():.1f}")
print(f"  Diplomacy + security overlap: {dip_only:.0f}%")
print(f"  War/peace ratio 1980s → 2020s: {decade_wp['ratio'].iloc[1]:.2f} → {decade_wp['ratio'].iloc[-1]:.2f}")
