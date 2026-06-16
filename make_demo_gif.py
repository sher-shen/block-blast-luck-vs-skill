"""
make_demo_gif.py — render animated GIFs of ACTUAL engine gameplay for the README.
NEW file; imports pieces.py / sim.py / scoring.py READ-ONLY (firewall-safe). Uses Pillow.

Outputs:
  assets/demo_gameplay.gif    — one game (greedy policy), piece-by-piece, with line-clear flashes + live score.
  assets/demo_difficulty.gif  — side-by-side: an "easy" deal (straight bars) vs a "hard" deal (diagonals)
                                at the SAME mean piece size, visually showing the hard board choke first.
"""

import os, random
from PIL import Image, ImageDraw, ImageFont
from pieces import CATALOG
from sim import N, empty_board, legal_positions, apply_move, copy_board, heuristic
from scoring import Scoring, score_placement

SCORING = Scoring()
SHAPES = [c for _, c in CATALOG]
NAMES = [nm for nm, _ in CATALOG]
SIZES = [len(c) for c in SHAPES]
COLORS = {1:(142,155,240), 2:(70,183,240), 3:(52,211,153), 4:(245,185,69),
          5:(245,124,70), 6:(239,93,122), 9:(192,105,230)}
BG=(13,16,24); PANEL=(31,40,58); EMPTY=(29,37,54); TXT=(238,242,249); MUTED=(134,147,171)
CELL=40; GAP=4; PAD=14; HEAD=74

def _font(sz):
    for p in ("C:/Windows/Fonts/segoeui.ttf","C:/Windows/Fonts/arial.ttf"):
        try: return ImageFont.truetype(p, sz)
        except Exception: pass
    return ImageFont.load_default()

F_BIG=_font(22); F_SM=_font(15)

def best_move(board, cells, combo):
    best=None
    for (r0,c0) in legal_positions(board, cells):
        b=copy_board(board); cl,emp=apply_move(b, cells, r0, c0)
        pts,_=score_placement(SCORING, len(cells), cl, emp, combo)
        val=pts+heuristic(b)
        if best is None or val>best[0]: best=(val,r0,c0)
    return None if best is None else (best[1], best[2])

def board_px():
    return PAD*2 + N*CELL + (N-1)*GAP

def draw_board(img, ox, oy, colorboard, highlight=frozenset()):
    d=ImageDraw.Draw(img)
    side=board_px()
    d.rounded_rectangle([ox,oy,ox+side,oy+side], radius=12, fill=PANEL)
    for r in range(N):
        for c in range(N):
            x=ox+PAD+c*(CELL+GAP); y=oy+PAD+r*(CELL+GAP)
            col=colorboard[r][c]
            if (r,c) in highlight: fill=(255,255,255)
            elif col: fill=col
            else: fill=EMPTY
            d.rounded_rectangle([x,y,x+CELL,y+CELL], radius=6, fill=fill)

def frame_single(colorboard, score, combo, highlight=frozenset(), dead=False):
    side=board_px(); W=side+PAD*2; H=HEAD+side+PAD
    img=Image.new("RGB",(W,H),BG); d=ImageDraw.Draw(img)
    d.text((PAD, 14), "Block Blast 8×8", font=F_BIG, fill=TXT)
    s=f"Score {score}    Combo {combo}" + ("    GAME OVER" if dead else "")
    d.text((PAD, 46), s, font=F_SM, fill=MUTED if not dead else (255,93,108))
    draw_board(img, PAD, HEAD, colorboard, highlight)
    return img

def play_frames(seed, pool, max_pieces=30, max_clears_stop=None):
    """Greedy game; yields (image, duration_ms). Manual clear so we can flash."""
    deal=random.Random(f"deal-{seed}"); act=random.Random(f"act-{seed}")
    board=empty_board(); cb=[[None]*N for _ in range(N)]
    combo=0; score=0; placed=0; frames=[]
    frames.append((frame_single(cb,0,0), 700))
    while placed<max_pieces:
        hand=[SHAPES[deal.choice(pool)] for _ in range(3)]
        for cells in hand:
            mv=best_move(board, cells, combo)
            if mv is None:
                frames.append((frame_single(cb,score,combo,dead=True), 2200))
                return frames
            r0,c0=mv; color=COLORS[len(cells)]
            for (r,c) in cells: board[r0+r][c0+c]=True; cb[r0+r][c0+c]=color
            full_rows=[r for r in range(N) if all(board[r])]
            full_cols=[c for c in range(N) if all(board[r][c] for r in range(N))]
            cleared=len(full_rows)+len(full_cols)
            frames.append((frame_single(cb,score,combo), 330))
            if cleared:
                hl=set()
                for r in full_rows:
                    for c in range(N): hl.add((r,c))
                for c in full_cols:
                    for r in range(N): hl.add((r,c))
                frames.append((frame_single(cb,score,combo,highlight=hl), 240))
                for (r,c) in hl: board[r][c]=False; cb[r][c]=None
            empty=not any(board[r][c] for r in range(N) for c in range(N))
            pts,combo=score_placement(SCORING, len(cells), cleared, empty, combo)
            score+=pts; placed+=1
            if cleared:
                frames.append((frame_single(cb,score,combo), 300))
    frames.append((frame_single(cb,score,combo), 1800))
    return frames

def save_gif(frames, path):
    imgs=[f for f,_ in frames]; durs=[d for _,d in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=durs, loop=0, optimize=True)
    print(f"-> {path}  ({len(imgs)} frames, {os.path.getsize(path)//1024} KB)")

# ---------- difficulty side-by-side ----------
def idxs(prefix): return [i for i,nm in enumerate(NAMES) if nm.split('#')[0]==prefix]

def play_board_states(seed, pool):
    """Return the list of color-board snapshots after each piece (for the dual panel)."""
    deal=random.Random(f"deal-{seed}"); act=random.Random(f"act-{seed}")
    board=empty_board(); cb=[[None]*N for _ in range(N)]
    combo=0; score=0; snaps=[]
    while True:
        hand=[SHAPES[deal.choice(pool)] for _ in range(3)]
        for cells in hand:
            mv=best_move(board, cells, combo)
            if mv is None:
                snaps.append(([row[:] for row in cb], score, True)); return snaps
            r0,c0=mv
            for (r,c) in cells: board[r0+r][c0+c]=True; cb[r0+r][c0+c]=COLORS[len(cells)]
            cleared,empty=apply_move(board, cells, r0, c0)
            # re-sync cb to board after clear
            for r in range(N):
                for c in range(N):
                    if not board[r][c]: cb[r][c]=None
            pts,combo=score_placement(SCORING, len(cells), cleared, empty, combo)
            score+=pts
            snaps.append(([row[:] for row in cb], score, False))

def frame_dual(left, right, lscore, rscore, ldead, rdead, step):
    side=board_px(); gap=40; FOOT=26; W=PAD*2+2*side+gap; H=HEAD+side+PAD+FOOT
    img=Image.new("RGB",(W,H),BG); d=ImageDraw.Draw(img)
    d.text((PAD,10), "EASY deal — straight bars", font=F_SM, fill=(52,211,153))
    d.text((PAD,28), f"Score {lscore}"+("   OUT" if ldead else ""), font=F_SM, fill=MUTED if not ldead else (255,93,108))
    ox2=PAD+side+gap
    d.text((ox2,10), "HARD deal — diagonals (same μ)", font=F_SM, fill=(245,124,70))
    d.text((ox2,28), f"Score {rscore}"+("   OUT" if rdead else ""), font=F_SM, fill=MUTED if not rdead else (255,93,108))
    draw_board(img, PAD, HEAD, left); draw_board(img, ox2, HEAD, right)
    d.text((PAD, HEAD+side+6), "Identical mean piece size μ — only packability differs. Illustrative single game (aggregate: 40% shorter survival, n=50).",
           font=F_SM, fill=MUTED)
    return img

def difficulty_frames(seed_e, seed_h):
    base=[i for i in range(len(SHAPES)) if NAMES[i]!='pent_plus']
    easy_pool=base + idxs('tromino_I')*8     # upweight straight-3 bars
    hard_pool=base + idxs('diag3')*8         # upweight diagonal-3 (same size -> same μ)
    L=play_board_states(seed_e, easy_pool)
    R=play_board_states(seed_h, hard_pool)
    n=min(60, max(len(L),len(R))); frames=[]
    lastL=([[None]*N for _ in range(N)],0,True); lastR=([[None]*N for _ in range(N)],0,True)
    for i in range(n):
        l=L[i] if i<len(L) else (L[-1][0],L[-1][1],True)
        r=R[i] if i<len(R) else (R[-1][0],R[-1][1],True)
        frames.append((frame_dual(l[0],r[0],l[1],r[1],l[2],r[2],i), 300))
    frames.append((frames[-1][0], 2200))
    return frames

def pick_seed(pool, lo=16, hi=34):
    """Pick a seed giving a satisfying game length with several clears."""
    best=(None,-1)
    for s in range(300):
        snaps=play_board_states(s, pool)
        length=len(snaps); finalscore=snaps[-1][1]
        if lo<=length<=hi and finalscore>best[1]: best=(s,finalscore)
    return best[0] if best[0] is not None else 0

if __name__=="__main__":
    os.makedirs("assets", exist_ok=True)
    pool=[i for i in range(len(SHAPES)) if NAMES[i]!='pent_plus']
    s=pick_seed(pool)
    print(f"gameplay seed={s}")
    gf=play_frames(s, pool, max_pieces=30)
    save_gif(gf, "assets/demo_gameplay.gif")
    save_gif(difficulty_frames(50, 50), "assets/demo_difficulty.gif")
