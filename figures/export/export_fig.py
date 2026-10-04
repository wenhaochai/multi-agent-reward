"""Run a writing:plot figure script and export what it plots as JSON for a native (SVG) re-draw: the canvas title and
legend, and per panel the lines, bars, scatter points, fills, y ticks with their printed values, x ticks with labels,
ranges, panel name and other texts. The numbers come from the script's own matplotlib figure, never retyped.
  python3 export_fig.py <script.py> <out.json>
"""
import json
import runpy
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mc  # noqa: E402
from matplotlib.collections import PathCollection, PolyCollection, LineCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

script, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
sys.path.insert(0, str(script.parent))
import style  # noqa: E402  (the script's own style.py)

REC = {'canvas': {}, 'labels': {}, 'done': False}
_canvas, _panel_label = getattr(style, 'canvas', None), getattr(style, 'panel_label', None)


def canvas(*a, **k):
    REC['canvas'] = {key: k.get(key) for key in ('title', 'subtitle', 'legend', 'quantity', 'xlabel', 'note')}
    REC['canvas']['rows'] = a[0] if a else k.get('rows', 1)
    REC['canvas']['cols'] = a[1] if len(a) > 1 else k.get('cols', 1)
    return _canvas(*a, **k)


def panel_label(ax, text, x=None):
    REC['labels'][id(ax)] = text
    return _panel_label(ax, text, x)


def hexc(c, alpha=None):
    r = mc.to_rgba(c)
    return {'c': mc.to_hex(r[:3]), 'a': round(r[3] if alpha is None else alpha, 3)}


def num(v, nd=6):
    v = float(v)
    return None if v != v else float(f'{v:.{nd}g}')


def _legend_wrap(fn):
    def f(where, items, *a, **k):
        REC.setdefault('legends', {})[id(where)] = [(it[0], it[1], {'-': 'line', '--': 'dash', 's': 'box', '^': 'tri', 'o': 'dot'}.get(
            it[2] if len(it) > 2 else '-', 'line')) for it in items]
        REC['canvas']['legend'] = REC['canvas'].get('legend') or []
        return fn(where, items, *a, **k)
    return f


def dump(fig, stem=None, *a, **k):
    if REC['done']:
        return
    REC['done'] = True
    fig.canvas.draw()
    if not REC['canvas'].get('title') and fig._suptitle is not None:
        REC['canvas']['title'] = fig._suptitle.get_text()
    panels = []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        pos = ax.get_position()
        p = {'pos': [round(pos.x0, 4), round(pos.y0, 4), round(pos.width, 4), round(pos.height, 4)],
             'xlim': [num(v) for v in ax.get_xlim()], 'ylim': [num(v) for v in ax.get_ylim()],
             'xscale': ax.get_xscale(), 'yscale': ax.get_yscale(), 'label': REC['labels'].get(id(ax)),
             'lines': [], 'bars': [], 'points': [], 'fills': [], 'texts': [], 'ylabel': ax.get_ylabel(),
             'xlabel': ax.get_xlabel()}
        if id(ax) in REC.get('legends', {}):
            p['legend'] = [{'name': it[0], **hexc(it[1]), 'kind': it[2]} for it in REC['legends'][id(ax)]]
        if not p['label'] and ax.get_title():
            p['label'] = ax.get_title()
        yt = [t for t in getattr(ax, '_style_values', [])]
        p['yticks'] = [[num(t.get_position()[1]), t.get_text()] for t in yt] or \
            [[num(v), l.get_text()] for v, l in zip(ax.get_yticks(), ax.get_yticklabels())]
        p['xticks'] = [[num(v), l.get_text()] for v, l in zip(ax.get_xticks(), ax.get_xticklabels())
                       if ax.get_xlim()[0] - 1e-9 <= v <= ax.get_xlim()[1] + 1e-9]
        p['xgrid'] = any(g.get_visible() for g in ax.get_xgridlines())
        for ln in ax.get_lines():
            x, y = ln.get_xdata(orig=False), ln.get_ydata(orig=False)
            p['lines'].append({'x': [num(v) for v in x], 'y': [num(v) for v in y], **hexc(ln.get_color(), ln.get_alpha()),
                               'w': ln.get_linewidth(), 'ls': ln.get_linestyle(), 'ds': ln.get_drawstyle(),
                               'marker': ln.get_marker() if ln.get_marker() not in (None, 'None', '') else None,
                               'ms': ln.get_markersize(), 'z': ln.get_zorder(), 'name': ln.get_label()})
        for pt in ax.patches:
            if isinstance(pt, Rectangle):
                p['bars'].append({'x': num(pt.get_x()), 'y': num(pt.get_y()), 'w': num(pt.get_width()),
                                  'h': num(pt.get_height()), **hexc(pt.get_facecolor())})
        for col in ax.collections:
            if isinstance(col, PathCollection) and len(col.get_offsets()):
                fc, ec = col.get_facecolors(), col.get_edgecolors()
                sz = col.get_sizes()
                for i, (x, y) in enumerate(col.get_offsets()):
                    f = fc[i % len(fc)] if len(fc) else (0, 0, 0, 0)
                    e = ec[i % len(ec)] if len(ec) else f
                    p['points'].append({'x': num(x), 'y': num(y), 'f': mc.to_hex(f[:3]), 'fa': round(float(f[3]), 3),
                                        'e': mc.to_hex(e[:3]), 's': num(sz[i % len(sz)] if len(sz) else 20)})
            elif isinstance(col, PolyCollection):
                fc = col.get_facecolors()
                for i, path in enumerate(col.get_paths()):
                    f = fc[i % len(fc)] if len(fc) else (0, 0, 0, .2)
                    p['fills'].append({'xy': [[num(a), num(b)] for a, b in path.vertices], 'c': mc.to_hex(f[:3]),
                                       'a': round(float(f[3]), 3)})
            elif isinstance(col, LineCollection):
                for seg, c in zip(col.get_segments(), list(col.get_colors()) * len(col.get_segments())):
                    p['lines'].append({'x': [num(v) for v in seg[:, 0]], 'y': [num(v) for v in seg[:, 1]],
                                       'c': mc.to_hex(c[:3]), 'a': round(float(c[3]), 3), 'w': 1, 'ls': '-',
                                       'ds': 'default', 'marker': None, 'ms': 0, 'z': 2, 'name': ''})
        skip = {id(t) for t in yt}
        for t in ax.texts:
            if id(t) in skip or t.get_text() == REC['labels'].get(id(ax)) or not t.get_text().strip():
                continue
            coords = 'axes' if t.get_transform() == ax.transAxes else 'data'
            x, y = t.get_position()
            if hasattr(t, 'xy') and coords == 'data':  # annotations: the anchor point
                x, y = t.xy
            p['texts'].append({'text': t.get_text(), 'x': num(x), 'y': num(y), 'coords': coords,
                               **hexc(t.get_color()), 'ha': t.get_ha()})
        panels.append(p)
    lg = REC['canvas'].get('legend') or REC.get('legends', {}).get(id(fig)) or []
    REC['canvas']['legend'] = [{'name': it[0], **hexc(it[1]), 'kind': it[2] if len(it) > 2 else 'line'} for it in lg]
    data = {'source': str(script), 'canvas': REC['canvas'], 'size_in': [round(v, 3) for v in fig.get_size_inches()],
            'panels': panels}
    out.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')))
    print(f'exported {script.name}: {len(panels)} panels, '
          f'{sum(len(p["lines"]) for p in panels)} lines, {sum(len(p["bars"]) for p in panels)} bars, '
          f'{sum(len(p["points"]) for p in panels)} points -> {out}')


if _canvas:
    style.canvas, style.panel_label, style.save = canvas, panel_label, dump
for name in ('header_legend', 'fig_header_legend'):
    if hasattr(style, name):
        setattr(style, name, _legend_wrap(getattr(style, name)))
import matplotlib.figure  # noqa: E402
_savefig = matplotlib.figure.Figure.savefig
matplotlib.figure.Figure.savefig = lambda self, *a, **k: dump(self)
import os  # noqa: E402
os.chdir(script.parent)
runpy.run_path(str(script), run_name='__main__')
