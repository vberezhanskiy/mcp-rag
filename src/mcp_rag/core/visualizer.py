"""HTML graph visualization.

Generates a self-contained HTML page (single file, vis-network from CDN)
that lets the user explore the project at three drill-down levels:

1. Modules — top-level path-prefix nodes plus cross-module edges
   weighted by underlying relation count. Always shown first.
2. Files — when the user double-clicks a module, render its files plus
   inter-file edges (dashed if the link leaves the module).
3. Entities — when the user double-clicks a file, render the entity
   list as a side panel (no inner graph for v1).

Data is computed once and embedded as JSON. The page is self-contained
apart from the vis-network CDN script.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Optional


# ────────────────── data shaping ──────────────────

def _module_id_for_file(rel_path: str, depth: int = 2) -> str:
    parts = rel_path.split("/")
    if len(parts) <= 1:
        return "<root>"
    return "/".join(parts[: min(depth, len(parts) - 1)])


def build_visualization_data(
    db_path: Path,
    project_root: Path,
    module_depth: int = 2,
) -> dict:
    """Compute one shared dataset; the JS view filters on demand.

    Returned shape (compact, no per-module duplication):
      {
        "modules":       [{id, files: [rel_paths…]}],
        "file_edges":    [{f, t, w}],     # f → t weighted by relation count
        "module_edges":  [{f, t, w}],     # aggregate of file_edges across modules
        "file_entities": {rel_path: {primary: [{n,t,l}], total: int}},
        "stats":         {modules, files, relations}
      }
    """
    with sqlite3.connect(db_path) as con:
        relations = con.execute(
            "SELECT file, from_name, relation, to_name FROM relations"
        ).fetchall()
        indexed_files = [r[0] for r in con.execute("SELECT file FROM file_meta").fetchall()]
        entities = con.execute(
            "SELECT file, name, type, line_start FROM entities"
        ).fetchall()

    file_to_module: dict[str, str] = {
        f: _module_id_for_file(f, module_depth) for f in indexed_files
    }

    # Resolve each entity name to a SINGLE defining file. Names defined
    # in multiple files (e.g. 'cn', 'props') would otherwise fan a single
    # call out to every defining file and explode the edge count
    # (200 MB+ HTML on ui-kit). Prefer real declarations over regex hits;
    # drop ambiguous names entirely.
    primary_types = {"class", "function", "method", "component", "interface", "enum", "type", "module"}
    entity_def: dict[str, tuple[str, int]] = {}  # name -> (file, priority)
    file_entity_count: dict[str, int] = defaultdict(int)
    for f, name, t, _ls in entities:
        file_entity_count[f] += 1
        if not name or len(name) < 2 or t not in primary_types:
            continue
        prev = entity_def.get(name)
        if prev is None:
            entity_def[name] = (f, 0)
        elif prev[0] != f:
            # Same name defined in another file → ambiguous, demote.
            entity_def[name] = (prev[0], prev[1] + 1)
    # Drop names that are defined in 3+ files — too ambient to attribute.
    entity_def_clean: dict[str, str] = {
        n: meta[0] for n, meta in entity_def.items() if meta[1] < 2
    }

    file_edge_weight: dict[tuple[str, str], int] = defaultdict(int)
    for rel_file, _from, rel_type, to_name in relations:
        if rel_type not in {"calls", "uses", "imports", "instantiates", "inherits"}:
            continue
        target = entity_def_clean.get(to_name)
        if target is None or target == rel_file:
            continue
        file_edge_weight[(rel_file, target)] += 1

    # module-level edges
    module_edge_weight: dict[tuple[str, str], int] = defaultdict(int)
    for (a, b), w in file_edge_weight.items():
        a_mod = file_to_module.get(a)
        b_mod = file_to_module.get(b)
        if not a_mod or not b_mod or a_mod == b_mod:
            continue
        module_edge_weight[(a_mod, b_mod)] += w

    # group files into modules
    module_files: dict[str, list[str]] = defaultdict(list)
    for f, mod in file_to_module.items():
        module_files[mod].append(f)
    modules = [
        {"id": mod, "files": sorted(module_files[mod])}
        for mod in sorted(module_files)
    ]

    # per-file entity summary (cap noise)
    file_entities: dict[str, dict] = {}
    by_file: dict[str, list[tuple[str, str, Optional[int]]]] = defaultdict(list)
    for f, name, t, ls in entities:
        by_file[f].append((name, t, ls))
    for f, ents in by_file.items():
        primary = sorted(
            (e for e in ents if e[1] in primary_types and len(e[0]) > 1),
            key=lambda e: (e[2] or 0, e[0]),
        )[:50]
        file_entities[f] = {
            "primary": [{"n": n, "t": t, "l": ls} for n, t, ls in primary],
            "total": len(ents),
        }

    return {
        "modules": modules,
        "file_edges": [
            {"f": a, "t": b, "w": w}
            for (a, b), w in file_edge_weight.items()
        ],
        "module_edges": [
            {"f": a, "t": b, "w": w}
            for (a, b), w in module_edge_weight.items()
        ],
        "file_entity_count": dict(file_entity_count),
        "file_entities": file_entities,
        "stats": {
            "modules": len(modules),
            "files": len(indexed_files),
            "relations": len(relations),
        },
    }


# ────────────────── HTML rendering ──────────────────

_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>%(title)s</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #1e1e1e; color: #d4d4d4; }
  #header { padding: 10px 16px; background: #2d2d30; border-bottom: 1px solid #3e3e42; display: flex; align-items: center; gap: 12px; flex-wrap: wrap; min-height: 30px; }
  #header h1 { margin: 0; font-size: 14px; font-weight: 600; }
  #breadcrumbs { display: flex; gap: 6px; align-items: center; font-size: 13px; }
  #breadcrumbs button { background: #3e3e42; color: #d4d4d4; border: none; padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 12px; }
  #breadcrumbs button:hover { background: #505054; }
  #breadcrumbs span.sep { color: #6d6d6d; }
  #meta { font-size: 12px; color: #9d9d9d; }
  #legend { font-size: 11px; color: #888; margin-left: auto; }
  #network { width: 100vw; height: calc(100vh - 51px); }
  #side {
    position: fixed; right: 0; top: 51px; width: 360px; height: calc(100vh - 51px);
    background: rgba(40,40,42,0.97); padding: 16px; overflow-y: auto;
    border-left: 1px solid #3e3e42; box-sizing: border-box; display: none; font-size: 13px;
  }
  #side h2 { margin: 0 0 8px 0; font-size: 14px; word-break: break-all; }
  #side .sub { font-size: 11px; color: #888; margin-bottom: 12px; }
  #side ul { margin: 0; padding-left: 18px; }
  #side li { margin: 3px 0; line-height: 1.4; }
  #side .type { color: #888; font-size: 11px; }
  #side .close { float: right; cursor: pointer; color: #888; }
  #side .close:hover { color: #fff; }
</style>
</head>
<body>
<div id="header">
  <h1>%(title)s</h1>
  <div id="breadcrumbs"></div>
  <div id="meta"></div>
  <div id="legend">double-click → drill in · single-click → details</div>
</div>
<div id="network"></div>
<div id="side"></div>
<script>
const DATA = %(data)s;

// Build a lookup: file → module
const FILE_TO_MOD = {};
for (const m of DATA.modules) {
  for (const f of m.files) FILE_TO_MOD[f] = m.id;
}

const palette = {
  module: '#4a9eff',
  file: '#7cc36e',
  fileExternal: '#5a8e57',
};

const TYPE_COLOR = {
  class: '#e07b53', function: '#d4c25b', method: '#c89060',
  component: '#9b6dd9', interface: '#5fb3a1', enum: '#b85eaa',
  type: '#7e9bd9', module: '#888',
};

let stack = [{level: 'modules'}];
let network = null;

const container = document.getElementById('network');
const side = document.getElementById('side');
const breadcrumbs = document.getElementById('breadcrumbs');
const meta = document.getElementById('meta');

function setData(nodes, edges) {
  if (network) network.destroy();
  // Bigger graphs need a stronger settle. Once the layout is stable we
  // turn physics off so dragging / clicking nodes doesn't kick the
  // simulation back into motion.
  const iters = Math.min(800, Math.max(200, nodes.length * 4));
  network = new vis.Network(container, {nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges)}, {
    physics: {
      enabled: true,
      stabilization: { enabled: true, iterations: iters, fit: true, updateInterval: 25 },
      barnesHut: {
        gravitationalConstant: -10000,
        centralGravity: 0.4,
        springLength: 140,
        springConstant: 0.04,
        damping: 0.5,
      },
    },
    layout: { improvedLayout: nodes.length < 200 },
    nodes: { borderWidth: 1, font: { color: '#d4d4d4', size: 14 }, shape: 'dot' },
    edges: { color: { color: '#5a5a5a', highlight: '#aaaaaa' }, smooth: { type: 'continuous' }, arrows: { to: { enabled: true, scaleFactor: 0.5 } } },
    interaction: { hover: true, tooltipDelay: 100, dragNodes: true },
  });
  network.once('stabilizationIterationsDone', () => {
    network.setOptions({ physics: { enabled: false } });
  });
  network.on('click', onClick);
  network.on('doubleClick', onDoubleClick);
}

function renderModules() {
  stack = [{level: 'modules'}];
  side.style.display = 'none';
  const nodes = DATA.modules.map(m => ({
    id: m.id,
    label: m.id + '\n(' + m.files.length + ')',
    title: m.id + '\n' + m.files.length + ' files',
    color: palette.module,
    size: Math.min(50, 14 + m.files.length / 4),
  }));
  const edges = DATA.module_edges.map(e => ({
    from: e.f, to: e.t, value: e.w, width: Math.min(8, 1 + Math.log2(e.w + 1)),
    title: e.f + ' → ' + e.t + ': ' + e.w + ' relations',
  }));
  setData(nodes, edges);
  meta.textContent = DATA.stats.modules + ' modules · ' + DATA.stats.files + ' files · ' + DATA.stats.relations + ' relations';
  renderBreadcrumbs();
}

function renderModule(modId) {
  stack.push({level: 'module', id: modId});
  side.style.display = 'none';
  const mod = DATA.modules.find(m => m.id === modId);
  if (!mod) { renderModules(); return; }
  const inside = new Set(mod.files);
  const externalIds = new Set();
  const edges = [];
  for (const e of DATA.file_edges) {
    const fromIn = inside.has(e.f), toIn = inside.has(e.t);
    if (!fromIn && !toIn) continue;
    if (!fromIn) externalIds.add(e.f);
    if (!toIn) externalIds.add(e.t);
    edges.push({
      from: e.f, to: e.t, value: e.w,
      width: Math.min(5, 1 + Math.log2(e.w + 1)),
      dashes: !(fromIn && toIn),
    });
  }
  const nodes = mod.files.map(f => ({
    id: f,
    label: f.split('/').pop(),
    title: f + '\n' + (DATA.file_entity_count[f] || 0) + ' entities',
    color: palette.file,
    size: Math.min(30, 8 + (DATA.file_entity_count[f] || 0) / 3),
  }));
  externalIds.forEach(id => {
    nodes.push({
      id, label: id.split('/').pop(), title: id + '  (external)',
      color: palette.fileExternal, size: 6,
    });
  });
  setData(nodes, edges);
  meta.textContent = nodes.length + ' files (' + inside.size + ' inside, ' + externalIds.size + ' linked) · ' + edges.length + ' edges';
  renderBreadcrumbs();
}

function renderFileSide(filePath) {
  stack.push({level: 'file', id: filePath});
  const info = DATA.file_entities[filePath];
  side.style.display = 'block';
  let html = '<span class="close" onclick="closeSide()">✕</span>';
  html += '<h2>' + escapeHtml(filePath) + '</h2>';
  if (!info || info.primary.length === 0) {
    html += '<div class="sub">No primary declarations indexed.</div>';
  } else {
    html += '<div class="sub">' + info.primary.length + ' shown of ' + info.total + ' total entities</div>';
    html += '<ul>';
    for (const e of info.primary) {
      const c = TYPE_COLOR[e.t] || '#888';
      html += '<li><span class="type" style="color:' + c + '">[' + e.t + ']</span> <b>' + escapeHtml(e.n) + '</b>';
      if (e.l) html += ' <span class="type">:' + e.l + '</span>';
      html += '</li>';
    }
    html += '</ul>';
  }
  side.innerHTML = html;
  renderBreadcrumbs();
}

function closeSide() {
  side.style.display = 'none';
  if (stack[stack.length - 1].level === 'file') {
    stack.pop();
    renderBreadcrumbs();
  }
}

function renderBreadcrumbs() {
  breadcrumbs.innerHTML = '';
  stack.forEach((s, i) => {
    const btn = document.createElement('button');
    btn.textContent = s.level === 'modules' ? '↑ Modules' : (s.level === 'module' ? s.id : s.id.split('/').pop());
    btn.onclick = () => { stack = stack.slice(0, i); renderAt(s); };
    breadcrumbs.appendChild(btn);
    if (i < stack.length - 1) {
      const sep = document.createElement('span');
      sep.className = 'sep'; sep.textContent = '›';
      breadcrumbs.appendChild(sep);
    }
  });
}

function renderAt(state) {
  if (state.level === 'modules') renderModules();
  else if (state.level === 'module') renderModule(state.id);
  else if (state.level === 'file') renderFileSide(state.id);
}

function onClick(params) {
  if (params.nodes.length === 0) return;
}

function onDoubleClick(params) {
  if (params.nodes.length === 0) return;
  const id = params.nodes[0];
  const cur = stack[stack.length - 1];
  if (cur.level === 'modules') renderModule(id);
  else if (cur.level === 'module') renderFileSide(id);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'})[c]);
}

renderModules();
</script>
</body>
</html>
"""


def render_html(data: dict, title: str) -> str:
    return _HTML_TEMPLATE % {
        "title": title,
        "data": json.dumps(data, ensure_ascii=False, separators=(",", ":")),
    }
