"""Analysis methods for CodeGraph — semantic neighbors, clone clusters,
dead code, test coverage, trait filtering, PageRank repo map.

Split out of ``graph.py`` to keep the core extractor / storage layer
under ~1500 LOC. Designed as a mixin: every method uses
``self.db_path`` / ``self.project_root`` / ``self.faiss_*`` / etc.,
which ``CodeGraph`` provides.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from collections import defaultdict
from typing import Optional


logger = logging.getLogger(__name__)


class GraphAnalysisMixin:
    """Adds analysis methods to ``CodeGraph``."""

    _PRIMARY_DEF_TYPES = frozenset({
        "class", "function", "method", "component",
        "interface", "enum", "type", "hook",
    })

    _DEFAULT_TEST_GLOBS = (
        # Python
        "test_*.py", "*_test.py", "**/tests/**", "**/test/**", "**/conftest.py",
        # JS / TS
        "*.test.js", "*.test.jsx", "*.test.ts", "*.test.tsx",
        "*.spec.js", "*.spec.jsx", "*.spec.ts", "*.spec.tsx",
        "**/__tests__/**", "**/spec/**",
        # Go
        "*_test.go",
        # JVM
        "*Test.java", "*Tests.java", "*Test.kt", "*Tests.kt",
        # Rust
        "**/tests/**", "**/benches/**",
    )

    @classmethod
    def _is_test_path(cls, rel_path: str, globs: tuple[str, ...]) -> bool:
        from fnmatch import fnmatch
        # fnmatch doesn't understand `**`; treat it as a substring marker.
        normalized = rel_path.replace("\\", "/")
        for glob in globs:
            if "**" in glob:
                segment = glob.replace("**", "").strip("/")
                if segment and segment in normalized:
                    return True
            elif fnmatch(normalized, glob) or fnmatch(normalized.split("/")[-1], glob):
                return True
        return False

    # ── FAISS-driven semantic search ─────────────────────────────────────────

    def find_similar_entities(
        self,
        entity_name: str,
        limit: int = 10,
        min_score: float = 0.4,
        entity_types: Optional[list[str]] = None,
    ) -> dict:
        """Semantically nearest entities to ``entity_name`` via FAISS.

        Use case: dedup detection. "Is there already a helper that does
        this?" Cross-checks neither Grep nor name-substring search can
        answer because matches are by *meaning*, not lexical overlap.

        The anchor and the candidates are filtered to "real definitions"
        (class/function/method/component/...) by default so that import
        rows whose snippet is just `import { A, B, C }` don't drag in
        their co-imported siblings.
        """
        from .embedder import encode_query

        if self.faiss_index is None or not self.faiss_names or self._faiss_dirty:
            self._rebuild_faiss()
        # Пару снимаем одной операцией: пересборка индекса в другом потоке
        # иначе подставляла новые имена к старому индексу, и поиск возвращал
        # чужие сущности.
        faiss_index, faiss_names = self.faiss_pair
        if faiss_index is None or not faiss_names:
            return {
                "anchor": entity_name,
                "results": [],
                "warning": "Graph FAISS index is empty — run rag_rebuild first.",
            }

        type_filter = set(entity_types) if entity_types else set(self._PRIMARY_DEF_TYPES)

        # Pick the best anchor row: prefer entries whose type lands in the
        # filter set AND whose description marks a real declaration ("Extracted
        # from <node>") over a regex sweep ("Referenced call target").
        primary_marker = ",".join("?" * len(type_filter)) if type_filter else "''"
        with sqlite3.connect(self.db_path) as con:
            row = con.execute(
                f"""
                SELECT file, name, description, snippet, type FROM entities
                WHERE name = ?
                ORDER BY
                  CASE WHEN type IN ({primary_marker}) THEN 0 ELSE 1 END,
                  CASE WHEN description LIKE 'Extracted from%' THEN 0 ELSE 1 END,
                  CASE WHEN snippet IS NULL OR snippet = '' THEN 1 ELSE 0 END
                LIMIT 1
                """,
                (entity_name, *type_filter) if type_filter else (entity_name,),
            ).fetchone()
            if not row:
                return {"anchor": entity_name, "results": [], "warning": f"Entity {entity_name!r} not in graph."}
            anchor_file, anchor_name, anchor_desc, anchor_snip, _ = row
            rels_rows = con.execute(
                "SELECT relation, to_name FROM relations WHERE file = ? AND from_name = ?",
                (anchor_file, anchor_name),
            ).fetchall()
        anchor_text = self._faiss_entity_text(
            anchor_name, anchor_desc, anchor_snip,
            relations=[(r[0], r[1]) for r in rels_rows],
        )

        try:
            q_vec = encode_query([anchor_text], normalize_embeddings=True,
                                 show_progress_bar=False).astype("float32")
            k = min(faiss_index.ntotal, max(limit * 8, 80))
            scores, indices = faiss_index.search(q_vec, k)
        except Exception as e:
            logger.warning("find_similar_entities: FAISS query failed: %s", e)
            return {"anchor": entity_name, "results": [], "warning": str(e)}

        seen: set[tuple[str, str]] = set()
        results: list[dict] = []
        with sqlite3.connect(self.db_path) as con:
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or score < min_score:
                    continue
                if idx >= len(faiss_names):
                    continue
                cand_name = faiss_names[idx]
                if cand_name == entity_name:
                    continue
                row = con.execute(
                    f"""
                    SELECT file, name, type, description, line_start, line_end, snippet
                    FROM entities WHERE name = ?
                    ORDER BY
                      CASE WHEN type IN ({','.join('?' * len(type_filter))}) THEN 0 ELSE 1 END,
                      CASE WHEN description LIKE 'Extracted from%' THEN 0 ELSE 1 END,
                      CASE WHEN line_start IS NULL THEN 1 ELSE 0 END,
                      file, line_start
                    LIMIT 1
                    """,
                    (cand_name, *type_filter),
                ).fetchone()
                if not row:
                    continue
                if row[2] not in type_filter:
                    continue
                key = (row[0], row[1])
                if key in seen:
                    continue
                seen.add(key)
                results.append({
                    "file": row[0], "name": row[1], "type": row[2],
                    "description": row[3], "line_start": row[4],
                    "line_end": row[5], "snippet": row[6] or "",
                    "score": float(score),
                })
                if len(results) >= limit:
                    break
        return {"anchor": entity_name, "results": results, "warning": None}

    # ── Dead-code & clone clustering ─────────────────────────────────────────

    # Methods dispatched by Angular runtime — they have no source-level
    # callers, so dead-code's "zero incoming edges" rule mis-flags them.
    _ANGULAR_FRAMEWORK_METHODS = frozenset({
        "ngOnInit", "ngOnDestroy", "ngOnChanges", "ngDoCheck",
        "ngAfterContentInit", "ngAfterContentChecked",
        "ngAfterViewInit", "ngAfterViewChecked",
        "transform", "resolve", "intercept",
        "canActivate", "canActivateChild", "canDeactivate",
        "canLoad", "canMatch",
        "writeValue", "registerOnChange", "registerOnTouched", "setDisabledState",
        "validate", "ngDoBootstrap",
    })

    # File-name suffixes whose methods are dispatched by templates / DI / decorators
    # rather than direct calls. Used when filter_angular=True to suppress those
    # noisy false positives wholesale.
    _ANGULAR_FRAMEWORK_FILE_SUFFIXES = (
        ".component.ts", ".pipe.ts", ".guard.ts", ".interceptor.ts",
        ".resolver.ts", ".directive.ts", ".module.ts",
    )

    def find_dead_code(
        self,
        entity_types: Optional[list[str]] = None,
        limit: int = 50,
        exclude_paths: Optional[list[str]] = None,
        filter_angular: bool = True,
    ) -> list[dict]:
        """Entities that no relation points to — never called, used, or instantiated.

        Defaults to functions/methods/classes/components since "dead" import
        or property symbols are usually external references, not local defs.

        ``exclude_paths`` is an optional list of fnmatch globs (e.g.
        ``["demoapp/*", "**/*.stories.*"]``).

        ``filter_angular`` (default True) drops two classes of well-known false
        positives that the call-graph extractor can't see end-to-end:
          * Methods whose name matches an Angular lifecycle / pipe / resolver /
            guard / interceptor / ControlValueAccessor hook — those are
            dispatched by the framework, not via an explicit call site.
          * Public methods declared in ``*.component.ts`` / ``*.pipe.ts`` /
            ``*.guard.ts`` / ``*.interceptor.ts`` / ``*.resolver.ts`` /
            ``*.directive.ts`` files — almost always wired by templates or DI
            tokens, which the extractor doesn't surface as call edges.
        Set to False to see the raw graph results (useful when auditing
        backend-only repos, or after the template scanner has been enriched).
        """
        types = entity_types or ["function", "method", "class", "component", "interface"]
        placeholders = ",".join("?" * len(types))
        # Over-fetch so client-side filters (exclude_paths, framework filter)
        # still have enough rows to honour ``limit``.
        over_fetch = limit
        if exclude_paths:
            over_fetch *= 5
        if filter_angular:
            over_fetch *= 4
        sql_limit = max(limit, over_fetch)
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute(
                f"""
                SELECT e.file, e.name, e.type, e.description, e.line_start, e.line_end, e.snippet
                FROM entities e
                WHERE e.type IN ({placeholders})
                  AND (e.description IS NULL
                       OR (e.description NOT LIKE 'Template %'
                           AND e.description NOT LIKE 'JSX component reference%'))
                  AND NOT EXISTS (
                      SELECT 1 FROM relations r
                      WHERE r.to_name = e.name
                        AND r.relation IN ('calls', 'uses', 'instantiates', 'inherits')
                  )
                ORDER BY e.file, e.line_start
                LIMIT ?
                """,
                (*types, sql_limit),
            ).fetchall()

        if filter_angular:
            hooks = self._ANGULAR_FRAMEWORK_METHODS
            suffixes = self._ANGULAR_FRAMEWORK_FILE_SUFFIXES
            rows = [
                r for r in rows
                if r[1] not in hooks
                and not any(r[0].endswith(sfx) for sfx in suffixes)
            ]

        if exclude_paths:
            from fnmatch import fnmatch
            rows = [r for r in rows if not any(fnmatch(r[0], g) for g in exclude_paths)]

        # Test functions are dispatched by the framework (pytest/jest collect
        # them by NAME pattern, never by an explicit call) — every test_*
        # definition would be flagged dead forever. Drop entities defined in
        # test files; "is this test dead?" is not a question this tool can
        # answer from the call graph.
        _t = re.compile(
            r"(^|/)(tests?|__tests__)/|(^|/)test_[^/]+$|[._-](test|spec)\.[a-z]+$|_test\.[a-z]+$",
            re.IGNORECASE,
        )
        rows = [r for r in rows if not _t.search(r[0].replace("\\", "/"))]

        # CONTENT VERIFICATION: the graph's incoming edges are only as good
        # as the build agent's reverse-grep diligence — same-file calls and
        # by-reference usages (tool registries, callbacks, lambdas) are
        # chronically under-recorded, producing false "dead" claims. Trust
        # the actual file content instead: a candidate with a bare-name
        # match in the FTS index that is neither its own definition line nor
        # a comment is ALIVE and gets dropped. Biases toward "alive"
        # (string/docstring mentions also count) — correct bias for a tool
        # whose output people act on by deleting code.
        # ``needed=limit`` verifies incrementally until ``limit`` survivors
        # are collected — NEVER appending an unverified tail (the old
        # ``cap=500`` behavior silently passed candidates 501+ through
        # unverified, which on big repos meant most of the output).
        rows = self._verify_dead_candidates(rows, needed=limit)

        return [
            {"file": r[0], "name": r[1], "type": r[2], "description": r[3],
             "line_start": r[4], "line_end": r[5], "snippet": r[6] or ""}
            for r in rows[:limit]
        ]

    # A line only counts as the DEFINITION of <name> when the name follows
    # the declaring keyword directly (``def name``, ``const name =``).
    # A generic "line starts with const/def" check is NOT enough: usage
    # lines like ``const ready = await waitForAgent();`` start with a
    # declaring keyword too, and treating them as definitions made the
    # verifier skip real call sites (false "dead" claims).
    _DEF_LINE_TMPL = (
        r"^\s*(?:export\s+(?:default\s+)?)?(?:public\s+|private\s+|protected\s+|static\s+|abstract\s+)*"
        r"(?:async\s+)?"
        r"(?:(?:def|class|function|interface|type|enum|fn|func)\s+{name}\b"
        r"|(?:const|let|var)\s+{name}\s*[=:(]"
        r"|{name}\s*[:=]\s*(?:async\s+)?(?:function\b|\()"
        r")"
    )

    def _verify_dead_candidates(
        self, rows: list, needed: Optional[int] = None, max_checks: int = 2000
    ) -> list:
        """Drop dead-code candidates that file content proves alive.

        ``needed`` stops the scan once that many candidates survived —
        bounds FTS work to roughly ``needed + <false positives seen>``
        searches instead of verifying the whole over-fetched list.
        ``max_checks`` is a hard ceiling on FTS searches; when hit, the
        REMAINING rows are dropped, not passed through unverified — an
        unverified "dead" claim is worse than a shorter list (people
        delete code based on this output).
        """
        import re as _re
        search = getattr(self, "search_regex", None)
        if search is None:
            return rows
        kept = []
        checks = 0
        for r in rows:
            if needed is not None and len(kept) >= needed:
                break
            if checks >= max_checks:
                break
            file, name, line_start = r[0], r[1], r[4]
            if not name or not _re.match(r"^[\w$.]+$", name):
                kept.append(r)          # unsafe to regex-verify — keep as-is
                continue
            bare = name.rsplit(".", 1)[-1]  # method names stored as Cls.meth
            if len(bare) < 3:
                kept.append(r)
                continue
            checks += 1
            try:
                # Word-boundary in the FTS query itself (the literal
                # extractor handles \b since the regex-escape fix). Without
                # it a short name like "ref" saturates the 60-match window
                # with substring hits (href/referrer/references) and the
                # real call sites never reach the client-side filter —
                # the candidate then reads falsely dead.
                out = search(pattern=rf"\b{_re.escape(bare)}\b", limit=60)
            except Exception:
                kept.append(r)
                continue
            word_re = _re.compile(rf"\b{_re.escape(bare)}\b")
            def_re = _re.compile(self._DEF_LINE_TMPL.format(name=_re.escape(bare)))
            alive = False
            for m in out.get("matches", []):
                ctx = (m.get("context") or "").strip()
                if not word_re.search(ctx):
                    continue
                # its own definition (same file, at/near the recorded line)
                if m.get("file") == file and line_start and abs((m.get("line") or 0) - line_start) <= 2:
                    continue
                # a line that DECLARES this exact name (any file)
                if def_re.match(ctx):
                    continue
                # comment-only lines
                if ctx.startswith(("#", "//", "*", "/*", "<!--", '"""', "'''")):
                    continue
                alive = True
                break
            if not alive:
                kept.append(r)
        return kept

    def find_clones(
        self,
        min_score: float = 0.85,
        min_shape_overlap: float = 0.3,
        top_k_per_entity: int = 5,
        entity_types: Optional[list[str]] = None,
        limit: int = 50,
    ) -> dict:
        """Detect clusters of semantically + structurally similar definitions.

        Two entities are paired as clones when:
          1. FAISS cosine similarity >= ``min_score`` (semantic match).
          2. Outgoing-relation Jaccard overlap >= ``min_shape_overlap``
             (structural match: same calls/uses/instantiates downstream).

        Pairs merged via union-find. Same-file/same-name pairs dropped.
        """
        if self.faiss_index is None or not self.faiss_names or self._faiss_dirty:
            self._rebuild_faiss()
        faiss_index, faiss_names = self.faiss_pair
        if faiss_index is None or faiss_index.ntotal == 0:
            return {"clusters": [], "warning": "Graph FAISS index is empty — run rag_rebuild first."}

        types = set(entity_types) if entity_types else set(self._PRIMARY_DEF_TYPES)

        with sqlite3.connect(self.db_path) as con:
            all_entities = con.execute(
                "SELECT file, name, type, line_start, line_end FROM entities ORDER BY id"
            ).fetchall()
            rels_rows = con.execute(
                "SELECT file, from_name, relation, to_name FROM relations"
            ).fetchall()

        if len(all_entities) != len(faiss_names):
            return {"clusters": [], "warning": "FAISS index out of sync; run rag_rebuild."}

        shape_map: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
        for f, fn, r, tn in rels_rows:
            shape_map[(f, fn)].add((r, tn))

        candidate_idxs = [
            i for i, row in enumerate(all_entities)
            if row[2] in types and shape_map.get((row[0], row[1]))
        ]
        if not candidate_idxs:
            return {"clusters": [], "warning": None}

        try:
            import numpy as np
            vecs = np.stack([faiss_index.reconstruct(int(i)) for i in candidate_idxs])
            k = min(faiss_index.ntotal, max(top_k_per_entity * 4, top_k_per_entity + 2))
            scores, neighbors = faiss_index.search(vecs, k)
        except Exception as e:
            return {"clusters": [], "warning": f"FAISS query failed: {e}"}

        candidate_set = set(candidate_idxs)

        # Union-find with path compression.
        parent: dict[int, int] = {}

        def find(x: int) -> int:
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        edges: list[tuple[int, int, float, float]] = []
        seen_pairs: set[tuple[int, int]] = set()

        for q_pos, q_idx in enumerate(candidate_idxs):
            q_row = all_entities[q_idx]
            q_key = (q_row[0], q_row[1])
            for score, n_idx in zip(scores[q_pos], neighbors[q_pos]):
                n_idx = int(n_idx)
                if n_idx < 0 or n_idx == q_idx or n_idx not in candidate_set:
                    continue
                if score < min_score:
                    continue
                pair = (min(q_idx, n_idx), max(q_idx, n_idx))
                if pair in seen_pairs:
                    continue
                n_row = all_entities[n_idx]
                if q_row[0] == n_row[0] and q_row[1] == n_row[1]:
                    continue
                n_key = (n_row[0], n_row[1])
                a, b = shape_map.get(q_key, set()), shape_map.get(n_key, set())
                union_size = len(a | b)
                if union_size == 0:
                    continue
                overlap = len(a & b) / union_size
                if overlap < min_shape_overlap:
                    continue
                seen_pairs.add(pair)
                edges.append((q_idx, n_idx, float(score), overlap))
                union(q_idx, n_idx)

        if not edges:
            return {"clusters": [], "warning": None}

        cluster_pairs: dict[int, list[tuple[int, int, float, float]]] = defaultdict(list)
        for a, b, sc, ov in edges:
            cluster_pairs[find(a)].append((a, b, sc, ov))

        out: list[dict] = []
        for root, pair_list in cluster_pairs.items():
            members_idx: set[int] = set()
            for a, b, _, _ in pair_list:
                members_idx.add(a)
                members_idx.add(b)
            avg_score = sum(s for _, _, s, _ in pair_list) / len(pair_list)
            avg_overlap = sum(o for _, _, _, o in pair_list) / len(pair_list)
            members = sorted(
                (
                    {
                        "file": all_entities[i][0],
                        "name": all_entities[i][1],
                        "type": all_entities[i][2],
                        "line_start": all_entities[i][3],
                        "line_end": all_entities[i][4],
                    }
                    for i in members_idx
                ),
                key=lambda m: (m["file"], m["line_start"] or 0, m["name"]),
            )
            out.append({
                "members": members,
                "avg_score": round(avg_score, 4),
                "avg_shape_overlap": round(avg_overlap, 4),
                "pair_count": len(pair_list),
            })

        out.sort(key=lambda c: (c["avg_score"], c["avg_shape_overlap"]), reverse=True)
        return {"clusters": out[:limit], "warning": None}

    # ── Test coverage map ────────────────────────────────────────────────────

    def find_test_coverage(
        self,
        mode: str = "summary",
        entity_name: Optional[str] = None,
        test_globs: Optional[list[str]] = None,
        target_path_filter: Optional[str] = None,
        limit: int = 50,
    ) -> dict:
        """Reverse-traverse from test files to surface uncovered prod defs.

        Modes: ``summary`` (counts), ``uncovered`` (list), ``entity``
        (which tests touch this entity).
        """
        globs = tuple(test_globs) if test_globs else self._DEFAULT_TEST_GLOBS

        with sqlite3.connect(self.db_path) as con:
            file_rows = con.execute("SELECT DISTINCT file FROM entities").fetchall()
            ent_rows = con.execute(
                "SELECT file, name, type, line_start FROM entities "
                "WHERE type IN ('function', 'method', 'class', 'component', 'interface')"
            ).fetchall()
            # ``imports`` is included alongside the call/use edges: a test that
            # imports a production symbol by name exercises it, and is a valid
            # coverage signal. Tests commonly record the dependency as
            # ``imports src.pkg.mod.Symbol`` (a dotted path) rather than a bare
            # ``calls Symbol`` — both forms are resolved below.
            rel_rows = con.execute(
                "SELECT file, from_name, relation, to_name FROM relations "
                "WHERE relation IN ('calls', 'uses', 'instantiates', 'imports')"
            ).fetchall()

        test_files = {f[0] for f in file_rows if self._is_test_path(f[0], globs)}
        prod_files = {f[0] for f in file_rows if f[0] not in test_files}

        coverage: dict[str, set[tuple[str, str]]] = defaultdict(set)
        for src_file, src_name, _rel, target in rel_rows:
            if src_file in test_files:
                # Match both the raw target and its bare leaf (last dotted
                # segment), so ``imports src.models.enums.ChatMode`` covers the
                # production entity named ``ChatMode``. Leaf keys that don't
                # match any production entity name are simply never looked up,
                # so this cannot inflate the covered count with phantoms.
                coverage[target].add((src_file, src_name))
                leaf = target.rsplit(".", 1)[-1]
                if leaf and leaf != target:
                    coverage[leaf].add((src_file, src_name))

        prod_entities = [
            {"file": f, "name": n, "type": t, "line_start": ls}
            for f, n, t, ls in ent_rows
            if f not in test_files
            and (not target_path_filter or target_path_filter in f.replace("\\", "/"))
        ]

        if mode == "entity":
            if not entity_name:
                return {"error": "mode='entity' requires entity_name"}
            hits = sorted(coverage.get(entity_name, set()))
            return {
                "entity": entity_name,
                "test_count": len(hits),
                "tests": [{"file": f, "from": fn} for f, fn in hits[:limit]],
            }

        covered = [e for e in prod_entities if e["name"] in coverage]
        uncovered = [e for e in prod_entities if e["name"] not in coverage]

        if mode == "uncovered":
            return {
                "uncovered_count": len(uncovered),
                "test_files": len(test_files),
                "uncovered": sorted(uncovered, key=lambda e: (e["file"], e["line_start"] or 0))[:limit],
            }

        by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"covered": 0, "uncovered": 0})
        for e in covered:
            by_type[e["type"]]["covered"] += 1
        for e in uncovered:
            by_type[e["type"]]["uncovered"] += 1
        total = len(prod_entities)
        return {
            "test_files": len(test_files),
            "production_files": len(prod_files),
            "production_entities": total,
            "covered": len(covered),
            "uncovered": len(uncovered),
            "coverage_pct": round((len(covered) / total * 100) if total else 0.0, 1),
            "by_type": dict(by_type),
        }

    # ── Trait filter ─────────────────────────────────────────────────────────

    def find_by_trait(
        self,
        traits: list[str],
        entity_types: Optional[list[str]] = None,
        path_filter: Optional[str] = None,
        limit: int = 50,
        match: str = "all",
    ) -> list[dict]:
        """Filter entities by detected traits (async/generator/abstract/...).

        ``match='all'`` requires every trait; ``match='any'`` returns
        entities carrying at least one.
        """
        traits = [t.strip().lower() for t in (traits or []) if t.strip()]
        if not traits:
            return []
        type_filter = list(entity_types or [])
        clauses: list[str] = []
        params: list = []
        for t in traits:
            clauses.append("instr(' ' || COALESCE(traits, '') || ' ', ?) > 0")
            params.append(f" {t} ")
        joiner = " AND " if match == "all" else " OR "
        where = "(" + joiner.join(clauses) + ")"
        if type_filter:
            where += " AND type IN (" + ",".join("?" * len(type_filter)) + ")"
            params.extend(type_filter)
        if path_filter:
            where += " AND file LIKE ?"
            params.append(f"%{path_filter}%")
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute(
                f"SELECT file, name, type, line_start, line_end, traits FROM entities "
                f"WHERE {where} ORDER BY file, line_start LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [
            {"file": r[0], "name": r[1], "type": r[2],
             "line_start": r[3], "line_end": r[4], "traits": (r[5] or "").split()}
            for r in rows
        ]

    # ── PageRank repo map ────────────────────────────────────────────────────

    def build_repo_map(
        self,
        token_budget: int = 8000,
        focus_files: Optional[list[str]] = None,
        focus_entities: Optional[list[str]] = None,
        chars_per_token: int = 4,
    ) -> dict:
        """Token-budgeted skeleton ranked by personalized PageRank.

        Inspired by Aider's repo-map: nodes are (file, name) pairs, edges
        are calls/uses/imports/instantiates/inherits. ``focus_files`` get
        a 10x personalization weight, ``focus_entities`` get 50x.
        """
        try:
            import networkx as nx
        except ImportError:
            return {
                "markdown": "",
                "warning": "networkx not installed — `pip install networkx` to enable graph_repo_map.",
                "selected_count": 0,
            }

        with sqlite3.connect(self.db_path) as con:
            # Only include entities that have a `defines` edge from their
            # own file. This drops the "phantom" rows that the symbol-relation
            # extractor inserts whenever a file *references* a name (JSX tags,
            # call sites, type annotations) — those would otherwise dominate
            # PageRank with names like ReactNode/dayjs/AButton in every file
            # that uses them.
            # Prefer entities with a `defines` edge (cleanest dedup).
            # Fall back to raw entity list when the graph was built via
            # manual write_batch (which skips the `defines` self-edge).
            ent_rows = con.execute(
                "SELECT e.file, e.name, e.type, e.line_start, e.line_end, e.snippet, e.description "
                "FROM entities e "
                "WHERE e.type IN ('class','function','method','component','interface','module','enum') "
                "  AND EXISTS ("
                "    SELECT 1 FROM relations r "
                "    WHERE r.relation = 'defines' "
                "      AND r.file = e.file "
                "      AND r.from_name = e.file "
                "      AND r.to_name = e.name"
                "  )"
            ).fetchall()
            if not ent_rows:
                ent_rows = con.execute(
                    "SELECT e.file, e.name, e.type, e.line_start, e.line_end, e.snippet, e.description "
                    "FROM entities e "
                    "WHERE e.type IN ('class','function','method','component','interface','module','enum')"
                ).fetchall()
            rel_rows = con.execute(
                "SELECT file, from_name, relation, to_name FROM relations "
                "WHERE relation IN ('calls','uses','instantiates','inherits','imports')"
            ).fetchall()
            # File-fanout count per name — ambiguous identifiers (Input, Title,
            # Text, Typography…) appear in many files and we can't reliably
            # attribute reference edges to a single definition. PageRank then
            # over-credits whichever node was picked first. Track counts so the
            # edge-wiring loop can skip ambiguous targets.
            name_fanout = dict(
                con.execute(
                    "SELECT name, COUNT(DISTINCT file) FROM entities "
                    "WHERE type IN ('class','function','method','component','interface','module','enum') "
                    "GROUP BY name"
                ).fetchall()
            )

        if not ent_rows:
            return {"markdown": "", "warning": "Graph is empty — run rag_rebuild first.", "selected_count": 0}

        graph = nx.DiGraph()
        ent_by_key: dict[tuple[str, str], dict] = {}
        for f, n, t, ls, le, sn, desc in ent_rows:
            key = (f, n)
            ent_by_key[key] = {
                "file": f, "name": n, "type": t,
                "line_start": ls, "line_end": le,
                "snippet": sn or "", "description": desc or "",
            }
            graph.add_node(key)

        name_to_files: dict[str, list[str]] = {}
        for (f, n) in ent_by_key:
            name_to_files.setdefault(n, []).append(f)

        # Ambiguity threshold: names defined in this many files (or more)
        # in the raw entity table are skipped during edge wiring. Empirically
        # 10 captures ant/MUI re-exports (Input=32, Typography=141, Text=182)
        # while keeping legitimately-popular project names (Layout, Header).
        FANOUT_AMBIGUITY_THRESHOLD = 10
        for src_file, src_name, _rel, dst_name in rel_rows:
            src_key = (src_file, src_name)
            if src_key not in ent_by_key:
                continue
            if name_fanout.get(dst_name, 0) >= FANOUT_AMBIGUITY_THRESHOLD:
                continue
            target_files = name_to_files.get(dst_name) or []
            if src_file in target_files:
                dst_key = (src_file, dst_name)
            elif target_files:
                dst_key = (target_files[0], dst_name)
            else:
                continue
            if dst_key == src_key:
                continue
            if graph.has_edge(src_key, dst_key):
                graph[src_key][dst_key]["weight"] += 1.0
            else:
                graph.add_edge(src_key, dst_key, weight=1.0)

        focus_files_norm = {f.replace("\\", "/") for f in (focus_files or [])}
        focus_entity_set = set(focus_entities or [])
        personalization: dict[tuple[str, str], float] = {}
        for key in graph.nodes:
            f, n = key
            weight = 1.0
            if focus_files_norm and any(ff in f.replace("\\", "/") for ff in focus_files_norm):
                weight += 10.0
            if n in focus_entity_set:
                weight += 50.0
            personalization[key] = weight

        try:
            scores = nx.pagerank(graph, alpha=0.85, personalization=personalization, max_iter=100)
        except Exception as e:
            logger.warning("PageRank failed (%s); falling back to in-degree.", e)
            scores = {k: float(graph.in_degree(k, weight="weight")) for k in graph.nodes}

        budget_chars = max(1000, token_budget * chars_per_token)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

        selected_by_file: dict[str, list[dict]] = {}
        used = 0
        selected_count = 0
        for key, _score in ranked:
            ent = ent_by_key[key]
            line = self._repo_map_line(ent)
            cost = len(line) + 2
            if used + cost > budget_chars:
                continue
            selected_by_file.setdefault(ent["file"], []).append(ent)
            used += cost
            selected_count += 1
            if used >= budget_chars * 0.95:
                break

        file_best_score = {
            f: max(scores.get((e["file"], e["name"]), 0.0) for e in entries)
            for f, entries in selected_by_file.items()
        }
        ordered_files = sorted(selected_by_file, key=lambda f: file_best_score[f], reverse=True)

        out_lines = [f"# Repo map ({selected_count} entities, ~{used // chars_per_token} tokens)"]
        if focus_files_norm or focus_entity_set:
            bits = []
            if focus_files_norm:
                bits.append(f"focus_files={sorted(focus_files_norm)}")
            if focus_entity_set:
                bits.append(f"focus_entities={sorted(focus_entity_set)}")
            out_lines.append(f"_Personalized: {', '.join(bits)}_")
        for f in ordered_files:
            out_lines.append(f"\n## `{f}`")
            for ent in sorted(selected_by_file[f], key=lambda e: e.get("line_start") or 0):
                out_lines.append("  " + self._repo_map_line(ent))

        return {
            "markdown": "\n".join(out_lines),
            "warning": None,
            "selected_count": selected_count,
            "approx_tokens": used // chars_per_token,
        }

    @staticmethod
    def _repo_map_line(ent: dict) -> str:
        loc = f":{ent['line_start']}" if ent.get("line_start") else ""
        snippet = (ent.get("snippet") or "").strip().splitlines()[0] if ent.get("snippet") else ""
        snippet = snippet[:120] + ("…" if len(snippet) > 120 else "")
        if snippet:
            return f"- [{ent['type']}] **{ent['name']}**{loc} — `{snippet}`"
        desc = (ent.get("description") or "").strip()[:80]
        return f"- [{ent['type']}] **{ent['name']}**{loc}" + (f" — {desc}" if desc else "")
