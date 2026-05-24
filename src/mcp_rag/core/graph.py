"""Code Knowledge Graph — tree-sitter + regex extraction, SQLite storage.

Builds a per-project graph of entities (classes, functions, methods, imports,
modules, components, …) and relations (defines, calls, imports, inherits,
uses, instantiates). Falls back to an injectable LLM extractor for files
that neither tree-sitter nor regex parsers can handle.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import sqlite3
import time
import tomllib
from pathlib import Path
from typing import Optional

import gitignore_parser
import json5

from .embedder import encode_batch_size, encode_documents
from .graph_analysis import GraphAnalysisMixin
from .graph_text import GraphTextMixin
from ..llm.extractor import LLMExtractor, NoOpExtractor

logger = logging.getLogger(__name__)


_ALLOWED_ENTITY_TYPES = {
    "class", "function", "method", "import", "module", "interface", "component",
    "hook", "type", "enum", "selector", "style", "template", "config", "variable", "symbol", "property",
    "constant",
    # Framework-specific (Angular / NestJS / React hooks)
    "service", "directive", "pipe", "controller",
    # HTTP/WS routes — same URL path string on frontend (consumer) and
    # backend (handler) lets the graph stitch cross-stack edges that
    # would otherwise be invisible (path is a string literal at both
    # ends, not a Python/TS identifier).
    "endpoint",
}

_ALLOWED_RELATION_TYPES = {
    "defines", "calls", "imports", "inherits", "uses", "instantiates",
}

_TREE_SITTER_LANGUAGES = {
    # ── Mainstream ─────────────────────────────────────────────────────
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".sc": "scala",
    ".go": "go",
    ".rs": "rust",
    # NB: `.cs` parser ('csharp') is built against grammar v15 but tree-sitter
    # 0.23.x ABI only supports v13-v14 — so C# silently falls back to LLM
    # extraction until language-pack ships a v14 grammar. Keep the entry so
    # it activates automatically on future pack/tree-sitter upgrades.
    ".cs": "csharp",
    ".php": "php",
    ".phtml": "php",
    ".rb": "ruby",
    ".pl": "perl",
    ".pm": "perl",
    # ── Systems / native ───────────────────────────────────────────────
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c++": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".hxx": "cpp",
    ".m": "objc",
    ".d": "d",
    ".zig": "zig",
    # ── JVM-adjacent ───────────────────────────────────────────────────
    ".groovy": "groovy",
    ".gradle": "groovy",
    ".clj": "clojure",
    ".cljs": "clojure",
    ".cljc": "clojure",
    ".edn": "clojure",
    # ── Functional ─────────────────────────────────────────────────────
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".hrl": "erlang",
    ".hs": "haskell",
    ".lhs": "haskell",
    # ── Scripting / dynamic ────────────────────────────────────────────
    ".lua": "lua",
    ".jl": "julia",
    ".r": "r",
    ".R": "r",
    ".dart": "dart",
    # ── Shell ──────────────────────────────────────────────────────────
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".fish": "fish",
    ".ps1": "powershell",
    ".psm1": "powershell",
    ".psd1": "powershell",
    # ── Web3 ───────────────────────────────────────────────────────────
    ".sol": "solidity",
    # ── Infrastructure as code ─────────────────────────────────────────
    ".tf": "terraform",
    ".tfvars": "terraform",
    ".hcl": "hcl",
    ".nix": "nix",
    # ── Data / schema ──────────────────────────────────────────────────
    # NB: .sql grammar same v15 incompatibility as csharp until pack upgrade.
    ".sql": "sql",
    ".prisma": "prisma",
    # ── Docs ───────────────────────────────────────────────────────────
    ".md": "markdown",
    ".markdown": "markdown",
    ".tex": "latex",
    ".rst": "rst",
    # ── Hardware / scientific ──────────────────────────────────────────
    ".v": "verilog",
    ".sv": "verilog",
    ".vhdl": "vhdl",
    ".vhd": "vhdl",
    ".f": "fortran",
    ".f90": "fortran",
    ".f95": "fortran",
    # ── Game / scripting ───────────────────────────────────────────────
    ".gd": "gdscript",
    # ── Assembly ───────────────────────────────────────────────────────
    ".asm": "asm",
    ".s": "asm",
    ".inc": "asm",
}

# NB: HTML/CSS/SCSS/LESS не добавляем сюда — `_extractor_strategy` ниже
# отдаёт их специализированным `_extract_stylesheet` / `_extract_template`
# (selectors, CSS vars, Angular component bindings) ДО tree-sitter, и они
# точнее. Tree-sitter html/css парсеры доступны, но overrride'нуть стратегию
# смысла мало — current extractors уже работают хорошо.

_TREE_SITTER_NODE_TYPES = {
    # Generic class / function / method
    "class_definition",
    "class_declaration",
    "function_definition",
    "function_declaration",
    "function_signature",  # Dart
    "method_definition",
    "method_declaration",
    "generator_function_declaration",
    "local_function",  # Lua
    # Variables / fields
    "lexical_declaration",
    "variable_declaration",
    "public_field_definition",
    "field_definition",
    "pair",
    "pair_pattern",
    # Types / interfaces
    "interface_declaration",
    "type_alias_declaration",
    "enum_declaration",
    "abstract_class_declaration",
    "trait_definition",        # Scala
    "object_definition",        # Scala
    "object_declaration",       # Kotlin
    "mixin_declaration",        # Dart
    "data_type",                # Haskell
    # Rust
    "struct_item",
    "enum_item",
    "function_item",
    "impl_item",
    "trait_item",
    "mod_item",
    "type_item",
    # Go
    "type_declaration",
    "type_spec",
    # Solidity
    "contract_declaration",
    "library_declaration",
    # Verilog / VHDL
    "module_declaration",
    "package_declaration",
    "entity_declaration",
    "architecture_body",
    # Fortran
    "subroutine",
    "subroutine_subprogram",
    "function_subprogram",
    "module_subprogram",
    "program_block",
    # Julia
    "struct_definition",
    "module_definition",
    "abstract_definition",
    # Markdown
    "atx_heading",
    "setext_heading",
    # HTML / CSS (used by stylesheet/template extractors)
    "element",
    "script_element",
    "style_element",
    "stylesheet",
    "rule_set",
    "qualified_rule",
    "class_selector",
    "id_selector",
}

_TREE_SITTER_NAME_FIELDS = ("name", "declarator", "property", "left")

_CODE_EXTENSIONS = [
    "*.py", "*.pyi",
    "*.js", "*.mjs", "*.cjs", "*.jsx",
    "*.ts", "*.mts", "*.cts", "*.tsx",
    "*.vue", "*.svelte", "*.astro",
    "*.java", "*.kt", "*.kts", "*.scala", "*.sc", "*.groovy", "*.gradle",
    "*.go", "*.rs", "*.swift", "*.dart", "*.zig", "*.d",
    "*.c", "*.h", "*.cpp", "*.cc", "*.cxx", "*.c++", "*.hpp", "*.hh", "*.hxx",
    "*.m", "*.mm", "*.cu", "*.cuh",
    "*.asm", "*.s", "*.S", "*.inc",
    "*.cs", "*.fs", "*.vb",
    "*.php", "*.phtml", "*.rb", "*.lua", "*.pl", "*.pm", "*.perl",
    "*.sh", "*.bash", "*.zsh", "*.fish",
    "*.ps1", "*.psm1", "*.psd1",
    "*.ex", "*.exs", "*.erl", "*.hrl", "*.hs", "*.lhs",
    "*.clj", "*.cljs", "*.cljc", "*.edn",
    "*.r", "*.R", "*.jl",
    "*.sql", "*.graphql", "*.gql", "*.proto", "*.prisma",
    "*.nim", "*.nims", "*.ml", "*.mli",
    "*.html", "*.htm", "*.css", "*.scss", "*.sass", "*.less",
    "*.jinja", "*.jinja2", "*.j2", "*.njk", "*.hbs", "*.ejs",
    "*.yaml", "*.yml", "*.toml", "*.json",
    "*.tf", "*.tfvars", "*.hcl", "*.nix",
    "*.sol",
    "*.md", "*.markdown", "*.tex", "*.rst",
    "*.v", "*.sv", "*.vhdl", "*.vhd",
    "*.f", "*.f90", "*.f95",
    "*.gd", "*.gdshader", "*.gdshaderinc", "*.godot", "*.cfg",
]

_IGNORE_DIRS = {
    "venv", ".venv", "env", ".env", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "site-packages", ".eggs", "*.egg-info",
    "node_modules", ".npm", ".yarn", ".pnp",
    "dist", "build", "out", "output", ".next", ".nuxt", ".svelte-kit",
    "target", "bin", "obj", "release", "debug",
    ".git", ".svn", ".hg",
    ".idea", ".vscode", ".vs",
    "logs", "log", "tmp", "temp", ".cache", ".tmp",
    ".gradle", "vendor", "CMakeFiles", "coverage", ".coverage",
    "htmlcoverage", ".tox", "buck-out", ".angular",
    ".godot", ".import", "addons",
}

# Files larger than this are skipped — typically minified bundles, lockfiles,
# generated SQL dumps, ML weights. Tree-sitter and regex extractors can hang
# or balloon memory on multi-MB inputs.
_MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MB


# ── Angular template expression scanning ─────────────────────────────────────
# Tree-sitter has no HTML extractor wired up, so methods invoked only from
# templates (.html plus inline `template:` strings) were invisible to the call
# graph — and graph_dead_code flagged every (click)-bound method as orphan.
# These regexes power _extract_template's binding scan; kept module-level so
# they compile once. Coverage:
#   - Event/property/two-way bindings:    (click)="save()"  [disabled]="x"  [(ngModel)]="y"
#   - Structural directives (legacy):     *ngIf="ready"  *ngFor="let u of users()"
#   - Interpolation:                      {{ user.fullName() }}
#   - Angular 17 control flow:            @if (cond) { … } @for (x of items; track …)
#   - Local declarations:                 @let total = sum(items);
_BINDING_ATTR_RE = re.compile(
    r'(?:\([\w.-]+\)|\[\(?[\w.()-]+\)?\]|\*[\w-]+)\s*=\s*["\']([^"\']+)["\']'
)
_INTERPOLATION_RE = re.compile(r'\{\{([^}]+)\}\}')
_CONTROL_FLOW_RE = re.compile(
    r'@(?:if|else\s+if|for|switch|case|defer|placeholder|loading|error|empty)'
    r'\b[^{;]*?\(([^)]*)\)?'
)
_LET_DECL_RE = re.compile(r'@let\s+\w+\s*=\s*([^;]+);')
_TEMPLATE_METHOD_CALL_RE = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*)\s*\(')
_TEMPLATE_IDENT_RE = re.compile(r'\b([a-zA-Z_][A-Za-z0-9_]*)\b(?!\s*\()')

# Template-expression reserved words / Angular built-ins / control-flow trigger
# names that look like identifiers but aren't class members. Filtered out so
# they don't pollute the relations table with edges to nonexistent entities.
# Always filtered, in both 'calls' and 'uses' contexts. These are JS literals
# or block-trigger keywords that can't possibly name a user-defined entity.
_TEMPLATE_HARD_KEYWORDS = frozenset({
    "true", "false", "null", "undefined", "this", "void",
    "let", "as", "of", "in", "track",
    "else", "if", "switch", "case", "default", "empty",
})

# Filtered only when seen as a bare identifier (a 'uses' edge). If the same
# word appears with parens (a 'calls' edge) it's a user method or signal —
# we keep it. Includes:
#   * structural-directive triggers that share names with valid identifiers
#     (`for`, `defer`, `placeholder`, `loading`, `error`, `prefetch`, `when`,
#     `on`, `idle`, `viewport`, `interaction`, `hover`, `immediate`, `timer`);
#   * legacy `*ngFor` implicit locals (`index`, `first`, `last`, `even`, `odd`,
#     `count`) — Angular 17 uses `$index`/`$count` so plain forms are rare in
#     modern code but appear in legacy templates.
_TEMPLATE_EXPR_KEYWORDS = frozenset({
    "for", "defer", "placeholder", "loading", "error", "prefetch",
    "when", "on", "idle", "viewport", "interaction", "hover",
    "immediate", "timer",
    "index", "first", "last", "even", "odd", "count",
})

# JS literals / keywords that may appear inside provideX(...) / withX(...)
# argument lists; filtered out so they don't become phantom 'uses' targets.
_ANGULAR_RUNTIME_KEYWORDS = frozenset({
    "true", "false", "null", "undefined", "void", "this",
    "new", "await", "async", "return", "import", "from", "as",
})

# Names that Angular runtime resolves by interface, not by call site. They have
# no callers in source — the framework dispatches them. Filtered out of
# graph_dead_code so lifecycle hooks / pipe transforms / resolvers / guards
# / interceptors / ControlValueAccessor methods aren't reported as dead.
_ANGULAR_LIFECYCLE_HOOKS = frozenset({
    # Component / Directive
    "ngOnInit", "ngOnDestroy", "ngOnChanges", "ngDoCheck",
    "ngAfterContentInit", "ngAfterContentChecked",
    "ngAfterViewInit", "ngAfterViewChecked",
    # Pipe
    "transform",
    # Resolver
    "resolve",
    # HttpInterceptor
    "intercept",
    # Route guards (functional + class-based)
    "canActivate", "canActivateChild", "canDeactivate",
    "canLoad", "canMatch",
    # ControlValueAccessor
    "writeValue", "registerOnChange", "registerOnTouched", "setDisabledState",
    # Validator / AsyncValidator
    "validate",
    # Bootstrap
    "ngDoBootstrap",
})


# ── Trait detection: per-language regex patterns ────────────────────────────
# Source-of-truth used by _detect_traits. Compiled once at module load.
# Add new languages here, not in the detector body — keeps the function
# pure-data-driven and the regex set easy to audit.

_LANG_FROM_EXT: dict[str, str] = {
    ".py": "python",
    ".js": "js", ".jsx": "js", ".mjs": "js", ".cjs": "js",
    ".ts": "ts", ".tsx": "ts",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".cs": "cs",
    ".kt": "kotlin", ".kts": "kotlin",
    ".swift": "swift",
    ".scala": "scala",
    ".rb": "ruby",
    ".php": "php",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp",
    # `.h` is ambiguous (C or C++). Map to cpp — the cpp pattern set is a
    # superset of the c set, so pure-C headers still get their static/extern
    # detection while C++ headers also pick up pure-virtual abstract markers.
    ".c": "c", ".h": "cpp",
    ".lua": "lua",
    ".dart": "dart",
}


def _rx(pattern: str, flags: int = 0) -> re.Pattern:
    return re.compile(pattern, flags)


_TRAIT_PATTERNS_PER_LANG: dict[str, dict[str, list[re.Pattern]]] = {
    "python": {
        "async":      [_rx(r"\basync\s+def\b")],
        "generator":  [_rx(r"\byield\b")],
        "abstract":   [_rx(r"@abstractmethod\b|@abstractclassmethod\b|@abstractproperty\b"),
                       _rx(r"\bclass\s+\w+\s*\(\s*[A-Za-z_.]*ABC[A-Za-z_.]*\s*[,)]")],
        "static":     [_rx(r"@staticmethod\b|@classmethod\b")],
    },
    "js": {
        "async":          [_rx(r"\basync\s+(?:function|\([^)]*\)|[A-Za-z_$][\w$]*\s*=)")],
        "generator":      [_rx(r"\bfunction\s*\*"), _rx(r"^\s*\*\s*\w+\s*\("), _rx(r"\byield\b")],
        "exported":       [_rx(r"\bexport(?:\s+default)?\b")],
        "default-export": [_rx(r"\bexport\s+default\b")],
        "static":         [_rx(r"^\s*static\s+", re.MULTILINE)],
    },
    "ts": {
        "async":          [_rx(r"\basync\s+(?:function|\([^)]*\)|[A-Za-z_$][\w$]*\s*=)")],
        "generator":      [_rx(r"\bfunction\s*\*"), _rx(r"^\s*\*\s*\w+\s*\("), _rx(r"\byield\b")],
        "abstract":       [_rx(r"\babstract\s+class\b"),
                           _rx(r"^\s*(?:public\s+|private\s+|protected\s+)?abstract\s+\w+", re.MULTILINE)],
        "exported":       [_rx(r"\bexport(?:\s+default)?\b")],
        "default-export": [_rx(r"\bexport\s+default\b")],
        "static":         [_rx(r"^\s*static\s+", re.MULTILINE)],
        # Angular: classic decorator-style + new signal-style class members.
        # Match in any order; multiple traits can fire on the same entity
        # (e.g. `= input(` produces both `input` and `signal`).
        "input":          [_rx(r"@Input\b"),
                           _rx(r"=\s*input(?:\.required)?\s*[<(]"),
                           _rx(r"=\s*model(?:\.required)?\s*[<(]")],
        "output":         [_rx(r"@Output\b"),
                           _rx(r"=\s*output\s*[<(]")],
        "signal":         [_rx(r"=\s*signal\s*[<(]"),
                           _rx(r"=\s*input(?:\.required)?\s*[<(]"),
                           _rx(r"=\s*output\s*[<(]"),
                           _rx(r"=\s*model(?:\.required)?\s*[<(]"),
                           _rx(r"=\s*computed\s*[<(]")],
        "computed":       [_rx(r"=\s*computed\s*[<(]")],
        "effect":         [_rx(r"\beffect\s*\(")],
        "viewchild":      [_rx(r"@(?:ViewChild|ViewChildren|ContentChild|ContentChildren)\b"),
                           _rx(r"=\s*(?:viewChild|contentChild)(?:\.required)?\s*[<(]")],
        "injected":       [_rx(r"=\s*inject\s*\(")],
    },
    # Go has no `async` keyword (goroutines are launched via the `go` statement,
    # which doesn't mark the *function* as async). Exported-by-capitalization
    # is handled in the name-based branch of _detect_traits.
    "go": {},
    "rust": {
        "async":      [_rx(r"\basync\s+fn\b"), _rx(r"\bpub\s+async\s+fn\b")],
        # `pub` may sit before any combination of async/unsafe/const/extern
        # before the actual item keyword — handle modifiers in any order.
        "exported":   [_rx(
            r"\bpub(?:\s*\([^)]+\))?\s+"
            r"(?:(?:async|unsafe|const|extern\s*(?:\"[^\"]*\")?)\s+)*"
            r"(?:fn|struct|trait|enum|mod|const|static|unsafe|type|impl|use)\b"
        )],
        "abstract":   [_rx(r"\btrait\s+\w+")],  # Rust traits are the closest analog.
    },
    "java": {
        "abstract":   [_rx(r"\babstract\s+(?:class|interface)\b"),
                       _rx(r"^\s*(?:public|protected|private)?\s*abstract\s+", re.MULTILINE)],
        "static":     [_rx(r"^\s*(?:public|protected|private)?\s*(?:final\s+)?static\s+", re.MULTILINE)],
        "exported":   [_rx(r"^\s*public\s+", re.MULTILINE)],
    },
    "cs": {
        "async":      [_rx(r"\basync\s+(?:[A-Za-z<>\[\],\s.]+\s+)?\w+\s*\(")],
        "abstract":   [_rx(r"\babstract\s+(?:class|interface)\b"),
                       _rx(r"^\s*(?:public|protected|private|internal)?\s*abstract\s+", re.MULTILINE)],
        "static":     [_rx(r"^\s*(?:public|protected|private|internal)?\s*static\s+", re.MULTILINE)],
        "exported":   [_rx(r"^\s*public\s+", re.MULTILINE)],
    },
    "kotlin": {
        "async":      [_rx(r"\bsuspend\s+fun\b")],
        "abstract":   [_rx(r"\babstract\s+(?:class|fun|val|var)\b")],
        # Kotlin defaults to public; explicit `private`/`internal`/`protected`
        # are the negative space. Mark with `exported` only when explicit so
        # we don't blanket-tag every entity.
        "exported":   [_rx(r"^\s*public\s+", re.MULTILINE)],
        "static":     [_rx(r"\b@JvmStatic\b")],
    },
    "swift": {
        "async":      [_rx(r"\basync\s+func\b"),
                       _rx(r"\bfunc\s+\w+\s*\([^)]*\)\s*(?:throws\s+)?async\b")],
        "static":     [_rx(r"\bstatic\s+(?:func|var|let)\b"), _rx(r"\bclass\s+(?:func|var)\b")],
        "exported":   [_rx(r"^\s*(?:public|open)\s+", re.MULTILINE)],
    },
    "scala": {
        "abstract":   [_rx(r"\babstract\s+class\b"), _rx(r"^\s*trait\s+", re.MULTILINE)],
        "exported":   [_rx(r"^\s*(?:public)?\s*(?:def|val|var|class|object|trait)\s+(?!_)\w", re.MULTILINE)],
    },
    "ruby": {
        "generator":  [_rx(r"\byield\b")],
        # Ruby has `private`/`public`/`protected` keywords that flip context.
        # Heuristic stays narrow to avoid noise.
    },
    "php": {
        "generator":  [_rx(r"\byield\b")],
        "abstract":   [_rx(r"\babstract\s+(?:class|function)\b")],
        "static":     [_rx(r"\bstatic\s+function\b")],
        "exported":   [_rx(r"\bpublic\s+function\b")],
    },
    "cpp": {
        "abstract":   [_rx(r"=\s*0\s*;")],  # pure virtual
        "static":     [_rx(r"^\s*static\s+", re.MULTILINE)],
        "exported":   [_rx(r"^\s*export\s+", re.MULTILINE)],  # C++20 modules
    },
    "c": {
        "static":     [_rx(r"^\s*static\s+", re.MULTILINE)],
        "exported":   [_rx(r"^\s*extern\s+", re.MULTILINE)],
    },
    "lua": {
        # Lua's local/global is the closest analog to private/public.
        "exported":   [_rx(r"^\s*function\s+\w+[.:]\w+", re.MULTILINE)],  # M.foo / M:foo
    },
    "dart": {
        "async":      [_rx(r"\basync\s*(?:\*)?\s*\{"), _rx(r"\)\s*async\s*(?:\*)?\s*\{")],
        "generator":  [_rx(r"\)\s*async\*\s*\{"), _rx(r"\)\s*sync\*\s*\{")],
        "abstract":   [_rx(r"\babstract\s+class\b")],
        "static":     [_rx(r"^\s*static\s+", re.MULTILINE)],
        # Dart: identifiers starting with `_` are library-private; everything
        # else is exported. Handled in the name-based branch if we extend it.
    },
}


# Universal markers that fire across all language comment dialects.
_TRAIT_PATTERNS_UNIVERSAL: list[tuple[str, re.Pattern]] = [
    ("deprecated", _rx(
        r"@deprecated\b|@Deprecated\b|\bDEPRECATED\b|"
        r"#\[deprecated\b|"        # Rust
        r"\[Obsolete\b|"           # C#
        r"@available\([^)]*deprecated", # Swift
        re.IGNORECASE,
    )),
]


class CodeGraph(GraphAnalysisMixin, GraphTextMixin):
    """Knowledge Graph of a codebase, persisted in SQLite + FAISS.

    Methods are split across mixins to keep this file under ~1.5 k LOC:
      • analysis tools (clone detection, dead code, test coverage,
        repo map, trait filter, FAISS-similar) live in
        :mod:`.graph_analysis`.
      • text-level search (FTS5 trigram regex, ast-grep structural)
        lives in :mod:`.graph_text`.
    """

    def __init__(
        self,
        project_root: str | Path,
        graph_dir: Path,
        llm_extractor: Optional[LLMExtractor] = None,
        project_config: Optional["ProjectConfig"] = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.graph_dir = Path(graph_dir)
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.graph_dir / "graph.db"
        self.llm_extractor: LLMExtractor = llm_extractor or NoOpExtractor()
        self._init_db()
        self.faiss_index = None
        self.faiss_names: list[str] = []
        self._is_building = False
        # Set when reindex_file runs without an immediate FAISS rebuild
        # (e.g. from the file-system watcher). Similarity tools check this
        # and trigger a rebuild before serving stale data.
        self._faiss_dirty = False

        # Per-project overrides via .mcp-rag.toml (auto-loaded from the
        # project root if not passed in).
        if project_config is None:
            from ..config import ProjectConfig as _PC
            project_config = _PC.load(self.project_root)
        self._project_config = project_config
        self._extra_ignore_dirs: set[str] = set(project_config.extra_ignore_dirs or [])
        self._extra_extensions: list[str] = list(project_config.extra_extensions or [])
        self._max_file_bytes: int = project_config.max_file_bytes or _MAX_FILE_BYTES

        self._gitignore_parser = None
        gitignore_path = self.project_root / ".gitignore"
        if gitignore_path.exists():
            try:
                self._gitignore_parser = gitignore_parser.parse_gitignore(gitignore_path)
            except Exception:
                pass

    @property
    def is_building(self) -> bool:
        return self._is_building

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS entities (
                    id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    file    TEXT NOT NULL,
                    name    TEXT NOT NULL,
                    type    TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    UNIQUE(file, name, type)
                );
                CREATE TABLE IF NOT EXISTS relations (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    file        TEXT NOT NULL,
                    from_name   TEXT NOT NULL,
                    relation    TEXT NOT NULL,
                    to_name     TEXT NOT NULL,
                    UNIQUE(file, from_name, relation, to_name)
                );
                CREATE TABLE IF NOT EXISTS file_meta (
                    file    TEXT PRIMARY KEY,
                    mtime   REAL NOT NULL,
                    indexed INTEGER DEFAULT 1
                );
                CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
                CREATE INDEX IF NOT EXISTS idx_relations_from ON relations(from_name);
                CREATE INDEX IF NOT EXISTS idx_relations_to ON relations(to_name);
                CREATE INDEX IF NOT EXISTS idx_entities_file ON entities(file);
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    hash    TEXT NOT NULL,
                    model   TEXT NOT NULL,
                    dim     INTEGER NOT NULL,
                    vec     BLOB NOT NULL,
                    PRIMARY KEY (hash, model)
                );
            """)
            columns = {row[1] for row in con.execute("PRAGMA table_info(entities)").fetchall()}
            if "line_start" not in columns:
                con.execute("ALTER TABLE entities ADD COLUMN line_start INTEGER")
            if "line_end" not in columns:
                con.execute("ALTER TABLE entities ADD COLUMN line_end INTEGER")
            if "snippet" not in columns:
                con.execute("ALTER TABLE entities ADD COLUMN snippet TEXT DEFAULT ''")
            if "traits" not in columns:
                con.execute("ALTER TABLE entities ADD COLUMN traits TEXT DEFAULT ''")
                con.execute("CREATE INDEX IF NOT EXISTS idx_entities_traits ON entities(traits)")
            # Run trait back-fill independently of the ALTER above so the
            # upgrade also lights up for users whose prior session added
            # the column but didn't have the back-fill code yet.
            has_any = con.execute(
                "SELECT 1 FROM entities WHERE COALESCE(traits, '') != '' LIMIT 1"
            ).fetchone()
            if has_any is None:
                rows = con.execute(
                    "SELECT rowid, file, name, type, snippet FROM entities "
                    "WHERE COALESCE(snippet, '') != ''"
                ).fetchall()
                updated = 0
                for rid, f, n, t, sn in rows:
                    traits = self._detect_traits(n or "", sn or "", f or "", t or "")
                    if traits:
                        con.execute("UPDATE entities SET traits = ? WHERE rowid = ?", (traits, rid))
                        updated += 1
                if updated:
                    con.commit()
                    logger.info("Back-filled traits for %d existing entities.", updated)

    def _should_ignore(self, path: Path) -> bool:
        if self._gitignore_parser is not None:
            try:
                if self._gitignore_parser(str(path)):
                    return True
            except Exception:
                pass
        # Exact-match against path components — substring check would filter
        # files like Layout.tsx/Login.tsx because their parent dir lowercase
        # contains "out"/"log".
        parts = set(path.parts)
        if parts & _IGNORE_DIRS:
            return True
        if self._extra_ignore_dirs and (parts & self._extra_ignore_dirs):
            return True
        return False

    def _should_ignore_dir(self, name: str) -> bool:
        if name in _IGNORE_DIRS:
            return True
        return name in self._extra_ignore_dirs

    def _get_files(self) -> list[Path]:
        seen: set[str] = set()
        files: list[Path] = []
        suffixes = {ext.lstrip("*").lower() for ext in (_CODE_EXTENSIONS + self._extra_extensions)}

        for dirpath, dirnames, filenames in os.walk(self.project_root):
            dirnames[:] = [d for d in dirnames if not self._should_ignore_dir(d)]

            for fname in filenames:
                fname_lower = fname.lower()
                if not any(fname_lower.endswith(suffix) for suffix in suffixes):
                    continue

                fpath = os.path.join(dirpath, fname)
                if fpath in seen:
                    continue

                path = Path(fpath)
                if not path.is_file() or self._should_ignore(path):
                    continue

                seen.add(fpath)
                files.append(path)

        return files

    def _get_files_indexed(self) -> list[str]:
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute("SELECT file FROM file_meta").fetchall()
        return [r[0] for r in rows]

    def _file_needs_update(self, filepath: Path) -> bool:
        rel = filepath.relative_to(self.project_root).as_posix()
        mtime = filepath.stat().st_mtime
        with sqlite3.connect(self.db_path) as con:
            row = con.execute(
                "SELECT mtime FROM file_meta WHERE file = ?", (rel,)
            ).fetchone()
        return row is None or row[0] != mtime

    def _delete_file_data(self, rel_path: str) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute("DELETE FROM entities WHERE file = ?", (rel_path,))
            con.execute("DELETE FROM relations WHERE file = ?", (rel_path,))
            con.execute("DELETE FROM file_meta WHERE file = ?", (rel_path,))

    def _cleanup_deleted_files(self, existing_files: list[Path]) -> int:
        existing_rel_paths = {p.relative_to(self.project_root).as_posix() for p in existing_files}
        indexed_rel_paths = self._get_files_indexed()
        deleted_rel_paths = sorted(set(indexed_rel_paths) - existing_rel_paths)
        for rel_path in deleted_rel_paths:
            self._delete_file_data(rel_path)
        return len(deleted_rel_paths)

    @staticmethod
    def _normalize_whitespace(text: str, limit: int = 400) -> str:
        cleaned = " ".join((text or "").strip().split())
        return cleaned[:limit]

    def _enrich_entity_locations(self, rel_path: str, entities: list[dict]) -> list[dict]:
        enriched = []
        for entity in entities:
            name = self._normalize_whitespace(str(entity.get("name", "")), limit=160)
            if not name:
                continue
            entity_type = self._normalize_whitespace(str(entity.get("type", "")), limit=40).lower() or "symbol"
            if entity_type not in _ALLOWED_ENTITY_TYPES:
                entity_type = "symbol"
            description = self._normalize_whitespace(str(entity.get("description", "")), limit=300)
            snippet_info = self._find_entity_snippet(rel_path, name)
            enriched.append({
                "name": name,
                "type": entity_type,
                "description": description,
                "line_start": snippet_info["line_start"],
                "line_end": snippet_info["line_end"],
                "snippet": snippet_info["snippet"],
            })
        return enriched

    def _sanitize_relations(self, rel_path: str, relations: list[dict], entity_names: set[str]) -> list[dict]:
        sanitized = []
        for relation in relations:
            from_name = self._normalize_whitespace(str(relation.get("from", "")), limit=160)
            to_name = self._normalize_whitespace(str(relation.get("to", "")), limit=160)
            rel_type = self._normalize_whitespace(str(relation.get("relation", "")), limit=40).lower()
            if not from_name or not to_name or rel_type not in _ALLOWED_RELATION_TYPES:
                continue
            if entity_names and from_name not in entity_names and to_name not in entity_names:
                continue
            sanitized.append({"from": from_name, "relation": rel_type, "to": to_name})
        return sanitized

    def _sanitize_extracted(self, rel_path: str, data: dict) -> dict:
        raw_entities = data.get("entities", []) if isinstance(data, dict) else []
        raw_relations = data.get("relations", []) if isinstance(data, dict) else []
        entities = self._enrich_entity_locations(rel_path, [e for e in raw_entities if isinstance(e, dict)])
        entity_names = {entity["name"] for entity in entities}
        # The file's own module-entity (its relpath) is an implicit, always-valid
        # endpoint: RAG-builder emits module-level edges (imports/calls) with
        # from_name = rel_path. Without seeding it here, _sanitize_relations would
        # drop every such edge (neither endpoint in the batch's entity set),
        # leaving files as "N entities, 0 relations".
        entity_names.add(rel_path)
        relations = self._sanitize_relations(rel_path, [r for r in raw_relations if isinstance(r, dict)], entity_names)
        return {"entities": entities, "relations": relations}

    @staticmethod
    def _extractor_strategy(filepath: Path) -> str:
        suffix = filepath.suffix.lower()
        if suffix in {".css", ".scss", ".sass", ".less"}:
            return "stylesheet"
        if suffix in {".json", ".yaml", ".yml", ".toml"}:
            return "config"
        if suffix in {".html", ".htm", ".vue", ".svelte", ".astro"}:
            return "template"
        if suffix in _TREE_SITTER_LANGUAGES:
            return "tree_sitter"
        return "llm"

    @staticmethod
    def _make_file_entity(rel_path: str, entity_type: str, description: str) -> dict:
        return {"name": rel_path, "type": entity_type, "description": description}

    def _extract_stylesheet(self, rel_path: str, code: str) -> dict:
        entities = [self._make_file_entity(rel_path, "style", "Stylesheet file")]
        relations: list[dict] = []
        seen_entities = {rel_path}

        for selector_block in re.findall(r"([^{}]+)\{", code):
            selector_block = selector_block.strip()
            if not selector_block or selector_block.startswith("@"):
                continue
            for selector in selector_block.split(","):
                selector = self._normalize_whitespace(selector, limit=160)
                if not selector or selector in seen_entities:
                    continue
                seen_entities.add(selector)
                entities.append({"name": selector, "type": "selector", "description": "Stylesheet selector"})
                relations.append({"from": rel_path, "relation": "defines", "to": selector})

        for var_name in re.findall(r"(--[A-Za-z0-9_-]+)\s*:", code):
            var_name = self._normalize_whitespace(var_name, limit=160)
            if not var_name or var_name in seen_entities:
                continue
            seen_entities.add(var_name)
            entities.append({"name": var_name, "type": "variable", "description": "CSS custom property"})
            relations.append({"from": rel_path, "relation": "defines", "to": var_name})

        for import_target in re.findall(r"@import\s+(?:url\()?['\"]([^'\"]+)['\"]", code):
            import_target = self._normalize_whitespace(import_target, limit=160)
            if not import_target:
                continue
            if import_target not in seen_entities:
                seen_entities.add(import_target)
                entities.append({"name": import_target, "type": "import", "description": "Stylesheet import"})
            relations.append({"from": rel_path, "relation": "imports", "to": import_target})

        return {"entities": entities, "relations": relations}

    def _extract_config(self, rel_path: str, code: str, suffix: str) -> dict:
        entities = [self._make_file_entity(rel_path, "config", "Configuration file")]
        relations: list[dict] = []
        seen_entities = {rel_path}

        def _add_key(key: str) -> None:
            key = self._normalize_whitespace(key, limit=160)
            if not key or key in seen_entities:
                return
            seen_entities.add(key)
            entities.append({"name": key, "type": "variable", "description": "Configuration key"})
            relations.append({"from": rel_path, "relation": "defines", "to": key})

        try:
            if suffix == ".toml":
                parsed = tomllib.loads(code)
                if isinstance(parsed, dict):
                    for key in parsed.keys():
                        _add_key(str(key))
                return {"entities": entities, "relations": relations}

            if suffix == ".json":
                parsed = json5.loads(code)
                if isinstance(parsed, dict):
                    for key in parsed.keys():
                        _add_key(str(key))
                    return {"entities": entities, "relations": relations}
        except Exception:
            pass

        for key in re.findall(r'^[ \t]*["\']?([A-Za-z0-9_.-]+)["\']?\s*[:=]', code, flags=re.MULTILINE):
            _add_key(key)

        return {"entities": entities, "relations": relations}

    def _extract_template(self, rel_path: str, code: str) -> dict:
        entities = [self._make_file_entity(rel_path, "template", "Template file")]
        relations: list[dict] = []
        seen_entities = {rel_path}
        seen_relations: set[tuple[str, str, str]] = set()

        def _add_entity(name: str, entity_type: str, description: str) -> None:
            name = self._normalize_whitespace(name, limit=160)
            if not name or name in seen_entities:
                return
            seen_entities.add(name)
            entities.append({"name": name, "type": entity_type, "description": description})
            key = (rel_path, "defines", name)
            if key not in seen_relations:
                seen_relations.add(key)
                relations.append({"from": rel_path, "relation": "defines", "to": name})

        def _add_ref(relation: str, name: str) -> None:
            name = self._normalize_whitespace(name, limit=160)
            if not name:
                return
            # Skip microsyntax tokens / Angular implicits only for the bare-ident
            # ('uses') scan — if the source actually writes `count()` with parens
            # that's clearly a method/signal call and must be recorded even
            # when the bare word `count` is a *ngFor implicit local.
            if relation == "uses" and name in _TEMPLATE_EXPR_KEYWORDS:
                return
            # Reserved JS literals / block-trigger keywords never refer to a
            # user-defined entity in either context.
            if name in _TEMPLATE_HARD_KEYWORDS:
                return
            # If we already recorded a 'calls' edge to this name, skip the
            # weaker 'uses' duplicate — dead-code treats both as live anyway.
            if relation == "uses" and (rel_path, "calls", name) in seen_relations:
                return
            key = (rel_path, relation, name)
            if key in seen_relations:
                return
            seen_relations.add(key)
            relations.append({"from": rel_path, "relation": relation, "to": name})

        def _scan_expression(expr: str) -> None:
            for m in _TEMPLATE_METHOD_CALL_RE.finditer(expr):
                _add_ref("calls", m.group(1))
            for m in _TEMPLATE_IDENT_RE.finditer(expr):
                _add_ref("uses", m.group(1))

        for tag in re.findall(r"<([A-Za-z][A-Za-z0-9:_-]*)", code):
            entity_type = "component" if ("-" in tag or ":" in tag) else "template"
            tag_norm = self._normalize_whitespace(tag, limit=160)
            if not tag_norm:
                continue
            if tag_norm not in seen_entities:
                seen_entities.add(tag_norm)
                entities.append({"name": tag_norm, "type": entity_type, "description": "Template tag"})
            # Component tags are USED by this template (not defined here);
            # the defining edge comes from the @Component class on the .ts side.
            relation = "uses" if entity_type == "component" else "defines"
            key = (rel_path, relation, tag_norm)
            if key not in seen_relations:
                seen_relations.add(key)
                relations.append({"from": rel_path, "relation": relation, "to": tag_norm})

        for class_block in re.findall(r'class(?:Name)?\s*=\s*["\']([^"\']+)["\']', code):
            for cls in re.split(r"\s+", class_block.strip()):
                if cls:
                    _add_entity(f".{cls}", "selector", "Template class selector")

        for item_id in re.findall(r'id\s*=\s*["\']([^"\']+)["\']', code):
            _add_entity(f"#{item_id}", "selector", "Template id selector")

        # Angular bindings → method/property edges so dead-code stops flagging
        # template-driven methods as orphans.
        #   (click)="save()"  [disabled]="isLoading"  *ngIf="hasAny(roles)"
        #   [(ngModel)]="value"  @if (hasAny(['admin'])) {  @for (u of users(); ...) {
        #   @let total = sum(items);  {{ user.fullName() }}
        for m in _BINDING_ATTR_RE.finditer(code):
            _scan_expression(m.group(1))
        for m in _INTERPOLATION_RE.finditer(code):
            _scan_expression(m.group(1))
        for m in _CONTROL_FLOW_RE.finditer(code):
            _scan_expression(m.group(1))
        for m in _LET_DECL_RE.finditer(code):
            _scan_expression(m.group(1))

        return {"entities": entities, "relations": relations}

    def _extract_structured(self, filepath: Path, code: str) -> dict:
        rel_path = filepath.relative_to(self.project_root).as_posix()
        strategy = self._extractor_strategy(filepath)
        if strategy == "stylesheet":
            return self._extract_stylesheet(rel_path, code)
        if strategy == "config":
            return self._extract_config(rel_path, code, filepath.suffix.lower())
        if strategy == "template":
            return self._extract_template(rel_path, code)
        return {"entities": [], "relations": []}

    @staticmethod
    def _infer_tree_sitter_entity_type(node_type: str, name: str, rel_path: str, suffix: str) -> str:
        node_type = (node_type or "").lower()
        is_jsx_file = suffix in {".tsx", ".jsx", ".vue", ".svelte", ".astro"}
        is_ts_or_jsx = is_jsx_file or suffix in {".ts", ".js"}
        is_pascal = bool(name) and name[:1].isupper()
        # React custom hooks: useXxx convention. Detect in any TS/JS-family
        # file (React hooks frequently live in plain .ts/.js too).
        is_hook_name = (
            len(name) > 3 and name.startswith("use") and name[3:4].isupper()
        )
        if is_ts_or_jsx and is_hook_name and ("function" in node_type or "variable" in node_type):
            return "hook"
        # In JSX/TSX files, PascalCase function or class declarations are
        # React components — classify them as such so entity_type='component'
        # filters work the way users expect (Button, Modal, AntdTable, …).
        if is_jsx_file and is_pascal and ("class" in node_type or "function" in node_type):
            return "component"
        if "class" in node_type:
            return "class"
        if "method" in node_type:
            return "method"
        if "function" in node_type:
            return "function"
        if "interface" in node_type:
            return "interface"
        if "enum" in node_type:
            return "enum"
        if "type_alias" in node_type or node_type == "type":
            return "type"
        if "import" in node_type:
            return "import"
        if node_type in {"element", "script_element"}:
            return "component" if ("-" in name or ":" in name) else "template"
        if node_type in {"style_element", "stylesheet", "rule_set", "qualified_rule"}:
            return "style"
        if node_type in {"class_selector", "id_selector"}:
            return "selector"
        if suffix in {".tsx", ".jsx", ".vue", ".svelte", ".astro"} and name[:1].isupper():
            return "component"
        if suffix in {".ts", ".tsx", ".js", ".jsx"}:
            return "variable"
        return "symbol"

    def _extract_import_targets(self, code: str) -> list[str]:
        targets: list[str] = []
        patterns = [
            r'^\s*import\s+.+?\s+from\s+["\']([^"\']+)["\']',
            r'^\s*import\s+["\']([^"\']+)["\']',
            r'^\s*from\s+([A-Za-z0-9_./-]+)\s+import\s+',
            r'^\s*require\(\s*["\']([^"\']+)["\']\s*\)',
        ]
        for pattern in patterns:
            for target in re.findall(pattern, code, flags=re.MULTILINE):
                normalized = self._normalize_whitespace(target, limit=160)
                if normalized and normalized not in targets:
                    targets.append(normalized)
        return targets

    @staticmethod
    def _infer_symbol_entity_type(symbol: str) -> str:
        if not symbol:
            return "symbol"
        if symbol[:1].isupper():
            return "class"
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", symbol) and symbol[:1].islower():
            return "function"
        return "property"

    def _extract_symbol_relations(
        self,
        rel_path: str,
        code: str,
        entities: list[dict],
        relations: list[dict],
        seen_entities: set[str],
        scopes: list[dict],
    ) -> None:
        lines = code.splitlines()
        excluded_calls = {
            "if", "for", "while", "switch", "catch", "return", "typeof", "await",
            "new", "super", "function", "def", "class", "elif", "with", "print",
        }
        suffix = Path(rel_path).suffix.lower()
        is_jsx = suffix in {".tsx", ".jsx", ".vue", ".svelte", ".astro"}
        # Calls whose first capitalized argument is the *real* dependency
        # we want to track. Angular DI hides class usages behind these
        # factories, so a plain "calls inject" relation is useless on its
        # own — we also emit "uses ServiceClass" to make
        # graph_find_usages('ServiceClass') see the dep.
        di_factories = {"inject", "TestBed.inject"}

        def scope_for_line(line_number: int) -> str:
            best_name = rel_path
            best_size: Optional[int] = None
            for scope in scopes:
                if scope["line_start"] <= line_number <= scope["line_end"]:
                    size = scope["line_end"] - scope["line_start"]
                    if best_size is None or size < best_size:
                        best_size = size
                        best_name = scope["name"]
            return best_name

        seen_relations = {(r.get("from", ""), r.get("relation", ""), r.get("to", "")) for r in relations}

        for idx, line in enumerate(lines, start=1):
            owner = scope_for_line(idx)

            for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_.$]*)\s*\(", line):
                raw_name = match.group(1).strip()
                if not raw_name:
                    continue
                symbol = raw_name.split(".")[-1]
                if symbol.lower() in excluded_calls:
                    continue
                if symbol not in seen_entities:
                    seen_entities.add(symbol)
                    entities.append({
                        "name": symbol,
                        "type": self._infer_symbol_entity_type(symbol),
                        "description": "Referenced call target",
                    })
                rel_key = (owner, "calls", symbol)
                if rel_key not in seen_relations:
                    seen_relations.add(rel_key)
                    relations.append({"from": owner, "relation": "calls", "to": symbol})

                # DI factory: read the first PascalCase identifier from the
                # arg list and emit a `uses` edge to it.
                if symbol in di_factories or raw_name in di_factories:
                    rest = line[match.end():]
                    arg_match = re.match(r"\s*([A-Z][A-Za-z0-9_]*)", rest)
                    if arg_match:
                        service = arg_match.group(1)
                        if service not in seen_entities:
                            seen_entities.add(service)
                            entities.append({
                                "name": service,
                                "type": self._infer_symbol_entity_type(service),
                                "description": f"Injected via {symbol}()",
                            })
                        di_key = (owner, "uses", service)
                        if di_key not in seen_relations:
                            seen_relations.add(di_key)
                            relations.append({"from": owner, "relation": "uses", "to": service})

            # JSX component usage: `<Alert />`, `<Layout.Sider>`, `<MyMenu prop=…>`.
            # Function calls (`Name(`) above already capture HOC/render-fn forms;
            # this block fills the gap for declarative JSX, which Tree-sitter
            # node types in our list don't surface.
            if is_jsx:
                for match in re.finditer(r"<\s*([A-Z][A-Za-z0-9_]*(?:\.[A-Z][A-Za-z0-9_]*)?)\b", line):
                    raw = match.group(1).strip()
                    if not raw:
                        continue
                    component = raw.split(".")[0]
                    if component not in seen_entities:
                        seen_entities.add(component)
                        entities.append({
                            "name": component,
                            "type": "component",
                            "description": "JSX component reference",
                        })
                    rel_key = (owner, "instantiates", component)
                    if rel_key not in seen_relations:
                        seen_relations.add(rel_key)
                        relations.append({"from": owner, "relation": "instantiates", "to": component})

            for match in re.finditer(r"(?:\.|->)([A-Za-z_][A-Za-z0-9_]*)", line):
                symbol = match.group(1).strip()
                if not symbol:
                    continue
                if symbol not in seen_entities:
                    seen_entities.add(symbol)
                    entities.append({
                        "name": symbol,
                        "type": "property",
                        "description": "Referenced property/member",
                    })
                rel_key = (owner, "uses", symbol)
                if rel_key not in seen_relations:
                    seen_relations.add(rel_key)
                    relations.append({"from": owner, "relation": "uses", "to": symbol})

    @staticmethod
    def _angular_decorator_overrides(code: str, suffix: str) -> dict[str, str]:
        """Map ``ClassName -> refined_type`` for Angular/NestJS-decorated classes.

        Tree-sitter's class_declaration node type loses the decorator context,
        so a quick regex pre-scan is the cheapest way to recover it. Each
        ``@Decorator`` is followed (after some chars of metadata) by the next
        ``class Foo`` declaration — that pairing is what we capture.
        """
        if suffix not in {".ts", ".js"}:
            return {}
        decorator_to_type = {
            # Angular
            "Component": "component",
            "Directive": "directive",
            "Injectable": "service",
            "NgModule": "module",
            "Pipe": "pipe",
            # NestJS (some shared with Angular: Injectable, Pipe)
            "Module": "module",
            "Controller": "controller",
        }
        out: dict[str, str] = {}
        pattern = r"@(" + "|".join(re.escape(k) for k in decorator_to_type) + r")\b"
        for m in re.finditer(pattern, code):
            tail = code[m.end() : m.end() + 4000]
            m2 = re.search(r"\bclass\s+([A-Z][A-Za-z0-9_]*)", tail)
            if m2:
                out[m2.group(1)] = decorator_to_type[m.group(1)]
        return out

    @staticmethod
    def _ts_constructor_dependencies(code: str, suffix: str) -> list[tuple[str, str]]:
        """Return [(class_name, dep_type), ...] from ``constructor(...)`` params.

        Catches Angular and NestJS DI which both use the same shape:
            constructor(private foo: FooService, public bar: BarService) {}
        Only PascalCase types are captured (TS primitives like ``string``
        are skipped — they don't correspond to graph entities).
        """
        if suffix not in {".ts", ".js"}:
            return []
        out: list[tuple[str, str]] = []
        for m_class in re.finditer(r"\bclass\s+([A-Z][A-Za-z0-9_]*)", code):
            cls_name = m_class.group(1)
            # Limit search window to before the next class declaration.
            next_class = re.search(r"\bclass\s+[A-Z]", code[m_class.end():])
            window_end = m_class.end() + (next_class.start() if next_class else len(code) - m_class.end())
            m_ctor = re.search(r"\bconstructor\s*\(", code[m_class.end():window_end])
            if not m_ctor:
                continue
            # Match the closing paren for the constructor parameter list.
            start = m_class.end() + m_ctor.end()
            depth = 1
            i = start
            while i < window_end and i < len(code) and depth > 0:
                c = code[i]
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                i += 1
            if depth != 0:
                continue
            params = code[start : i - 1]
            for m_p in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_]*\s*:\s*([A-Z][A-Za-z0-9_]*)", params):
                out.append((cls_name, m_p.group(1)))
        return out

    @staticmethod
    def _angular_component_metadata(code: str, suffix: str) -> list[dict]:
        """Return template-related links emitted by ``@Component({...})``.

        For each decorator block we extract:
          * ``selector`` string  → defines edge: ``Class defines 'app-foo'``
            (template extractor emits matching ``uses`` edges from .html files)
          * ``templateUrl`` path → uses edge:  ``Class uses ./foo.html``
          * each ``styleUrls`` path → uses edge: ``Class uses ./foo.scss``

        Each entry has shape ``{"class": str, "kind": "selector"|"template"|"style", "value": str}``.
        """
        if suffix not in {".ts", ".js"}:
            return []
        out: list[dict] = []
        for m in re.finditer(r"@(Component|Directive)\s*\(\s*\{", code):
            start = m.end()
            depth = 1
            i = start
            while i < len(code) and depth > 0:
                c = code[i]
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                i += 1
            if depth != 0:
                continue
            body = code[start : i - 1]
            rest = code[i : i + 4000]
            m_cls = re.search(r"\bclass\s+([A-Z][A-Za-z0-9_]*)", rest)
            if not m_cls:
                continue
            owner = m_cls.group(1)
            m_sel = re.search(r"selector\s*:\s*['\"]([^'\"]+)['\"]", body)
            if m_sel:
                out.append({"class": owner, "kind": "selector", "value": m_sel.group(1).strip()})
            m_tpl = re.search(r"templateUrl\s*:\s*['\"]([^'\"]+)['\"]", body)
            if m_tpl:
                out.append({"class": owner, "kind": "template", "value": m_tpl.group(1).strip()})
            m_styles = re.search(r"styleUrls?\s*:\s*\[([^\]]*)\]", body)
            if m_styles:
                for m_p in re.finditer(r"['\"]([^'\"]+)['\"]", m_styles.group(1)):
                    out.append({"class": owner, "kind": "style", "value": m_p.group(1).strip()})
        return out

    @staticmethod
    def _module_metadata_dependencies(code: str, suffix: str) -> list[tuple[str, str]]:
        """Return [(module_class, identifier), ...] from ``@NgModule``/``@Module`` arrays.

        Looks at the declarations/imports/providers/bootstrap/controllers/
        exports fields — anything in those arrays that looks like a class.
        """
        if suffix not in {".ts", ".js"}:
            return []
        out: list[tuple[str, str]] = []
        for m in re.finditer(r"@(NgModule|Module)\s*\(\s*\{", code):
            start = m.end()
            depth = 1
            i = start
            while i < len(code) and depth > 0:
                c = code[i]
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                i += 1
            if depth != 0:
                continue
            body = code[start : i - 1]
            rest = code[i : i + 4000]
            m_cls = re.search(r"\bclass\s+([A-Z][A-Za-z0-9_]*)", rest)
            if not m_cls:
                continue
            owner = m_cls.group(1)
            for m_arr in re.finditer(
                r"\b(declarations|imports|providers|bootstrap|controllers|exports|entryComponents)\s*:\s*\[([^\]]*)\]",
                body,
            ):
                for m_id in re.finditer(r"\b[A-Z][A-Za-z0-9_]*\b", m_arr.group(2)):
                    out.append((owner, m_id.group(0)))
        return out

    @staticmethod
    def _angular_runtime_references(code: str, rel_path: str, suffix: str) -> list[tuple[str, str]]:
        """Identifiers wired by Angular runtime APIs the call-graph would otherwise miss.

        Covers the four ways modern Angular apps reference services / guards /
        components without an explicit call site:

          * Standalone bootstrap helpers:  ``provideRouter(routes)``,
            ``provideHttpClient(withInterceptors([jwt, error]))``,
            ``provideAnimations()`` — the inner identifiers are wired by the DI
            container, never called directly.
          * Route guard / resolver arrays: ``canActivate: [authGuard]``,
            ``resolve: { user: UserResolver }``.
          * Provider config objects: ``{ provide: X, useFactory: initAuth }``
            / ``useClass: FooImpl`` / ``useExisting: BarToken``.
          * Route component refs: ``{ path: 'home', component: HomeComponent }``.

        Returns ``(from_entity, dep_name)`` tuples — ``from_entity`` is the file
        path because these references usually live at module scope, not inside
        a class.
        """
        if suffix not in {".ts", ".js", ".mts", ".cts", ".mjs", ".cjs"}:
            return []
        out: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()

        def _emit(dep: str) -> None:
            if not dep or dep[:1].isdigit():
                return
            if dep in _ANGULAR_RUNTIME_KEYWORDS:
                return
            key = (rel_path, dep)
            if key in seen:
                return
            seen.add(key)
            out.append(key)

        # provideXxx(...) / withXxx(...) — standalone API surface.
        for m in re.finditer(r'\b(?:provide|with)[A-Z]\w*\s*\(([^)]*)\)', code):
            for m_id in re.finditer(r'\b([A-Za-z_]\w*)\b', m.group(1)):
                _emit(m_id.group(1))

        # Route guard / resolver arrays.
        for m in re.finditer(
            r'\b(?:canActivate|canActivateChild|canDeactivate|canLoad|canMatch)'
            r'\s*:\s*\[([^\]]*)\]',
            code,
        ):
            for m_id in re.finditer(r'\b([A-Za-z_]\w*)\b', m.group(1)):
                _emit(m_id.group(1))

        # resolve: { key: Resolver, ... }  — object form, distinct from the
        # method named 'resolve' on a Resolver class.
        for m in re.finditer(r'\bresolve\s*:\s*\{([^}]*)\}', code):
            for m_id in re.finditer(r':\s*([A-Z]\w*)', m.group(1)):
                _emit(m_id.group(1))

        # Provider object forms.
        for m in re.finditer(r'\buse(?:Factory|Class|Existing)\s*:\s*([A-Za-z_]\w*)', code):
            _emit(m.group(1))

        # Route component refs.
        for m in re.finditer(r'\bcomponent\s*:\s*([A-Z]\w*)', code):
            _emit(m.group(1))

        # Lazy-loaded routes: loadComponent: () => import('...').then(m => m.FooComponent)
        for m in re.finditer(r'\bm\s*\.\s*([A-Z]\w*)\s*\)', code):
            _emit(m.group(1))

        return out

    def _extract_with_tree_sitter(self, filepath: Path, code: str) -> dict:
        parser = self._get_tree_sitter_parser(filepath)
        rel_path = filepath.relative_to(self.project_root).as_posix()
        if parser is None:
            return {"entities": [], "relations": []}
        try:
            source = code.encode("utf-8", errors="ignore")
            tree = parser.parse(source)
        except Exception:
            return {"entities": [], "relations": []}

        suffix = filepath.suffix.lower()
        ng_overrides = self._angular_decorator_overrides(code, suffix)
        entities = [self._make_file_entity(rel_path, "module", "Source file")]
        relations: list[dict] = []
        seen_entities = {rel_path}
        scopes: list[dict] = []

        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            if node.type in _TREE_SITTER_NODE_TYPES:
                name = self._extract_tree_sitter_name(node, source)
                name = self._normalize_whitespace(name, limit=160)
                if name and name not in seen_entities:
                    seen_entities.add(name)
                    entity_type = self._infer_tree_sitter_entity_type(node.type, name, rel_path, suffix)
                    if entity_type == "class" and name in ng_overrides:
                        entity_type = ng_overrides[name]
                    entities.append({
                        "name": name,
                        "type": entity_type,
                        "description": f"Extracted from {node.type}",
                    })
                    relations.append({"from": rel_path, "relation": "defines", "to": name})
                if name:
                    scopes.append({
                        "name": name,
                        "line_start": node.start_point[0] + 1,
                        "line_end": node.end_point[0] + 1,
                    })
            stack.extend(reversed(node.children))

        for target in self._extract_import_targets(code):
            if target not in seen_entities:
                seen_entities.add(target)
                entities.append({"name": target, "type": "import", "description": "Imported module"})
            relations.append({"from": rel_path, "relation": "imports", "to": target})

        self._extract_symbol_relations(rel_path, code, entities, relations, seen_entities, scopes)

        # Angular / NestJS DI — `uses` edges from constructor-injected types
        # and from @NgModule / @Module metadata arrays. Tree-sitter doesn't
        # surface these on its own, but they're the dominant way deps wire
        # up in those frameworks.
        seen_relations = {(r.get("from", ""), r.get("relation", ""), r.get("to", "")) for r in relations}
        for owner, dep in self._ts_constructor_dependencies(code, suffix):
            if dep not in seen_entities:
                seen_entities.add(dep)
                entities.append({
                    "name": dep,
                    "type": self._infer_symbol_entity_type(dep),
                    "description": "Injected via constructor",
                })
            key = (owner, "uses", dep)
            if key not in seen_relations:
                seen_relations.add(key)
                relations.append({"from": owner, "relation": "uses", "to": dep})
        for owner, dep in self._module_metadata_dependencies(code, suffix):
            if dep not in seen_entities:
                seen_entities.add(dep)
                entities.append({
                    "name": dep,
                    "type": self._infer_symbol_entity_type(dep),
                    "description": "Module metadata reference",
                })
            key = (owner, "uses", dep)
            if key not in seen_relations:
                seen_relations.add(key)
                relations.append({"from": owner, "relation": "uses", "to": dep})

        # Standalone provideX / withX / canActivate arrays / useFactory etc.
        # — see _angular_runtime_references for the full list of patterns.
        for owner, dep in self._angular_runtime_references(code, rel_path, suffix):
            if dep not in seen_entities:
                seen_entities.add(dep)
                entities.append({
                    "name": dep,
                    "type": self._infer_symbol_entity_type(dep),
                    "description": "Angular runtime reference",
                })
            key = (owner, "uses", dep)
            if key not in seen_relations:
                seen_relations.add(key)
                relations.append({"from": owner, "relation": "uses", "to": dep})

        # @Component({selector, templateUrl, styleUrls}) — link the class to
        # its template tag, html file, and stylesheet so graph_find_usages
        # of either side surfaces the other.
        parent_dir = Path(rel_path).parent.as_posix()
        for link in self._angular_component_metadata(code, suffix):
            owner = link["class"]
            value = link["value"]
            kind = link["kind"]
            if kind == "selector":
                if value not in seen_entities:
                    seen_entities.add(value)
                    entities.append({
                        "name": value,
                        "type": "selector",
                        "description": "Angular component selector",
                    })
                key = (owner, "defines", value)
                if key not in seen_relations:
                    seen_relations.add(key)
                    relations.append({"from": owner, "relation": "defines", "to": value})
            else:
                # Resolve relative paths against the .ts file's directory so
                # the target matches whatever the .html / .scss extractor stored.
                resolved = value
                if value.startswith("./") or value.startswith("../") or not value.startswith("/"):
                    try:
                        resolved = (Path(parent_dir) / value).as_posix()
                    except Exception:
                        resolved = value
                resolved = resolved.lstrip("./")
                if resolved not in seen_entities:
                    seen_entities.add(resolved)
                    entities.append({
                        "name": resolved,
                        "type": "template" if kind == "template" else "style",
                        "description": f"Angular {kind} reference",
                    })
                key = (owner, "uses", resolved)
                if key not in seen_relations:
                    seen_relations.add(key)
                    relations.append({"from": owner, "relation": "uses", "to": resolved})

        return {"entities": entities, "relations": relations}

    async def _extract_with_strategy(
        self, filepath: Path, code: str, force_llm: bool = False
    ) -> tuple[dict, str]:
        rel_path = filepath.relative_to(self.project_root).as_posix()
        # ``force_llm`` skips tree-sitter / structured extractors and goes
        # directly to the LLM. Used when the caller explicitly wants a
        # semantic extraction (e.g. tree-sitter mis-parses a dialect, or the
        # file is in a language the deterministic extractors don't cover
        # well). Returns whatever the LLM produced — including empty when
        # no LLM is configured (NoOpExtractor) — so the caller sees the
        # outcome instead of silently falling back.
        if force_llm:
            return await self.llm_extractor.extract(rel_path, code), "llm"
        strategy = self._extractor_strategy(filepath)
        if strategy == "tree_sitter":
            extracted = self._extract_with_tree_sitter(filepath, code)
            if extracted.get("entities") or extracted.get("relations"):
                return extracted, strategy
        if strategy != "llm":
            structured = self._extract_structured(filepath, code)
            if structured.get("entities") or structured.get("relations"):
                return structured, strategy
        return await self.llm_extractor.extract(rel_path, code), "llm"

    @staticmethod
    def _detect_traits(name: str, snippet: str, file: str, entity_type: str) -> str:
        """Detect language-aware markers from the entity head.

        Returns a single space-separated lowercase string ('async exported')
        for cheap LIKE-based filtering downstream. Empty string when nothing
        matches. False positives bias ranking; false negatives hide the
        trait — patterns are deliberately conservative.

        Per-language detectors run only when the file extension matches.
        Three orthogonal layers:
          • snippet-based (regex over head[:300]) — needs a non-empty snippet
          • name-based (Go capitalization, Python ``_`` prefix)
          • path-based (test files) — always runs, even on empty-snippet rows
        """
        traits: set[str] = set()
        snip = (snippet or "").strip()
        head = snip[:300]
        f_lower = (file or "").lower().replace("\\", "/")
        basename = f_lower.rsplit("/", 1)[-1]
        suffix = "." + f_lower.rsplit(".", 1)[-1] if "." in basename else ""
        lang = _LANG_FROM_EXT.get(suffix)

        if snip and lang:
            for trait, patterns in _TRAIT_PATTERNS_PER_LANG.get(lang, {}).items():
                if any(p.search(head) or p.search(snip) for p in patterns):
                    traits.add(trait)
            # Universal patterns (deprecated marker variants across langs).
            for trait, pattern in _TRAIT_PATTERNS_UNIVERSAL:
                if pattern.search(snip):
                    traits.add(trait)

        # Name-based heuristics (cheap and useful even without a snippet).
        if name:
            if lang == "go" and name[0].isupper() and entity_type in {"function", "method", "class", "interface", "type"}:
                traits.add("exported")
            elif lang == "python" and not name.startswith("_") and entity_type in {"function", "method", "class"}:
                traits.add("exported")

        # Path-based test trait — language-independent.
        if (
            basename.startswith("test_") or basename.startswith("conftest")
            or basename.endswith("_test.py") or basename.endswith("_test.go")
            or basename.endswith("test.java") or basename.endswith("tests.java")
            or basename.endswith("test.kt") or basename.endswith("tests.kt")
            or basename.endswith("test.scala") or basename.endswith("tests.scala")
            or basename.endswith("test.rb") or basename.endswith("_spec.rb")
            or basename.endswith("test.cs") or basename.endswith("tests.cs")
            or ".test." in basename or ".spec." in basename
            or any(seg in f_lower for seg in ("/tests/", "/test/", "/__tests__/", "/spec/", "/specs/"))
        ):
            traits.add("test")

        return " ".join(sorted(traits))

    def rebackfill_traits(self) -> int:
        """Recompute traits for every existing entity from scratch.

        Useful after upgrading mcp-rag to pick up improved trait
        detection without a full ``graph_clear + graph_build``. Returns
        the number of rows that ended up with a non-empty traits string.
        """
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute(
                "SELECT rowid, file, name, type, COALESCE(snippet, '') FROM entities"
            ).fetchall()
            updated = 0
            for rid, f, n, t, sn in rows:
                traits = self._detect_traits(n or "", sn or "", f or "", t or "")
                con.execute("UPDATE entities SET traits = ? WHERE rowid = ?", (traits, rid))
                if traits:
                    updated += 1
            con.commit()
        logger.info("rebackfill_traits: tagged %d / %d entities.", updated, len(rows))
        return updated

    def _store_extracted(self, rel_path: str, mtime: float, data: dict) -> None:
        with sqlite3.connect(self.db_path) as con:
            for e in data.get("entities", []):
                traits = self._detect_traits(
                    e.get("name", ""), e.get("snippet", ""), rel_path, e.get("type", ""),
                )
                # ON CONFLICT preserves richer description/snippet from an
                # earlier indexing pass when this re-insert carries empty
                # text — see add_entity for the same pattern and rationale.
                con.execute(
                    """INSERT INTO entities(file, name, type, description, line_start, line_end, snippet, traits)
                    VALUES(?,?,?,?,?,?,?,?)
                    ON CONFLICT(file, name, type) DO UPDATE SET
                        description = COALESCE(NULLIF(excluded.description, ''), description),
                        line_start  = COALESCE(excluded.line_start, line_start),
                        line_end    = COALESCE(excluded.line_end, line_end),
                        snippet     = COALESCE(NULLIF(excluded.snippet, ''), snippet),
                        traits      = excluded.traits""",
                    (
                        rel_path,
                        e.get("name", ""),
                        e.get("type", ""),
                        e.get("description", ""),
                        e.get("line_start"),
                        e.get("line_end"),
                        e.get("snippet", ""),
                        traits,
                    ),
                )
                # Auto-emit ``file --defines--> entity`` — the same trivial
                # edge add_entity emits. Without it this write path produced
                # no defines edges at all (callers are told not to pass them),
                # leaving file-structure / repo-map / dead-code analyses blind.
                con.execute(
                    "INSERT OR IGNORE INTO relations(file, from_name, relation, to_name) VALUES(?,?,?,?)",
                    (rel_path, rel_path, "defines", e.get("name", "")),
                )
            for r in data.get("relations", []):
                con.execute(
                    "INSERT OR REPLACE INTO relations(file, from_name, relation, to_name) VALUES(?,?,?,?)",
                    (rel_path, r.get("from", ""), r.get("relation", ""), r.get("to", "")),
                )
            con.execute(
                "INSERT OR REPLACE INTO file_meta(file, mtime, indexed) VALUES(?,?,1)",
                (rel_path, mtime),
            )

    def _mark_file_seen(self, rel: str, mtime: float) -> None:
        """Record file_meta without entities so the file isn't considered stale forever."""
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "INSERT OR REPLACE INTO file_meta(file, mtime, indexed) VALUES(?,?,1)",
                (rel, mtime),
            )

    async def index_file(self, filepath: Path, force_llm: bool = False) -> None:
        # ``force_llm`` bypasses the freshness check and the tree-sitter /
        # structured extractors: the file is sent straight to the LLM
        # extractor, even if it was already indexed by tree-sitter. Use
        # when re-indexing a file whose semantic extraction needs the LLM
        # (e.g. tree-sitter pulled garbage names, mixed-dialect file).
        if not force_llm and not self._file_needs_update(filepath):
            return
        rel = filepath.relative_to(self.project_root).as_posix()
        try:
            stat = filepath.stat()
            mtime = stat.st_mtime
            if stat.st_size > self._max_file_bytes:
                self._mark_file_seen(rel, mtime)
                logger.info("Skipped %s: %.1f MB exceeds %.1f MB limit",
                            rel, stat.st_size / 1024 / 1024, self._max_file_bytes / 1024 / 1024)
                return
            code = filepath.read_text(encoding="utf-8", errors="ignore")
            if len(code.strip()) < 50:
                self._mark_file_seen(rel, mtime)
                return
            self._delete_file_data(rel)
            raw_data, strategy = await self._extract_with_strategy(filepath, code, force_llm=force_llm)
            data = self._sanitize_extracted(rel, raw_data)
            entities_count = len(data.get("entities", []))
            relations_count = len(data.get("relations", []))
            if entities_count > 0 or relations_count > 0:
                self._store_extracted(rel, mtime, data)
                logger.info("Indexed %s: %d entities, %d relations [%s]",
                            rel, entities_count, relations_count, strategy)
            elif strategy == "llm":
                # LLM may have transiently failed — keep as unindexed so a
                # later build retries it.
                logger.warning("Empty LLM extraction for %s — will retry on next build", rel)
            else:
                # Deterministic parser found nothing (re-export only, empty
                # stylesheet, etc.). Mark as seen so it doesn't stay stale.
                self._mark_file_seen(rel, mtime)
                logger.debug("Empty %s extraction for %s — marked seen", strategy, rel)
        except Exception as e:
            logger.warning("Failed to index %s: %s", filepath, e)

    async def reindex_file(
        self, filepath: Path, rebuild_faiss: bool = True, force_llm: bool = False
    ) -> None:
        rel = filepath.relative_to(self.project_root).as_posix()
        self._delete_file_data(rel)
        await self.index_file(filepath, force_llm=force_llm)
        if rebuild_faiss:
            self._rebuild_faiss()
        else:
            self._faiss_dirty = True

    def write_batch(
        self,
        filepath: "Path | str",
        entities: list[dict],
        relations: list[dict],
        rebuild_faiss: bool = False,
    ) -> dict:
        """Direct graph write — bypasses tree-sitter / LLM extractor pipeline.

        Use case: a RAG-builder sub-agent does extraction itself (its own
        LLM call over multi-file context) and submits the resulting graph
        via this method. Skips ``_extract_with_strategy`` but reuses
        ``_sanitize_extracted`` and ``_store_extracted`` so the resulting
        rows are byte-identical to auto-extraction — same validation,
        same entity-type allowlist, same line/snippet enrichment.

        Args:
            filepath: Relative or absolute path. Must exist on disk so
                ``_find_entity_snippet`` can locate line numbers and
                ``stat().st_mtime`` can be recorded.
            entities: List of dicts with at least ``name`` and ``type``.
                ``description`` optional. Line/snippet are auto-filled
                from the file content via ``_enrich_entity_locations``,
                so the caller doesn't need to compute them.
            relations: List of dicts with ``from``, ``relation``, ``to``.
                Unknown relation types are dropped silently.
            rebuild_faiss: When True, rebuilds the FAISS index immediately.
                A sub-agent writing many files at once should leave this
                False and call ``_rebuild_faiss()`` once at the end —
                rebuilding per-file would be O(N²) on the entity corpus.

        Returns:
            ``{"file": rel_path, "entities": N, "relations": M}`` —
            counts after sanitize. The number written may be less than
            the number passed in if entries failed validation.
        """
        if isinstance(filepath, str):
            filepath = Path(filepath)
        if not filepath.is_absolute():
            filepath = self.project_root / filepath
        rel = filepath.relative_to(self.project_root).as_posix()
        try:
            mtime = filepath.stat().st_mtime
        except OSError:
            # File got deleted between the sub-agent's read and this write.
            # Skip — _cleanup_deleted_files on the next graph_build will
            # take care of any stale entries.
            logger.warning("write_batch: %s vanished from disk, skipping", rel)
            return {"file": rel, "entities": 0, "relations": 0, "skipped": "file_not_found"}

        # R3 enforcement: code-extension file with entities but ZERO relations
        # means the caller skipped the import block (every code file has
        # imports at the top). Reject so the model is forced to re-read and
        # emit them. Non-code assets (.html/.css/.json/.md) are exempt.
        # Reverse-grep writes (entities=[], relations=[...]) are also exempt
        # — that branch is handled below by the additive guard.
        _CODE_EXTS = {
            ".py", ".pyi",
            ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
            ".go", ".rs", ".java", ".kt", ".scala",
            ".cs", ".cpp", ".cc", ".c", ".h", ".hpp",
            ".rb", ".php", ".swift",
            # Templates with behaviour: Angular .html has (click)/[binding]
            # handlers and <component-selector> references, Vue/Svelte
            # carry script + template + event handlers in one file.
            ".html", ".vue", ".svelte",
        }
        if entities and not relations and filepath.suffix.lower() in _CODE_EXTS:
            return {
                "file": rel,
                "entities": 0,
                "relations": 0,
                "skipped": (
                    "R3 violated: code/template file requires ≥1 relation. "
                    "For source files: emit --imports--> edges from the "
                    "import block. For Angular/Vue/Svelte templates: emit "
                    "--uses--> for each <component-selector> referenced and "
                    "--calls--> for each (click)/(submit)/event-handler. "
                    "Re-read the file and resubmit with relations populated."
                ),
            }

        # Wipe-then-write is the default (mirrors index_file — gives
        # idempotent re-indexing). BUT guard against destructive
        # reverse-grep calls: when the caller passes only relations
        # (entities=[], relations=[edge,...]) the intent is "add edges
        # to an already-indexed file", NOT "replace it". Wiping would
        # destroy the file's previously-indexed entities. So only wipe
        # when entities are present, or when both lists are empty
        # (explicit full clear of this file).
        if entities or not relations:
            self._delete_file_data(rel)
        else:
            logger.info(
                "write_batch: %s called with entities=[] and %d relations "
                "— treating as additive reverse-grep write, skipping wipe",
                rel, len(relations),
            )

        raw_entities = list(entities or [])
        # Ensure the file's own module-entity exists as a graph node, so the
        # module-level import/call edges (from_name = rel_path) point at a real
        # node — mirrors what the tree-sitter path does via _make_file_entity.
        # Only for full-file writes (entities present); additive reverse-grep
        # writes (entities=[]) must not resurrect a wiped file node.
        if entities and not any(e.get("name") == rel for e in raw_entities if isinstance(e, dict)):
            raw_entities.append(self._make_file_entity(rel, "module", "Module file"))
        raw = {"entities": raw_entities, "relations": relations or []}
        data = self._sanitize_extracted(rel, raw)
        self._store_extracted(rel, mtime, data)

        if rebuild_faiss:
            self._rebuild_faiss()
        else:
            # Defer the costly FAISS rebuild. Caller should explicitly
            # invoke ``_rebuild_faiss()`` after the batch finishes; until
            # then ``search_code`` will see the old vector set.
            self._faiss_dirty = True

        return {
            "file": rel,
            "entities": len(data["entities"]),
            "relations": len(data["relations"]),
        }

    def rebuild_faiss(self) -> None:
        """Public alias for the FAISS reindex — called by RAG-builder
        sub-agents after a batch of write_batch() calls to surface the
        new entities to ``search_code`` / ``graph_find_similar``.
        """
        self._rebuild_faiss()

    def add_entity(
        self,
        filepath: "Path | str",
        name: str,
        type: str,
        description: str = "",
    ) -> dict:
        """Add a single entity to the graph WITHOUT wiping the file's
        existing rows. Idempotent — the entities table has a UNIQUE
        constraint on ``(file, name, type)`` so re-inserting the same
        triple updates the description / snippet in-place.

        Auto-fills line_start / line_end / snippet via the same
        ``_enrich_entity_locations`` helper auto-extraction uses, so
        the caller doesn't need to supply them. The file-entity row
        (the ``module`` record that ``_make_file_entity`` creates for
        every indexed file) is auto-created on first call so relations
        targeting this file resolve cleanly.

        Used by RAG-builder sub-agents that read source themselves and
        add entities one at a time as they walk the file — simpler
        per-call shape than ``write_batch`` (which requires the full
        entities + relations arrays upfront).
        """
        if isinstance(filepath, str):
            filepath = Path(filepath)
        if not filepath.is_absolute():
            filepath = self.project_root / filepath
        rel = filepath.relative_to(self.project_root).as_posix()
        try:
            mtime = filepath.stat().st_mtime
        except OSError:
            logger.warning("add_entity: %s vanished from disk, skipping", rel)
            return {"file": rel, "added": 0, "skipped": "file_not_found"}

        # Sanitize through the same pipeline as full extraction —
        # validates entity type against _ALLOWED_ENTITY_TYPES and
        # enriches line/snippet.
        sanitized = self._sanitize_extracted(rel, {
            "entities": [{"name": name, "type": type, "description": description}],
            "relations": [],
        })
        if not sanitized["entities"]:
            return {"file": rel, "added": 0, "skipped": "invalid_type_or_empty_name"}

        with sqlite3.connect(self.db_path) as con:
            # Self-heal: ensure the file-entity exists. ``add_entity`` may
            # be the first call for a file (no prior tree-sitter pass),
            # in which case ``_make_file_entity`` never ran. Without this
            # row, relations targeting this file's path won't have an
            # endpoint to resolve against.
            con.execute(
                "INSERT OR IGNORE INTO entities(file, name, type, description) VALUES(?,?,?,?)",
                (rel, rel, "module", "Source file"),
            )
            for e in sanitized["entities"]:
                traits = self._detect_traits(
                    e.get("name", ""), e.get("snippet", ""), rel, e.get("type", ""),
                )
                # Use ON CONFLICT (not INSERT OR REPLACE) so a re-insert with
                # empty description/snippet does NOT overwrite the richer text
                # captured by an earlier pass. Common scenario: the primary
                # indexing pass writes a full description, then the reverse-
                # grep pass (or another sub-agent invocation) re-adds the
                # same entity with description="" — without COALESCE the
                # rich text would be wiped, leaving search worse than before.
                con.execute(
                    """INSERT INTO entities(file, name, type, description, line_start, line_end, snippet, traits)
                    VALUES(?,?,?,?,?,?,?,?)
                    ON CONFLICT(file, name, type) DO UPDATE SET
                        description = COALESCE(NULLIF(excluded.description, ''), description),
                        line_start  = COALESCE(excluded.line_start, line_start),
                        line_end    = COALESCE(excluded.line_end, line_end),
                        snippet     = COALESCE(NULLIF(excluded.snippet, ''), snippet),
                        traits      = excluded.traits""",
                    (
                        rel, e["name"], e["type"], e.get("description", ""),
                        e.get("line_start"), e.get("line_end"),
                        e.get("snippet", ""), traits,
                    ),
                )
                # Auto-emit ``file --defines--> entity``. This edge is
                # trivially true for every declared entity ("the file
                # this is declared in defines it") — sub-agents shouldn't
                # spend tool calls re-asserting it. By auto-emitting, the
                # K/M ratio surfaces ONLY the meaningful cross-entity
                # edges (calls / uses / imports / inherits / instantiates)
                # the agent actively wrote, which is the better signal
                # for "is this graph well-connected".
                con.execute(
                    "INSERT OR IGNORE INTO relations(file, from_name, relation, to_name) VALUES(?,?,?,?)",
                    (rel, rel, "defines", e["name"]),
                )
            con.execute(
                "INSERT OR REPLACE INTO file_meta(file, mtime, indexed) VALUES(?,?,1)",
                (rel, mtime),
            )

        # FAISS rebuild deferred — caller batches add_entity() calls and
        # invokes graph_rebuild_faiss() once at the end.
        self._faiss_dirty = True
        return {"file": rel, "added": len(sanitized["entities"]), "name": name, "type": sanitized["entities"][0]["type"]}

    def add_relation(
        self,
        filepath: "Path | str",
        from_name: str,
        relation: str,
        to_name: str,
    ) -> dict:
        """Add a single relation edge to the graph WITHOUT wiping the
        file's existing rows. Idempotent — the relations table has a
        UNIQUE constraint on ``(file, from_name, relation, to_name)``
        so duplicate calls are silently dropped.

        Validates ``relation`` against ``_ALLOWED_RELATION_TYPES``.
        Trims/normalizes ``from_name`` and ``to_name`` the same way
        the bulk-write path does. Used by RAG-builder sub-agents to
        layer cross-file build-time edges (e.g.
        ``write_cover_active uses cover_active.h``) one at a time.

        Note: relations point at NAMES, not files — so ``to_name`` can
        be either a file-entity name (``"cover_active.h"`` to link to
        another file) or an entity defined inside any file (``"User"``
        to link to a class defined elsewhere). The lookup is name-based.
        """
        if isinstance(filepath, str):
            filepath = Path(filepath)
        if not filepath.is_absolute():
            filepath = self.project_root / filepath
        rel = filepath.relative_to(self.project_root).as_posix()

        from_name = self._normalize_whitespace(str(from_name or ""), limit=160)
        to_name = self._normalize_whitespace(str(to_name or ""), limit=160)
        rel_type = self._normalize_whitespace(str(relation or ""), limit=40).lower()

        if not from_name or not to_name:
            return {"file": rel, "added": 0, "skipped": "empty_name"}
        if rel_type not in _ALLOWED_RELATION_TYPES:
            return {
                "file": rel, "added": 0,
                "skipped": f"unknown_relation_type:{rel_type}",
                "allowed": sorted(_ALLOWED_RELATION_TYPES),
            }

        with sqlite3.connect(self.db_path) as con:
            cur = con.execute(
                "INSERT OR IGNORE INTO relations(file, from_name, relation, to_name) VALUES(?,?,?,?)",
                (rel, from_name, rel_type, to_name),
            )
            inserted = cur.rowcount  # 0 if duplicate, 1 if new

        self._faiss_dirty = True
        return {
            "file": rel, "added": inserted,
            "from": from_name, "relation": rel_type, "to": to_name,
            "duplicate": inserted == 0,
        }

    @staticmethod
    def get_schema() -> dict:
        """Return the graph's validation schema — which entity types
        and relation types are accepted by ``write_batch`` / ``add_entity``
        / ``add_relation``. Single source of truth, queryable so
        callers don't have to duplicate the lists in their prompts
        or docstrings (which inevitably drift from this code).

        Returns:
            ``{"entity_types": sorted[str],
               "relation_types": sorted[str]}``

        Anything emitted with a type / relation outside these sets is
        silently dropped by ``_sanitize_relations`` / sanitize-extracted.
        """
        return {
            "entity_types": sorted(_ALLOWED_ENTITY_TYPES),
            "relation_types": sorted(_ALLOWED_RELATION_TYPES),
        }

    def mark_stale(
        self,
        filepath: "Path | str",
        cascade: bool = True,
    ) -> dict:
        """Flag a file (and optionally its dependents) as needing reindex.

        Mechanics: sets ``file_meta.mtime`` to 0 so ``_file_needs_update``
        compares 0 to disk-mtime, they differ, returns True. The next
        ``graph_pending_files`` / ``graph_build`` / ``rag_rebuild
        scope=stale`` query picks the file up.

        Entities and relations are NOT deleted here — old data stays
        queryable for ``graph_explain`` / ``graph_find_usages`` until
        the actual reindex runs. The stale flag is purely a hint.

        When ``cascade=True`` (default), also marks dependents stale —
        every file whose graph edges target an entity defined in the
        changed file. Rationale: if I rename a function in ``foo.py``,
        the ``imports`` / ``calls`` edges from ``bar.py`` (which uses
        that function) are now broken, even though ``bar.py``'s own
        mtime didn't change. Without cascade, those edges stay stale
        forever — the partial reindex of ``foo.py`` alone makes them
        worse, not better.

        Args:
            filepath: Project-relative path of the changed file.
            cascade: If True, also flag every file that depends on
                ``filepath``'s entities via ``affected_files``.

        Returns: ``{"marked": [<rel>, ...]}`` — list of all files
            flagged (always includes the target itself).
        """
        if isinstance(filepath, str):
            filepath = Path(filepath)
        if not filepath.is_absolute():
            filepath = self.project_root / filepath
        try:
            rel = filepath.relative_to(self.project_root).as_posix()
        except ValueError:
            return {"marked": [], "skipped": "outside_project_root"}

        targets = {rel}
        if cascade:
            try:
                affected = self.affected_files(filepath)
                targets.update(affected.get("files", []))
            except Exception as e:
                # affected_files might fail if the file isn't in the
                # graph yet (new file the agent just created) — that's
                # fine, just skip cascade.
                logger.debug("mark_stale: cascade skipped for %s: %s", rel, e)

        marked: list[str] = []
        with sqlite3.connect(self.db_path) as con:
            for t in targets:
                cur = con.execute(
                    "UPDATE file_meta SET mtime = 0 WHERE file = ?", (t,),
                )
                if cur.rowcount > 0:
                    marked.append(t)
        return {"marked": sorted(marked)}

    @staticmethod
    def _import_candidates(rel: str) -> list[str]:
        """All plausible ``to_name`` forms an importer might use for
        the file at ``rel`` (project-relative, forward slashes).

        Imports edges are stored as the module string as it appears
        in source. That's a different shape for each language and
        sometimes for each style:

          Python ``from src.services.user import X``  → ``"src.services.user"``
          Python ``import src.services.user``         → ``"src.services.user"``
          TS    ``from "./services/user"``            → ``"./services/user"``
          TS    ``from "../services/user.ts"``        → ``"../services/user.ts"``
          File path itself (rare convention)          → ``"src/services/user.py"``

        ``affected_files`` needs to match all of them. Build the candidate
        set here in one place.
        """
        cands = {rel}

        # Strip extension once if present.
        if "." in rel.rsplit("/", 1)[-1]:
            base = rel.rsplit(".", 1)[0]   # ``src/services/user.py`` → ``src/services/user``
        else:
            base = rel

        # Path form without extension (TS convention).
        cands.add(base)
        cands.add(f"./{base}")
        cands.add(f"./{base.rsplit('/', 1)[-1]}")  # ``./user``

        # Python dotted form: drop trailing ``__init__`` and any
        # common prefix variations.
        py = base.replace("/", ".")
        if py.endswith(".__init__"):
            py = py[: -len(".__init__")]
        cands.add(py)
        # Also strip leading ``src.`` since many projects import without it.
        if py.startswith("src."):
            cands.add(py[4:])
        # And without the top dir entirely (e.g. ``services.user``).
        parts = py.split(".")
        if len(parts) > 2:
            cands.add(".".join(parts[1:]))
            cands.add(".".join(parts[2:]))

        return sorted(c for c in cands if c)

    def affected_files(self, filepath: "Path | str") -> dict:
        """Files whose graph entries depend on ``filepath``'s exports.

        Returns the union of:
          * files that ``import`` from this file (forward import edge),
          * files that ``call`` / ``use`` / ``instantiate`` / ``inherit``
            any entity defined in this file (reverse-resolution via the
            entity name).

        Used by RAG-builder sub-agents for incremental rebuilds: when
        ``foo.py`` is modified, ``affected_files('foo.py')`` returns
        ``bar.py``, ``baz.py``, etc. — the agent reindexes that set
        instead of the whole project.

        Args:
            filepath: Relative or absolute path. The file itself is
                always included in the result (since it changed).

        Returns:
            ``{"target": rel_path, "files": [<rel>, ...]}`` sorted.
            Result includes the target file itself.
        """
        if isinstance(filepath, str):
            filepath = Path(filepath)
        if not filepath.is_absolute():
            filepath = self.project_root / filepath
        rel = filepath.relative_to(self.project_root).as_posix()

        with sqlite3.connect(self.db_path) as con:
            # Entity names defined in the target file.
            target_names = {
                row[0] for row in con.execute(
                    "SELECT name FROM entities WHERE file = ?", (rel,)
                ).fetchall()
            }

            affected: set[str] = {rel}

            # Reverse-resolution: any file whose relations point at our
            # target's exports (calls / uses / inherits / instantiates).
            if target_names:
                placeholders = ",".join("?" * len(target_names))
                rows = con.execute(
                    f"""SELECT DISTINCT file FROM relations
                        WHERE to_name IN ({placeholders})
                          AND relation IN ('calls','uses','inherits','instantiates')""",
                    tuple(target_names),
                ).fetchall()
                affected.update(r[0] for r in rows)

            # Forward import edge — files we import are not "affected" by
            # OUR change (their entities didn't change), so we skip that
            # direction. But files that import US obviously are.
            #
            # The trap: import edges are stored under the module string
            # as it appears in source (``"src.services.user"`` Python,
            # ``"./services/user"`` TS) — NOT the project-relative file
            # path. Without ``_import_candidates`` this query would only
            # match the rare convention where the importer emitted the
            # raw path; standard Python / TS imports would silently
            # never match → ``affected_files(foo.py) = [foo.py]`` only.
            cands = self._import_candidates(rel)
            placeholders = ",".join("?" * len(cands))
            rows = con.execute(
                f"SELECT DISTINCT file FROM relations "
                f"WHERE to_name IN ({placeholders}) AND relation = 'imports'",
                tuple(cands),
            ).fetchall()
            affected.update(r[0] for r in rows)

        return {"target": rel, "files": sorted(affected)}

    def get_build_status(self) -> dict:
        files = self._get_files()
        indexed_files = self._get_files_indexed()
        existing_rel_paths = {p.relative_to(self.project_root).as_posix() for p in files}
        deleted_files = len(set(indexed_files) - existing_rel_paths)
        indexed_project_files = len(set(indexed_files) & existing_rel_paths)
        stale_files = sum(1 for f in files if self._file_needs_update(f))
        return {
            "total_files": len(files),
            "indexed_files": len(indexed_files),
            "indexed_project_files": indexed_project_files,
            "deleted_files": deleted_files,
            "stale_files": stale_files,
            "needs_build": stale_files > 0 or deleted_files > 0,
        }

    def get_pending_files(self) -> dict:
        """List files that don't match the graph: never indexed, stale, or deleted on disk."""
        files = self._get_files()
        indexed = set(self._get_files_indexed())
        existing = {p.relative_to(self.project_root).as_posix(): p for p in files}

        unindexed: list[str] = []
        stale: list[str] = []
        for rel, path in existing.items():
            if rel not in indexed:
                unindexed.append(rel)
            elif self._file_needs_update(path):
                stale.append(rel)

        missing = sorted(indexed - set(existing.keys()))
        return {
            "unindexed": sorted(unindexed),
            "stale": sorted(stale),
            "missing": missing,
        }

    async def build(
        self, max_files: Optional[int] = None, force_llm: bool = False
    ) -> dict:
        """Index every stale file by default. Pass ``max_files`` to cap one call.

        ``force_llm=True`` routes every file through the LLM extractor instead
        of tree-sitter / structured parsers, AND re-indexes files that aren't
        stale — useful when tree-sitter mis-parsed a previous build or when
        you need semantic extraction across the board. Expensive: every file
        becomes one LLM call. Cap with ``max_files`` to keep cost predictable.
        """
        self._is_building = True
        try:
            t0 = time.time()
            files = self._get_files()
            t1 = time.time()
            logger.info("Graph file scan: %d files in %.2fs", len(files), t1 - t0)
            deleted_files = self._cleanup_deleted_files(files)
            if force_llm:
                # Re-index everything, not just stale — caller asked for an
                # LLM pass and probably wants existing tree-sitter rows replaced.
                stale = list(files)
            else:
                stale = [f for f in files if self._file_needs_update(f)]
            to_update = stale if max_files is None else stale[:max_files]
            remaining = max(0, len(stale) - len(to_update))
            logger.info("Graph build: %d/%d files need indexing (%d remaining, force_llm=%s)",
                        len(to_update), len(files), remaining, force_llm)

            sem = asyncio.Semaphore(5)

            async def _index_with_sem(f: Path) -> None:
                async with sem:
                    await self.index_file(f, force_llm=force_llm)

            await asyncio.gather(*[_index_with_sem(f) for f in to_update])

            with sqlite3.connect(self.db_path) as con:
                e_count = con.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
                r_count = con.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
                f_count = con.execute("SELECT COUNT(*) FROM file_meta").fetchone()[0]

            if to_update or deleted_files:
                self._rebuild_faiss()

            logger.info("Graph build done in %.2fs: %d indexed, %d entities",
                        time.time() - t0, len(to_update), e_count)
            return {
                "indexed": len(to_update),
                "remaining": remaining,
                "deleted": deleted_files,
                "total_files": f_count,
                "entities": e_count,
                "relations": r_count,
            }
        finally:
            self._is_building = False

    @staticmethod
    def _summarize_relations(rels: list[tuple[str, str]], cap_per_rel: int = 6) -> str:
        """Compact one-line digest of an entity's outgoing relations.

        ``rels`` is a list of (relation, to_name) tuples. We dedup, group
        by relation type, and cap each group so the embed text stays
        bounded. Output looks like:
            "instantiates AntFlex, FlexProps; calls cn; uses className"
        """
        if not rels:
            return ""
        by_rel: dict[str, list[str]] = {}
        for rel, target in rels:
            bucket = by_rel.setdefault(rel, [])
            if target not in bucket:
                bucket.append(target)
        parts = []
        for rel in ("instantiates", "calls", "inherits", "uses", "imports", "defines"):
            if rel not in by_rel:
                continue
            targets = by_rel[rel][:cap_per_rel]
            parts.append(f"{rel} {', '.join(targets)}")
        return "; ".join(parts)

    @classmethod
    def _faiss_entity_text(
        cls,
        name: str,
        description: Optional[str],
        snippet: Optional[str],
        relations: Optional[list[tuple[str, str]]] = None,
    ) -> str:
        """Build the text we feed into FAISS for one entity.

        Layered context:
        - name (always)
        - relation digest if available — gives short generic names like
          ``Flex`` a structural fingerprint (instantiates AntFlex, calls
          cn) so FAISS can place them near other wrappers
        - snippet (capped) if available — actual code semantics
        - description as a last-resort fallback for module-level rows
        """
        parts: list[str] = [name]
        rel_summary = cls._summarize_relations(relations or [])
        if rel_summary:
            parts.append(rel_summary)
        snip = (snippet or "").strip()
        if snip:
            parts.append(snip[:400])
        elif description:
            parts.append((description or "").strip())
        return "\n".join(parts)

    def _rebuild_faiss(self) -> None:
        self._faiss_dirty = False
        try:
            import faiss
            with sqlite3.connect(self.db_path) as con:
                rows = con.execute(
                    "SELECT file, name, description, snippet FROM entities ORDER BY id"
                ).fetchall()
                # Pre-load all relations grouped by (file, from_name) so the
                # per-entity lookup is O(1) instead of O(N²) sub-queries.
                rels_rows = con.execute(
                    "SELECT file, from_name, relation, to_name FROM relations"
                ).fetchall()
            if not rows:
                self.faiss_index = None
                self.faiss_names = []
                return
            from collections import defaultdict
            rels_map: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
            for f, fn, r, tn in rels_rows:
                rels_map[(f, fn)].append((r, tn))
            texts = [
                self._faiss_entity_text(
                    r[1], r[2], r[3],
                    relations=rels_map.get((r[0], r[1]), []),
                )
                for r in rows
            ]
            self.faiss_names = [r[1] for r in rows]

            # Content-addressable cache (Cursor-style Merkle reuse): each
            # entity's embedding is keyed by sha256(text + model_id). On
            # branch switches / partial reindexes most rows hit the cache
            # and skip the encoder entirely.
            import hashlib
            from .embedder import _embed_model_id
            model_id = _embed_model_id()
            hashes = [hashlib.sha256(f"{model_id}\n{t}".encode("utf-8")).hexdigest()[:32] for t in texts]
            cached_vecs: dict[str, "np.ndarray"] = {}
            with sqlite3.connect(self.db_path) as con:
                # Look up in batches — SQLite caps `IN (?,?,...)` at ~999 params.
                unique_hashes = list({h for h in hashes})
                for i in range(0, len(unique_hashes), 800):
                    batch = unique_hashes[i:i + 800]
                    placeholders = ",".join("?" * len(batch))
                    rows_c = con.execute(
                        f"SELECT hash, vec, dim FROM embedding_cache "
                        f"WHERE model = ? AND hash IN ({placeholders})",
                        (model_id, *batch),
                    ).fetchall()
                    for h, blob, _dim in rows_c:
                        import numpy as _np
                        cached_vecs[h] = _np.frombuffer(blob, dtype=_np.float32)

            missing_idxs = [i for i, h in enumerate(hashes) if h not in cached_vecs]
            cache_hits = len(hashes) - len(missing_idxs)
            self._last_cache_hits = cache_hits
            self._last_cache_total = len(hashes)

            if missing_idxs:
                missing_texts = [texts[i] for i in missing_idxs]
                new_vecs = encode_documents(
                    missing_texts, normalize_embeddings=True,
                    show_progress_bar=False, batch_size=encode_batch_size(),
                ).astype("float32")
                for pos, src_idx in enumerate(missing_idxs):
                    cached_vecs[hashes[src_idx]] = new_vecs[pos]
                # Persist new entries.
                with sqlite3.connect(self.db_path) as con:
                    con.executemany(
                        "INSERT OR REPLACE INTO embedding_cache(hash, model, dim, vec) VALUES (?, ?, ?, ?)",
                        [
                            (hashes[i], model_id, int(new_vecs.shape[1]), new_vecs[pos].tobytes())
                            for pos, i in enumerate(missing_idxs)
                        ],
                    )
                    con.commit()

            import numpy as np
            embeddings = np.stack([cached_vecs[h] for h in hashes]).astype("float32")
            dim = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(embeddings)
            logger.info(
                "FAISS index built: %d entities  (cache hits %d/%d = %.1f%%)",
                len(self.faiss_names), cache_hits, len(hashes),
                (cache_hits / max(len(hashes), 1)) * 100,
            )
        except Exception as e:
            self.faiss_index = None
            self.faiss_names = []
            logger.warning("FAISS rebuild failed: %s", e)

    def find_usages(self, name: str) -> list[dict]:
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute(
                "SELECT file, from_name, relation, to_name FROM relations WHERE to_name = ? ORDER BY file, from_name, relation",
                (name,),
            ).fetchall()
        return [{"file": r[0], "from": r[1], "relation": r[2], "to": r[3]} for r in rows]

    def get_file_deps(self, rel_path: str) -> list[dict]:
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute(
                "SELECT from_name, relation, to_name FROM relations WHERE file = ? AND relation IN ('imports','inherits','uses') ORDER BY relation, from_name, to_name",
                (rel_path,),
            ).fetchall()
        return [{"from": r[0], "relation": r[1], "to": r[2]} for r in rows]

    def get_callers(self, function_name: str) -> list[dict]:
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute(
                "SELECT file, from_name FROM relations WHERE to_name = ? AND relation = 'calls' ORDER BY file, from_name",
                (function_name,),
            ).fetchall()
        return [{"file": r[0], "caller": r[1]} for r in rows]

    def get_file_entities(self, rel_path: str) -> list[dict]:
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute(
                "SELECT name, type, description, line_start, line_end, snippet FROM entities "
                "WHERE file = ? "
                "ORDER BY CASE WHEN line_start IS NULL THEN 1 ELSE 0 END, line_start, type, name",
                (rel_path,),
            ).fetchall()
        return [
            {"name": r[0], "type": r[1], "description": r[2], "line_start": r[3], "line_end": r[4], "snippet": r[5] or ""}
            for r in rows
        ]

    @staticmethod
    def format_entity_location(entity: dict) -> str:
        location = f"{entity['file']}"
        if entity.get("line_start"):
            location += f":{entity['line_start']}"
            if entity.get("line_end") and entity["line_end"] != entity["line_start"]:
                location += f"-{entity['line_end']}"
        return location

    def format_entity_result(self, entity: dict, include_snippet: bool = True, indent: str = "  • ") -> list[str]:
        desc = f": {entity['description']}" if entity.get("description") else ""
        lines = [f"{indent}[{entity['type']}] {entity['name']}{desc}  ({self.format_entity_location(entity)})"]
        if include_snippet and entity.get("snippet"):
            for snippet_line in entity["snippet"].splitlines():
                lines.append(f"      {snippet_line}")
        return lines

    @staticmethod
    def _get_tree_sitter_parser(path: Path):
        language_name = _TREE_SITTER_LANGUAGES.get(path.suffix.lower())
        if not language_name:
            return None
        try:
            from tree_sitter_language_pack import get_parser
            return get_parser(language_name)
        except Exception:
            try:
                from tree_sitter_languages import get_parser
                return get_parser(language_name)
            except Exception:
                return None

    @staticmethod
    def _extract_tree_sitter_name(node, source: bytes) -> str:
        # `const Flex = ...` parses as lexical_declaration → variable_declarator
        # → name. The old fallback loop grabbed the leading "const"/"let"
        # keyword as the entity name, so every const/let declaration ended up
        # in the graph as "const"/"let" instead of its real identifier. Find
        # the declarator child first and read its `name` field.
        if node.type in ("lexical_declaration", "variable_declaration"):
            for child in node.children:
                if child.type in ("variable_declarator", "init_declarator"):
                    name_node = child.child_by_field_name("name")
                    if name_node is None:
                        # Some grammars expose the identifier as a positional
                        # child rather than a named field.
                        for grand in child.children:
                            if grand.type in ("identifier", "property_identifier", "type_identifier"):
                                name_node = grand
                                break
                    if name_node is not None:
                        return source[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="ignore").strip()
        for field_name in _TREE_SITTER_NAME_FIELDS:
            child = node.child_by_field_name(field_name)
            if child is not None:
                return source[child.start_byte:child.end_byte].decode("utf-8", errors="ignore").strip()
        # Last-resort identifier scan, but skip the variable-declaration
        # keywords that tripped the old logic.
        skip = {"const", "let", "var", "function", "class", "type", "interface", "enum"}
        for child in node.children:
            child_text = source[child.start_byte:child.end_byte].decode("utf-8", errors="ignore").strip()
            if not child_text or child_text in skip:
                continue
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.$]*", child_text):
                return child_text
        return ""

    def _find_entity_with_tree_sitter(self, path: Path, entity_name: str, lines: list[str]) -> dict:
        parser = self._get_tree_sitter_parser(path)
        if parser is None:
            return {"line_start": None, "line_end": None, "snippet": ""}
        try:
            source = path.read_bytes()
        except Exception:
            return {"line_start": None, "line_end": None, "snippet": ""}
        try:
            tree = parser.parse(source)
        except Exception:
            return {"line_start": None, "line_end": None, "snippet": ""}
        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            if node.type in _TREE_SITTER_NODE_TYPES:
                node_name = self._extract_tree_sitter_name(node, source)
                if node_name == entity_name:
                    start_line = node.start_point[0] + 1
                    end_line = node.end_point[0] + 1
                    snippet = "\n".join(lines[node.start_point[0]:node.end_point[0] + 1]).strip()
                    return {"line_start": start_line, "line_end": end_line, "snippet": snippet}
            stack.extend(reversed(node.children))
        return {"line_start": None, "line_end": None, "snippet": ""}

    def _find_entity_snippet(self, rel_path: str, entity_name: str) -> dict:
        path = self.project_root / rel_path
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return {"line_start": None, "line_end": None, "snippet": ""}
        lines = text.splitlines()
        if not lines:
            return {"line_start": None, "line_end": None, "snippet": ""}
        tree_sitter_result = self._find_entity_with_tree_sitter(path, entity_name, lines)
        if tree_sitter_result["line_start"] is not None:
            return tree_sitter_result
        # Priority-ordered: scan ALL lines for a real definition form
        # (def/class/function/const/...) before falling back to a bare
        # textual mention. A line-by-line loop that breaks on the first
        # match of ANY pattern lets a comment merely mentioning the name
        # (e.g. line 43) shadow the actual `def` further down (line 121).
        # So loop patterns in the OUTER position: the weakest pattern
        # (bare name) is only consulted if no definition form exists.
        patterns = [
            f"def {entity_name}",
            f"class {entity_name}",
            f"function {entity_name}",
            f"const {entity_name}",
            f"let {entity_name}",
            f"var {entity_name}",
            entity_name,
        ]
        match_index = None
        for pattern in patterns:
            for idx, line in enumerate(lines):
                if pattern in line:
                    match_index = idx
                    break
            if match_index is not None:
                break
        if match_index is None:
            return {"line_start": None, "line_end": None, "snippet": ""}
        # Anchor line_start to the matched definition line itself; keep a
        # couple of leading lines only for snippet context.
        start = max(0, match_index - 2)
        end = min(len(lines), match_index + 3)
        snippet = "\n".join(lines[start:end]).strip()
        return {"line_start": match_index + 1, "line_end": end, "snippet": snippet}

    def _search_raw_occurrences(self, query: str, limit: int = 10) -> list[dict]:
        normalized = query.strip()
        if not normalized:
            return []
        results: list[dict] = []
        lowered = normalized.lower()
        for path in self._get_files():
            rel_path = path.relative_to(self.project_root).as_posix()
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            lines = text.splitlines()
            for idx, line in enumerate(lines):
                if lowered not in line.lower():
                    continue
                start = max(0, idx - 2)
                end = min(len(lines), idx + 3)
                snippet = "\n".join(lines[start:end]).strip()
                if not snippet:
                    continue
                results.append({
                    "file": rel_path,
                    "name": normalized,
                    "type": "symbol",
                    "description": "Raw code match (not indexed as graph entity)",
                    "line_start": idx + 1,
                    "line_end": idx + 1,
                    "snippet": snippet,
                })
                if len(results) >= limit:
                    return results
        return results

    def search_entity(self, query: str, entity_type: Optional[str] = None, limit: int = 10) -> list[dict]:
        # Tokenize so multi-word queries ("Layout Sider Header") don't fall
        # through as a single LIKE that nothing matches. Each token contributes
        # an OR-clause; we then favor rows that hit the most tokens.
        # Also split CamelCase / snake_case inside each token so 'UserService'
        # finds 'user_service' and 'render_fn' is found by 'renderFn'.
        raw_tokens = [t for t in re.findall(r"[A-Za-z0-9_]+", query) if len(t) > 1]
        seen: set[str] = set()
        tokens: list[str] = []
        for tok in raw_tokens:
            for piece in [tok, *tok.split("_"), *re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z0-9]+|[A-Z]+", tok)]:
                if len(piece) < 2:
                    continue
                key = piece.lower()
                if key in seen:
                    continue
                seen.add(key)
                tokens.append(piece)
        if not tokens:
            tokens = [query.strip()] if query.strip() else []

        with sqlite3.connect(self.db_path) as con:
            if not tokens:
                # Empty query — list-all mode. Useful when the caller only
                # wants a type filter ("show every class", browse-by-type
                # in a UI). Returns rows ordered by file/line.
                params: list = []
                where_type = ""
                if entity_type:
                    where_type = " WHERE type = ?"
                    params.append(entity_type)
                params.append(limit * 3)
                sql = (
                    "SELECT file, name, type, description, line_start, line_end, snippet "
                    f"FROM entities{where_type} "
                    "ORDER BY file, "
                    "  CASE WHEN line_start IS NULL THEN 1 ELSE 0 END, line_start, name "
                    "LIMIT ?"
                )
                rows = con.execute(sql, params).fetchall()
            else:
                like_clauses = " OR ".join(["lower(name) LIKE lower(?)"] * len(tokens))
                like_params = [f"%{t}%" for t in tokens]
                # Score = count of tokens that match (descending), then exact-match bonus.
                score_terms = " + ".join(
                    [f"(CASE WHEN lower(name) LIKE lower(?) THEN 1 ELSE 0 END)"] * len(tokens)
                )
                score_params = [f"%{t}%" for t in tokens]
                exact_q = query.lower()
                # SQLite binds ``?`` placeholders by position in the query
                # string, so the params list MUST follow the same order:
                #   1. score_terms in SELECT
                #   2. like_clauses in WHERE
                #   3. AND type = ? (when filtering)
                #   4. ORDER BY exact-match CASE
                #   5. ORDER BY prefix-match CASE
                #   6. LIMIT
                params: list = [*score_params, *like_params]
                where_type = ""
                if entity_type:
                    where_type = " AND type = ?"
                    params.append(entity_type)
                params.extend([exact_q, f"{exact_q}%", limit * 3])

                sql = (
                    f"SELECT file, name, type, description, line_start, line_end, snippet, "
                    f"  ({score_terms}) AS hits "
                    f"FROM entities "
                    f"WHERE ({like_clauses}){where_type} "
                    f"ORDER BY hits DESC, "
                    f"  CASE WHEN lower(name) = ? THEN 0 "
                    f"       WHEN lower(name) LIKE ? THEN 1 "
                    f"       ELSE 2 END, "
                    f"  name "
                    f"LIMIT ?"
                )
                rows = con.execute(sql, params).fetchall()
        results = []
        for r in rows:
            line_start = r[4]
            line_end = r[5]
            snippet = r[6] or ""
            if line_start is None or not snippet:
                snippet_info = self._find_entity_snippet(r[0], r[1])
                line_start = snippet_info["line_start"]
                line_end = snippet_info["line_end"]
                snippet = snippet_info["snippet"]
            results.append({
                "file": r[0],
                "name": r[1],
                "type": r[2],
                "description": r[3],
                "line_start": line_start,
                "line_end": line_end,
                "snippet": snippet,
            })
        if self.faiss_index is not None and self.faiss_names:
            try:
                q_vec = encode_query([query], normalize_embeddings=True,
                                     show_progress_bar=False).astype("float32")
                k = min(limit, self.faiss_index.ntotal)
                scores, indices = self.faiss_index.search(q_vec, k)
                faiss_top = {self.faiss_names[i] for s, i in zip(scores[0], indices[0])
                             if i >= 0 and s > 0.3}
                results.sort(key=lambda r: (0 if r["name"] in faiss_top else 1, r["name"]))
            except Exception as e:
                logger.warning("FAISS search failed: %s", e)
        results = results[:limit]
        if results or entity_type:
            return results
        # LIKE found nothing — try FAISS-only as a fallback for natural-language
        # queries that don't match any entity name (e.g. Russian questions).
        if self.faiss_index is not None and self.faiss_names:
            try:
                q_vec = encode_query([query], normalize_embeddings=True,
                                     show_progress_bar=False).astype("float32")
                k = min(limit, self.faiss_index.ntotal)
                scores, indices = self.faiss_index.search(q_vec, k)
                faiss_names = [self.faiss_names[i] for s, i in zip(scores[0], indices[0])
                               if i >= 0 and s > 0.1][:limit]
                if faiss_names:
                    results = self.search_entity(" ".join(faiss_names), entity_type=entity_type, limit=limit)
                    if results:
                        return results
            except Exception as e:
                logger.warning("FAISS fallback search failed: %s", e)
        return self._search_raw_occurrences(query, limit=limit)

    def get_subgraph(
        self,
        entity_name: str,
        depth: int = 2,
        per_node_cap: int = 50,
        hub_fanout_threshold: int = 10,
    ) -> dict:
        """BFS expansion around an entity.

        Common names like ``Layout``/``Header`` may appear as ``to_name`` in
        thousands of relations because every file declaring ``const Header = ...``
        contributes a separate node by lexical name. To keep results usable we
        cap how many relations we walk through *per BFS node* — the rest are
        counted as ``truncated_at`` so the caller sees the partial-result flag.

        BFS walks by *name* (since relations only store names, not entity
        ids). Names defined in 10+ files (``log``, ``useEffect``, ``FC``)
        would otherwise pull in unrelated neighborhoods on the next hop —
        we stop expanding through them at ``cur_depth >= 1`` while still
        recording the relation so the caller sees the connection.
        """
        visited: set[str] = set()
        seen_rels: set = set()
        all_relations: list = []
        truncated_nodes: list[str] = []
        skipped_hubs: list[str] = []
        queue: list[tuple[str, int]] = [(entity_name, 0)]
        with sqlite3.connect(self.db_path) as con:
            # Pre-compute file-fanout per name so we can recognize hubs in O(1)
            # during BFS instead of running a count query per neighbor.
            name_fanout = dict(con.execute(
                "SELECT name, COUNT(DISTINCT file) FROM entities GROUP BY name"
            ).fetchall())
            while queue:
                current, cur_depth = queue.pop(0)
                if current in visited or cur_depth > depth:
                    continue
                # Stop expanding through hubs after the first hop. The anchor
                # itself (cur_depth == 0) is always expanded — user explicitly
                # asked about it — but neighbor hubs are too noisy to follow.
                if cur_depth > 0 and name_fanout.get(current, 0) >= hub_fanout_threshold:
                    visited.add(current)
                    skipped_hubs.append(f"{current} ({name_fanout[current]} files)")
                    continue
                visited.add(current)
                # Probe count first to surface truncation in the result.
                total = con.execute(
                    "SELECT COUNT(*) FROM relations WHERE from_name = ? OR to_name = ?",
                    (current, current),
                ).fetchone()[0]
                rows = con.execute(
                    "SELECT file, from_name, relation, to_name FROM relations "
                    "WHERE from_name = ? OR to_name = ? LIMIT ?",
                    (current, current, per_node_cap),
                ).fetchall()
                if total > per_node_cap:
                    truncated_nodes.append(f"{current} ({total} total, kept {per_node_cap})")
                for r in rows:
                    key = (r[1], r[2], r[3])
                    if key in seen_rels:
                        continue
                    seen_rels.add(key)
                    all_relations.append({"file": r[0], "from": r[1], "relation": r[2], "to": r[3]})
                    if r[1] not in visited:
                        queue.append((r[1], cur_depth + 1))
                    if r[3] not in visited:
                        queue.append((r[3], cur_depth + 1))
            entities = []
            for name in visited:
                row = con.execute(
                    "SELECT file, name, type, description, line_start, line_end, snippet FROM entities "
                    "WHERE name = ? "
                    "ORDER BY CASE WHEN line_start IS NULL THEN 1 ELSE 0 END, file, line_start "
                    "LIMIT 1",
                    (name,),
                ).fetchone()
                if row:
                    entities.append({
                        "file": row[0], "name": row[1], "type": row[2], "description": row[3],
                        "line_start": row[4], "line_end": row[5], "snippet": row[6] or "",
                    })
        entities.sort(key=lambda e: (e.get("file", ""), e.get("line_start") or 0, e.get("name", "")))
        all_relations.sort(key=lambda r: (r.get("file", ""), r.get("from", ""), r.get("relation", ""), r.get("to", "")))
        return {
            "entities": entities,
            "relations": all_relations,
            "truncated_nodes": truncated_nodes,
            "skipped_hubs": skipped_hubs,
        }

    def get_stats(self) -> dict:
        with sqlite3.connect(self.db_path) as con:
            e = con.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            r = con.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
            f = con.execute("SELECT COUNT(*) FROM file_meta").fetchone()[0]
            types = con.execute(
                "SELECT type, COUNT(*) FROM entities GROUP BY type ORDER BY COUNT(*) DESC"
            ).fetchall()
            try:
                cache_rows = con.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()[0]
            except sqlite3.OperationalError:
                cache_rows = 0
        out = {"files": f, "entities": e, "relations": r, "by_type": {t: c for t, c in types}}
        out["embedding_cache_rows"] = cache_rows
        if hasattr(self, "_last_cache_total") and self._last_cache_total:
            out["last_cache_hit_rate"] = round(
                self._last_cache_hits / self._last_cache_total, 4
            )
        return out

    def explain_file(self, rel_path: str, top_callers: int = 5) -> dict:
        """One-shot view of a file: defined entities, deps, and who uses them."""
        defined = self.get_file_entities(rel_path)
        deps = self.get_file_deps(rel_path)
        # For each top-level definition, find external callers/usages.
        used_by: list[dict] = []
        for ent in defined:
            if ent["type"] not in {"function", "method", "class", "component", "interface"}:
                continue
            usages = [u for u in self.find_usages(ent["name"]) if u["file"] != rel_path]
            if usages:
                used_by.append({
                    "name": ent["name"],
                    "type": ent["type"],
                    "callers": usages[:top_callers],
                    "total": len(usages),
                })
        return {"file": rel_path, "entities": defined, "deps": deps, "used_by": used_by}

    def visualize(self, output_path: Path, title: Optional[str] = None, module_depth: int = 3) -> dict:
        """Build the per-project HTML graph viewer. Returns metadata dict
        with ``output_path`` and counts. Caller is responsible for opening
        the file in a browser.
        """
        from .visualizer import build_visualization_data, render_html
        data = build_visualization_data(
            db_path=self.db_path,
            project_root=self.project_root,
            module_depth=module_depth,
        )
        html = render_html(data, title or f"Code graph — {self.project_root.name}")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
        return {
            "output_path": str(output_path),
            "modules": data["stats"]["modules"],
            "files": data["stats"]["files"],
            "relations": data["stats"]["relations"],
        }

    def clear(self) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.executescript("DELETE FROM entities; DELETE FROM relations; DELETE FROM file_meta;")
        # SQLite keeps freed pages for reuse — DELETE alone won't shrink
        # the .db file. VACUUM rewrites the database to compact form, but
        # it can't run inside a transaction, so we open a separate
        # autocommit connection.
        try:
            con = sqlite3.connect(self.db_path, isolation_level=None)
            try:
                con.execute("VACUUM")
            finally:
                con.close()
        except Exception as e:
            logger.warning("VACUUM after clear failed: %s", e)
        self.faiss_index = None
        self.faiss_names = []
        self._faiss_dirty = False
        logger.info("Code graph cleared")
