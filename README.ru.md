[English](README.md) · **Русский**

# mcp-rag

Self-hosted MCP-сервер, который превращает любое дерево исходников в **граф знаний по коду** + **гибридный семантический + лексический поиск** + **интерактивную HTML-визуализацию**. Заточен под Claude Code, но говорит на чистом MCP — подключится к любому MCP-клиенту (Cursor, Continue, кастомные агенты).

Цель: дать AI-агенту ту же кросс-файловую структурную осведомлённость, что есть у IDE — «кто вызывает эту функцию», «где определён этот компонент», «что похоже на этот хелпер» — без перечитывания проекта на каждый промпт.

---

## Зачем это

Из коробки Claude Code (и большинство coding-агентов) читают файлы по запросу через `Read`/`Grep`/`Glob`. Это покрывает ~90% работы, но ломается на:

- **Концептуальных запросах** — «где у нас auth flow?» — Grep'у нужны точные токены, Read жрёт контекст.
- **Refactor scope** — «если переименовать `Button`, что сломается?» — нужен структурный usage tracking, а не текстовый поиск.
- **Code discovery** — «есть ли уже хелпер, который делает X?» — нужно семантическое сопоставление, не паттерны имён.
- **Multi-language проектах** — Angular + NestJS + Python + FastAPI — у каждого свои конвенции, но кросс-вопросы («какой фронт зовёт этот endpoint?») требуют единого индекса.

mcp-rag заранее считает **граф проекта** (tree-sitter для 10+ языков, regex-fallback для остального) и **chunk-индекс** (BM25 + плотные эмбеддинги через `bge-m3`, cross-encoder rerank). Результат: набор MCP-тулов, где один вызов заменяет 5–10 чтений файлов.

---

## Quick start

### Установка

```bash
git clone <your-fork>
cd mcp-rag
python -m venv .venv
.venv/Scripts/activate           # или `source .venv/bin/activate`
pip install -e ".[llm,dev]"
```

### Подключение к Claude Code

```bash
claude mcp add rag --scope user -- /abs/path/to/.venv/bin/mcp-rag
```

Или правишь `~/.claude.json` напрямую:

```json
{
  "mcpServers": {
    "rag": {
      "type": "stdio",
      "command": "/abs/path/to/.venv/bin/mcp-rag"
    }
  }
}
```

Сервер берёт проект из `cwd` Claude Code на старте. Перезапусти Claude Code из корня проекта — и всё. Первый вызов `graph_*` авто-соберёт граф.

### Первые шаги

```
graph_build               # или просто вызови любой graph_* — авто-build при пустом графе
graph_explain src/Foo.tsx # карточка файла одним вызовом
search_code "auth flow"   # концептуальный поиск по тексту кода
graph_find_usages User    # точный refactor scope по имени
graph_visualize           # генерит HTML, открывай в браузере
```

---

## Конфигурация

Всё через environment-переменные. Задаются в MCP-конфиге в блоке `"env"` или в shell.

### Проект и хранилище

| Переменная | Default | Что делает |
|---|---|---|
| `MCP_RAG_PROJECT` | cwd | Корень проекта для индексации. Перебивается CLI-флагом `--project`. |
| `MCP_RAG_STORAGE` | `~/.mcp-rag` | Где живут граф / faiss / cache / models. |

### Embedder

| Переменная | Default | Что делает |
|---|---|---|
| `MCP_RAG_EMBED_MODEL` | `BAAI/bge-m3` | sentence-transformers model id. Любая совместимая работает. |
| `MCP_RAG_RERANKER_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder для rerank в `search_code`. |
| `MCP_RAG_RERANK` | `1` | Поставь `0` чтобы отключить rerank (быстрее, но шумнее). |
| `MCP_RAG_DEVICE` | auto | `cuda`/`mps`/`cpu`. Авто-детектится; force-override здесь. |

### LLM-extractor (опционально)

Tree-sitter + regex покрывают типичный кейс. Эти переменные нужны если хочешь fallback для редких языков:

| Переменная | Обязательно | Что делает |
|---|---|---|
| `MCP_RAG_LLM_BASE_URL` | да | OpenAI-совместимый Chat Completions endpoint. |
| `MCP_RAG_LLM_API_KEY` | да | Bearer token. |
| `MCP_RAG_LLM_MODEL` | нет, default `deepseek-chat` | Model id. |

### Поведение

| Переменная | Default | Что делает |
|---|---|---|
| `MCP_RAG_NO_WATCH` | unset | `1` чтобы выключить FS-watcher (без авто-реиндекса при сохранении). |
| `MCP_RAG_NO_MEMORY` | unset | `1` чтобы скрыть `memory_*` тулы — полезно если у хоста есть своя память (Claude Code's `~/.claude/memory/` и т.п.). |
| `MCP_RAG_LOG` | `INFO` | `DEBUG`/`INFO`/`WARNING`/`ERROR`. |

### Пример: полный конфиг для Claude Code

JSON не поддерживает комменты, но можно положить рядом с `env` поле
`_available_env_options` — справочник со всеми вариантами. Claude Code
неизвестные поля игнорирует. (Caveat: `claude mcp add/remove` может
переписать файл и стереть неизвестные ключи — держи копию где-то ещё
если это критично.)

```json
{
  "mcpServers": {
    "rag": {
      "type": "stdio",
      "command": "D:\\Projects\\mcp-rag\\.venv\\Scripts\\mcp-rag.exe",
      "args": [],
      "env": {
        "MCP_RAG_EMBED_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
        "MCP_RAG_RERANKER_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "MCP_RAG_NO_MEMORY": "1"
      },
      "_available_env_options": {
        "MCP_RAG_PROJECT":        "Корень проекта (default: cwd)",
        "MCP_RAG_STORAGE":        "Корень storage (default: ~/.mcp-rag)",
        "MCP_RAG_EMBED_MODEL":    "Sentence-transformers id (default: BAAI/bge-m3). Lightweight: sentence-transformers/all-MiniLM-L6-v2",
        "MCP_RAG_RERANKER_MODEL": "Cross-encoder для rerank (default: BAAI/bge-reranker-base). Lightweight: cross-encoder/ms-marco-MiniLM-L-6-v2",
        "MCP_RAG_RERANK":         "0 чтобы выключить cross-encoder rerank (default: 1)",
        "MCP_RAG_DEVICE":         "cuda / mps / cpu (default: auto)",
        "MCP_RAG_NO_WATCH":       "1 чтобы выключить filesystem watcher",
        "MCP_RAG_NO_MEMORY":      "1 чтобы скрыть memory_* тулы (15 вместо 20)",
        "MCP_RAG_LOG":            "DEBUG / INFO / WARNING / ERROR (default: INFO)",
        "MCP_RAG_LLM_BASE_URL":   "OpenAI-совместимый endpoint для LLM fallback",
        "MCP_RAG_LLM_API_KEY":    "Bearer token для LLM fallback",
        "MCP_RAG_LLM_MODEL":      "Model id для LLM fallback (default: deepseek-chat)"
      }
    }
  }
}
```

### Пример: с LLM fallback

Claude Code передаёт значения `env` в subprocess **как есть** — никаких `${VAR}`-подстановок не делает — поэтому API-ключ должен быть реальной строкой, не ссылкой. Либо инлайнишь ключ (и держишь файл вне git), либо оборачиваешь launcher в shell-скрипт, читающий из `.env`:

```json
{
  "mcpServers": {
    "rag": {
      "type": "stdio",
      "command": "/usr/local/bin/mcp-rag",
      "env": {
        "MCP_RAG_LLM_BASE_URL": "https://api.deepseek.com/v1",
        "MCP_RAG_LLM_API_KEY": "sk-replace-with-your-key",
        "MCP_RAG_LLM_MODEL": "deepseek-chat"
      }
    }
  }
}
```

Альтернатива через wrapper-скрипт на Windows (`start-mcp-rag.bat`):

```bat
@echo off
set MCP_RAG_LLM_BASE_URL=https://api.deepseek.com/v1
set MCP_RAG_LLM_API_KEY=sk-...
set MCP_RAG_LLM_MODEL=deepseek-chat
"D:\Projects\mcp-rag\.venv\Scripts\mcp-rag.exe" %*
```

```json
{
  "mcpServers": {
    "rag": {
      "type": "stdio",
      "command": "cmd.exe",
      "args": ["/c", "D:\\path\\start-mcp-rag.bat"]
    }
  }
}
```

---

## Tool reference

### Daily-use (90% пользы)

| Tool | Что делает |
|---|---|
| `search_code` | Гибрид BM25 + dense + cross-encoder + IDF-взвешенный literal-bonus по тексту проекта. RU/EN. Концептуальные запросы, поиск паттернов, когда не знаешь точных имён. |
| `graph_explain` | Досье на файл одним вызовом: declared entities + dependency map + external callers. Заменяет три других тула. |
| `graph_find_usages` | Каждое место, где используется entity по точному имени (calls, JSX, instantiations, inheritance). Использовать перед rename/refactor. |
| `graph_pending_files` | Файлы где граф разошёлся с диском (unindexed / stale / missing). Sanity check после правок. |
| `graph_stats` | Счётчики files / entities / relations + breakdown по типам. |

### Ситуативные

| Tool | Что делает |
|---|---|
| `graph_find_similar` | Семантически ближайшие сущности к anchor — детект дубликатов. Embed text это `name + outgoing relations + snippet`, поэтому структурные fingerprint'ы кластеризуются (все обёртки antd-компонентов оказываются рядом). |
| `graph_dead_code` | Функции/классы/компоненты, на которые не указывает ни одна relation. Параметр `exclude_paths` (globs) скрывает scaffolding. |
| `graph_get_subgraph` | BFS по relations вокруг сущности. Cap по узлу — общие имена типа `Layout`/`Header` дают много truncated nodes; надёжно работает на уникальных именах. |
| `graph_visualize` | Рендерит интерактивный HTML с тремя уровнями (modules → files → entities). Self-contained, vis-network с CDN. |
| `graph_build` / `graph_clear` / `graph_index_file` | Lifecycle. `graph_build` авто-вызывается на первом data-needing тулe при пустом графе. |

### Нишевые

| Tool | Что делает |
|---|---|
| `graph_search` | Найти entities по имени (substring match по таблице entities) с фильтром по типу. Кейс: «список классов с `Button` в имени». Для concept-поиска бери `search_code`; для точного refactor — `graph_find_usages`. |
| `graph_get_file_deps` / `graph_file_structure` | Подмножества `graph_explain`. Бери `graph_explain` если не нужна одна конкретная секция. |

### Memory (выключается через `MCP_RAG_NO_MEMORY=1`)

| Tool | Что делает |
|---|---|
| `memory_add` / `memory_search` / `memory_list` / `memory_delete` / `memory_clear` | Per-project memory store с гибридным поиском. Полезно для хостов без своей памяти. |

---

## Resources (`@`-attachable в Claude Code)

| URI | Что возвращает |
|---|---|
| `rag://overview` | Дайджест проекта: top-level структура, breakdown по типам entity, top-referenced symbols. |
| `rag://file/{path}` | Компактное досье файла (та же форма что `graph_explain`). |
| `rag://search/{query}` | Готовый блок результата `search_code`. |
| `rag://explain/{entity}` | Карточка entity: location, snippet, callers. |

---

## Архитектура

```
project source
      │
      ▼
┌─ extractors ────────────────────────────────────────┐
│  tree-sitter (10+ langs)                            │
│  regex (HTML, CSS, configs, JSX usages)             │
│  optional LLM fallback (OpenAI-compat)              │
└──────────────────┬──────────────────────────────────┘
                   ▼
   ┌────────────────────────────────┐
   │ entities + relations → SQLite  │
   │ chunks → diskcache             │
   └────┬──────────────┬────────────┘
        ▼              ▼
   ┌─────────┐   ┌────────────────┐
   │ graph_* │   │ search_code    │
   │ tools   │   │  bm25 +        │
   │         │   │  bge-m3 dense +│
   │ + FAISS │   │  bge-reranker +│
   │  index  │   │  literal bonus │
   └─────────┘   └────────────────┘
```

FS-watcher (watchdog) держит граф свежим между явными `graph_build`. Rebuild FAISS отложен — переэнкодить ~40k entities на каждое сохранение слишком дорого — поэтому similarity-тулы (`graph_find_similar`) триггерят rebuild по требованию, если установлен dirty-флаг.

---

## Hardware и производительность

Замерено на **RTX 5060 Ti 16 GB**, Windows 11, Python 3.13, проект 1931 файл / 42k entities / 50k relations.

- Полный graph build с bge-m3 + bf16: **~140с** end-to-end.
- Только rebuild FAISS: **~80с** на 42k entities.
- `search_code` query (после warmup): **~100–200 мс** включая cross-encoder rerank.
- File-watcher reindex: **~50–250 мс** на файл (debounce 1с).

Для CPU-only ставь `MCP_RAG_DEVICE=cpu` — работает, просто медленнее. Для Apple Silicon: `MCP_RAG_DEVICE=mps` (авто-детектится).

### Размер модели vs VRAM

| Модель | Params | VRAM (bf16) | Языки | Заметки |
|---|---|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | 23M | ~80 MB | EN only | Лёгкий escape-hatch — влезает куда угодно, на CPU работает с осмысленной скоростью, но без RU/multilingual. Бери если проект на английском и нужен минимальный footprint. |
| `BAAI/bge-m3` (default) | 568M | ~1.2 GB | 100+ | Encoder-only, безопасно на 16GB GPU, контекст 8k. Хороший RU/EN баланс, без префиксного гимнастика. |
| `Qwen/Qwen3-Embedding-0.6B` | 600M | нестабильно на 16GB | 100+ | Decoder-only LLM с KV-cache; на бумаге MTEB Multilingual выше, но OOM-ит на consumer GPU когда скармливаешь реальный корпус. Документирован как opt-in, не как рекомендация. |

### Preset: лёгкий стек (EN-only, CPU-friendly)

Пара: MiniLM bi-encoder для retrieval + MS MARCO cross-encoder для rerank.
Обе модели ~80 MB, обе с разумной скоростью на CPU, обе под английский.
Drop-in конфиг:

```json
"env": {
  "MCP_RAG_EMBED_MODEL":    "sentence-transformers/all-MiniLM-L6-v2",
  "MCP_RAG_RERANKER_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2"
}
```

Бери это для английских репо когда нет GPU или нужен минимальный
footprint. Multilingual теряется, но качество поиска по английскому
коду остаётся приличным — cross-encoder rerank вытаскивает top-K из
шумного MiniLM-выхода.

---

## Storage layout

```
~/.mcp-rag/
├── models/
│   └── BAAI_bge-m3/                           # скачивается один раз
└── projects/
    └── <project_name>_<short-hash>/
        ├── graph/
        │   └── graph.db                       # SQLite: entities, relations, file_meta
        ├── retriever/
        │   └── cache.db                       # diskcache: bm25 + faiss + chunks
        ├── memory/
        │   ├── memories.json
        │   └── embeddings_cache/
        └── graph.html                         # вывод graph_visualize
```

Корень проекта хешируется — два чекаута одного репозитория по разным путям получат отдельные хранилища.

---

## Языки

Tree-sitter (структурное извлечение): Python, JS/TS/TSX/JSX, Vue, Svelte, Astro, Go, Rust, Java, C#, C/C++, PHP, Ruby.

Regex-extractors: HTML, CSS/SCSS/LESS/Sass, JSON, YAML, TOML, Jinja-шаблоны, Godot/GDScript.

Всё остальное идёт в LLM fallback (если настроен), иначе остаётся только file-level entity.

---

## CLI

```
mcp-rag --help
  --project PATH        корень проекта (default: cwd)
  --storage PATH        корень storage (default: ~/.mcp-rag)
  --log-level LEVEL     DEBUG | INFO | WARNING | ERROR
  --no-watch            выключить filesystem watcher
```

Имена CLI-флагов те же, что у env-переменных, без префикса `MCP_RAG_`.

---

## Разработка

```bash
pip install -e ".[llm,dev]"          # ruff + pytest extras
ruff check src/
pytest                                 # тесты пока в TODO; см. TODO.md
```

Smoke-проверка MCP-handshake без Claude Code:

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"smoke","version":"0.1"}}}' | mcp-rag --project /path/to/repo
```

---

## License

MIT.
