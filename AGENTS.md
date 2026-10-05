# AGENTS.md

## Project Overview

- **Tech Stack:** Python 3.11, scipy, matplotlib, Poetry, semantic-release on `main`
- **Architecture:** code lives under `src/`; data in `data/`; docs in `docs`
- **Primary Goal:** Algorithm for use in a processing workflows that require deduplication of tracked particles
 

## Dependency Rules

- **Environment:** Use a simple venv environment.
- **Runtime:**  set `PYTHONPATH=src:.` 
- **External packages:** Prefer common libraries (`scikit-learn', 'numpy`, `pandas`, `opencv`, `opencv-headless`). 

## Testing & Verification

- No formal test suite or CI test job as this is a primarily a sandbox for developing an algorithm.  This may change over time. 

## Commits (semantic-release)

Use **Angular-style** commit messages so `python-semantic-release` can version correctly (`pyproject.toml`).

**Format:** `<type>[optional scope]: <description>`

**Allowed types:** `feat`, `fix`, `perf`, `docs`, `build`, `ci`, `chore`, `style`, `refactor`, `test`

| Type | Release impact |
|------|----------------|
| `feat` | Minor bump |
| `fix`, `perf` | Patch bump |
| Others | Typically no version bump (see changelog exclude patterns) |

**Examples:**

- `feat: add cross-burst time gate option to hungarian matcher`
- `fix(dedup): propagate track identity through transitive matches`
- `docs: expand README usage section`

Do **not** use ad-hoc prefixes (`Update`, `WIP`, version-only messages) for changes that should ship.