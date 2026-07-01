# AGENTS.md

## Project Overview

- **Tech Stack:** Python 3.11, scipy, matplotlib    
- **Architecture:** code lives under `src/`; data in `data/`; docs in `docs`
- **Primary Goal:** Algorithm for use in a processing workflows that require deduplication of tracked particles
 

## Dependency Rules

- **Environment:** Use a simple venv environment.
- **Runtime:**  set `PYTHONPATH=src:.` 
- **External packages:** Prefer common libraries (`scikit-learn', 'numpy`, `pandas`, `opencv`, `opencv-headless`). 

## Testing & Verification

- No formal test suite or CI test job as this is a primarily a sandbox for developing an algorithm.  This may change over time. 

## Commits

**Style:** Use [Conventional Commits](https://www.conventionalcommits.org/) compatible with `python-semantic-release` (angular parser).