# GitHub Workflow (Understand What You Are Doing)

This repo is already a git repo (`neural-analyst/.git`) and has an `origin` remote configured.

## 1) Check where you are and what changed

```powershell
cd neural-analyst
git status -sb
git remote -v
```

- `git status` tells you which files are modified / new / deleted.
- `git remote -v` tells you which GitHub repo you will push to.

## 2) Create a branch (recommended)

Working on a branch keeps `main` clean and makes PRs easy:

```powershell
git switch -c feature/dataset-processing
```

## 3) Stage changes (choose what you want to include)

Stage everything:

```powershell
git add -A
```

Or stage specific files:

```powershell
git add ai-data-analyst/backend/app
git add ai-data-analyst/frontend/src
```

Staging is the "I want these exact changes in the next commit" step.

## 4) Commit (make a checkpoint)

```powershell
git commit -m "Improve dataset processing + UI polling"
```

Commits are like save-points you can refer back to or revert.

## 5) Push your branch to GitHub

```powershell
git push -u origin feature/dataset-processing
```

- `-u` sets upstream so future pushes can be just `git push`.

## 6) Open a Pull Request (PR)

On GitHub, create a PR from your branch into `main`.

Why PRs matter:

- They keep history clean.
- You can review changes before merging.
- CI can run on the PR.

## 7) Keep your branch up to date (when others push to main)

```powershell
git fetch origin
git rebase origin/main
```

Then push again:

```powershell
git push --force-with-lease
```

`--force-with-lease` is safer than `--force` (it refuses to overwrite remote work you do not have locally).

## 8) Secrets: what NOT to commit

Never commit:

- `.env` files with real keys
- API keys / tokens
- database passwords

This repo keeps templates like:

- `ai-data-analyst/backend/.env.example`

Use those as "what variables exist", and keep real secrets in your local `.env`.

