# Setup (macOS)

This repo currently requires:

- **Python 3.11+** (the `ai-data-analyst` backend uses Python 3.10+ syntax and declares 3.11+)
- **Node >= 20.9.0** (required by the installed `next@16.0.7` engine in `ai-data-analyst/frontend`)

## 1) Install Node (recommended: nvm)

```bash
brew install nvm
mkdir -p ~/.nvm
```

Add this to your shell profile (`~/.zshrc`):

```bash
export NVM_DIR="$HOME/.nvm"
source "$(brew --prefix nvm)/nvm.sh"
```

Then:

```bash
cd /Users/laxmi/Downloads/data_analyst
nvm install
nvm use
node -v
```

## 2) Install Python 3.11+

Option A (Homebrew):

```bash
brew install python@3.11
python3.11 --version
```

Option B (pyenv):

```bash
brew install pyenv
pyenv install 3.11.9
pyenv local 3.11.9
python --version
```

## 3) Optional: PostgreSQL (for full `ai-data-analyst` functionality)

The analyst backend is designed around Postgres + SQLAlchemy async.

```bash
brew install postgresql@14
brew services start postgresql@14
createdb ai_data_analyst
```

Then copy and edit:

```bash
cd ai-data-analyst/backend
cp .env.example .env
```

## 4) Run

```bash
cd /Users/laxmi/Downloads/data_analyst
make setup
make dev-analyst
```

Or:

```bash
make dev-validator
```

