# Migration Guide: Creating a New GitHub Repository

This guide explains how to create a new GitHub repository with all the code from this project.

## Option 1: Create New Repository via GitHub Web Interface

### Step 1: Create the New Repository
1. Go to https://github.com/new
2. Enter repository name (e.g., `CPSC491Fall2025-MyVersion`)
3. Choose visibility: **Public** or **Private**
4. **Do NOT** initialize with README, .gitignore, or license (we already have these)
5. Click "Create repository"

### Step 2: Push Code to New Repository
After creating the repository, run these commands in your local clone:

```bash
# Add the new repository as a remote
git remote add new-repo https://github.com/YOUR-USERNAME/YOUR-NEW-REPO-NAME.git

# Push all branches to the new repository
git push new-repo --all

# Push all tags to the new repository (if any)
git push new-repo --tags
```

Replace:
- `YOUR-USERNAME` with your GitHub username
- `YOUR-NEW-REPO-NAME` with your new repository name

### Step 3: Verify the Migration
1. Visit your new repository on GitHub
2. Check that all files are present
3. Verify the README.md displays correctly
4. Check that the .gitignore is working (chroma_fcc_storage/ should not be tracked)

## Option 2: Fork the Repository

If you want to maintain a connection to the original repository:

1. Go to https://github.com/JeffGoldthorp/CPSC491Fall2025
2. Click the "Fork" button in the top right
3. Choose your account as the destination
4. Optionally rename the repository during the fork process

## Option 3: Create Clean Copy (No Git History)

If you want to start fresh without the git history:

```bash
# Create a new directory
mkdir CPSC491Fall2025-Fresh
cd CPSC491Fall2025-Fresh

# Copy all files except .git directory
rsync -av --progress /path/to/CPSC491Fall2025/ . --exclude .git

# Initialize new git repository
git init
git add .
git commit -m "Initial commit"

# Create new repo on GitHub, then:
git remote add origin https://github.com/YOUR-USERNAME/YOUR-NEW-REPO-NAME.git
git branch -M main
git push -u origin main
```

## What's Included in This Repository

### Core Files
- ✅ **VectordB/ChromaChat.py** - Main chat interface with all features
- ✅ **VectordB/ChromaDB.py** - Database utilities
- ✅ **config.py** - Configuration management
- ✅ **compare_models.py** - Model comparison tools
- ✅ **requirements.txt** - All Python dependencies

### Documentation
- ✅ **README.md** - Comprehensive project documentation
- ✅ **IMPROVEMENT_GUIDE.md** - Model improvement strategies
- ✅ **MIGRATION_GUIDE.md** - This file

### Data & Training
- ✅ **doc/** - Documentation and text files
- ✅ **Front-End/** - Frontend and training scripts
- ✅ **improve_training_data.py** - Training data enhancement
- ✅ **improved_training_examples.jsonl** - Enhanced training examples

### Configuration
- ✅ **.gitignore** - Properly configured to exclude:
  - Virtual environments (.venv, .venv/, venv/, env/)
  - Environment files (.env)
  - Python cache (__pycache__, *.pyc)
  - ChromaDB storage (chroma_fcc_storage/)
  - Vim swap files (*.swp, *.swo, *~)

## Files NOT Included (by design)

These files are excluded via .gitignore and should remain local:

- ❌ `.env` - Contains your API keys (create new in destination)
- ❌ `chroma_fcc_storage/` - Local ChromaDB data (will be regenerated)
- ❌ `__pycache__/` - Python cache (auto-generated)
- ❌ `.venv/`, `venv/`, or `env/` - Virtual environment directories (create new with `pip install -r requirements.txt`)

## After Migration: Setup Steps

Once you've created the new repository, follow these steps:

1. **Clone the new repository**:
   ```bash
   git clone https://github.com/YOUR-USERNAME/YOUR-NEW-REPO-NAME.git
   cd YOUR-NEW-REPO-NAME
   ```

2. **Create virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Create .env file**:
   ```bash
   cat > .env << 'EOF'
   OPENAI_API_KEY=your_openai_key_here
   SERPAPI_API_KEY=your_serpapi_key_here
   CHROMA_PERSIST_PATH=./chroma_fcc_storage
   CHROMA_COLLECTION=fcc_documents
   EOF
   ```

5. **Run ChromaChat**:
   ```bash
   python3 VectordB/ChromaChat.py
   ```

## Troubleshooting

### Issue: "Permission denied" when pushing
**Solution**: Make sure you have write access to the new repository and your SSH keys are configured properly.

### Issue: ChromaDB storage not working
**Solution**: Check that `chroma_fcc_storage/` is listed in .gitignore. It will be created automatically when you first run ChromaChat.py.

### Issue: Missing API keys error
**Solution**: Create a `.env` file in the project root with your OpenAI and SerpAPI keys (see step 4 above).

### Issue: Import errors
**Solution**: Make sure you've activated the virtual environment and installed all requirements:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Repository Maintenance Tips

1. **Keep .env out of git**: Never commit API keys or sensitive data
2. **Update .gitignore**: Add any new local-only files or directories
3. **Document changes**: Update README.md when adding new features
4. **Regular commits**: Commit frequently with descriptive messages
5. **Use branches**: Create feature branches for major changes

## Questions?

If you encounter issues during migration, check:
- GitHub's documentation: https://docs.github.com/
- Git documentation: https://git-scm.com/doc
- This project's README.md for setup instructions
