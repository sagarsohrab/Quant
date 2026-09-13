# Deploy to GitHub

A step-by-step, narrative guide for pushing this repo to GitHub as a public
showcase. Nothing here is run automatically — you execute these steps yourself.

`[REPO_URL]` is a placeholder you'll fill with the actual URL GitHub gives you.

---

## 0. Before you start

- You are in `/Users/sagar.sohrab/Quant`.
- The repo is already `git init`-ed on branch `main`, but has **no commits yet**.
- Large cached data (`data/*.parquet`) and the `.venv` environment are already
  excluded via `.gitignore`, so they will **not** be pushed. Double-check with:

```bash
git status          # shows what will be committed
```

---

## 1. Create a private repo on github.com

1. Go to <https://github.com/new> in your browser.
2. Give it a name and a short description (e.g. "Systematic quant research
   pipeline: data → features → models → backtest → live → ML ops → LLM").
3. **Leave it Private** for now — you can make it public later (see step 6).
4. **Do NOT** check "Add a README", "Add a .gitignore", or "Add a license".
   Your repo already has its own files. You want an **empty** repo.
5. Click **Create repository**.
6. GitHub will show you a `git remote add origin ...` line. Copy the URL —
   that's `[REPO_URL]`.

---

## 2. Authenticate the GitHub CLI

Use either of the following:

**Option A — browser SSO (recommended):**

```bash
gh auth login
```

Follow the prompts: choose GitHub.com, choose HTTPS, say Yes to authenticate
via your browser, and follow the login flow in the browser that opens.

**Option B — personal access token:**

```bash
gh auth login --with-token <<< "YOUR_GITHUB_TOKEN"
```

Create the token at <https://github.com/settings/tokens> with `repo` scope.

Verify authentication:

```bash
gh auth status
```

---

## 3. Commit everything (first commit)

Stage all files and create the initial commit:

```bash
git add -A
git commit -m "Initial commit: full-stack quant research pipeline"
```

Optional — verify what was staged before committing:

```bash
git status          # confirm .venv and data/*.parquet are absent
git diff --cached --stat
```

Data files are large; confirm none snuck in:

```bash
git ls-files | grep -E "\.(parquet|pkl)$" || echo "no large artifacts tracked"
```

---

## 4. Point the repo at your new GitHub remote

Rename the local branch (already `main`, but safe to be explicit) and add the
remote from step 1:

```bash
git branch -M main
git remote add origin https://github.com/sagarsohrab/Quant.git
```

Verify the remote is set:

```bash
git remote -v
```

---

## 5. Push

```bash
git push -u origin main
```

`-u` sets the upstream so future pushes are just `git push`. Your code and
CI workflow are now on GitHub. GitHub Actions will run CI on this push — you
can watch it under the repo's **Actions** tab.

---

## 6. Switch public vs. private

On the repo page:

1. Go to **Settings → General** (or **Danger Zone** at the bottom).
2. Under **Danger Zone**, find **Change repository visibility**.
3. Choose **Make public** (or **Make private**) and click it.
4. GitHub may ask you to confirm the repo's name.

> Tip: keep it **private** until you've confirmed CI is green and the README
> renders, then flip it public for your portfolio.

---

## 7. Make the README / repo presentable

- The **root `README.md`** is your personal "learning journey" — it renders
  automatically on the repo page and shows your progress tracker.
- A recruiter-facing summary lives in `docs/RECRUITER_README.md`. To feature it
  as the front-page README instead, copy it over the root README:

```bash
cp docs/RECRUITER_README.md README.md
```

- CI status badge (once pushed) — paste into any README:

```md
![CI](https://github.com/USERNAME/REPONAME/actions/workflows/ci.yml/badge.svg)
```

- To see the rendered site, GitHub renders Markdown on every repo page
  automatically — no extra steps. Just open `https://github.com/USERNAME/REPONAME`.

---

## 8. Share on resume / LinkedIn

- Add the repo URL (now `https://github.com/USERNAME/REPONAME`) to your resume
  and LinkedIn "Projects" or "Recent" section.
- Link the README text you're proudest of (walk-forward, no-lookahead, ML ops).
- Consider adding a link to a report `.pdf` or the `docs/apply/` materials as
  supporting evidence describing what you measured and what you learned.

---

## Checklist

- [ ] Private repo created and empty (no auto-README/.gitignore/license)
- [ ] Repo URL filled in: https://github.com/sagarsohrab/Quant.git
- [ ] `gh auth status` confirms authentication
- [ ] `git add -A` done
- [ ] `git commit -m "..."` done
- [ ] `git branch -M main` done
- [ ] `git remote add origin https://github.com/sagarsohrab/Quant.git` done; `git remote -v` shows origin
- [ ] `git ls-files` confirms **no** `.parquet`/`.pkl`/`.venv` committed
- [ ] `git push -u origin main` succeeded
- [ ] GitHub Actions CI ran green (Actions tab)
- [ ] README renders correctly on the repo page
- [ ] Repo switched public (Settings → Danger Zone)
- [ ] Link added to resume / LinkedIn
