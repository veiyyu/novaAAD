# How to put this in a repo (no git experience needed)

This whole thing lives in **one self-contained folder** (`neuro-attention/`). It adds
new files only and touches nothing your teammates wrote, so it can't overwrite or
break their work. Pick whichever path below is easiest for you.

---

## Easiest: your own repo, no command line (drag-and-drop)

Good for "worst case, I'll make my own and share it."

1. Go to <https://github.com/new>. Give it a name (e.g. `neuro-attention`), leave the
   defaults, click **Create repository**.
2. On the new repo page, click **"uploading an existing file"** (in the "quick setup"
   box), or **Add file → Upload files**.
3. Unzip the folder I sent you, then **drag the `neuro-attention` folder's contents**
   into the browser. Wait for them to finish uploading.
4. Type a short message like `Add AAD viability + neuro-steered demo` and click
   **Commit changes**.
5. Share it: **Settings → Collaborators → Add people** (type each teammate's GitHub
   username), or make it public under **Settings → General → Change visibility** and
   send them the link.

That's it — no terminal, no git commands.

---

## To add it to your team's existing NOVA repo (safe, via a branch)

A **branch** is a private copy of the repo where your changes live until the team
chooses to merge them. Working on a branch is what guarantees you don't disturb
anyone's work on `main`.

### With GitHub Desktop (no command line)

1. Install GitHub Desktop (desktop.github.com) and sign in.
2. **File → Clone repository →** pick `KineticJetIce245/NOVA2026` → Clone.
3. In GitHub Desktop: **Current Branch → New Branch**, name it `aad-viability`,
   click Create.
4. In Finder, copy the `neuro-attention` folder into the cloned repo folder (put it
   wherever fits — the repo root or under `src/`).
5. Back in GitHub Desktop you'll see the new files listed. Type a summary
   (`Add AAD viability + neuro-steered demo`) and click **Commit to aad-viability**.
6. Click **Push origin**, then **Create Pull Request** (opens the browser). Submit it.
   Your teammates review and merge when ready — `main` is untouched until they do.

### With the command line

```bash
# clone the repo if you don't have it yet (skip if you already do):
git clone https://github.com/KineticJetIce245/NOVA2026.git
cd NOVA2026

git checkout main && git pull            # start from the latest team code
git checkout -b aad-viability            # make your own branch

# copy the neuro-attention folder into the repo here (Finder, or:)
cp -R /path/to/neuro-attention .

git add neuro-attention                  # stage only your new folder
git commit -m "Add AAD viability analysis + real-time neuro-steered demo"
git push -u origin aad-viability         # upload your branch

# then open the printed link to create a Pull Request on GitHub
```

If `git push` asks you to sign in, use your GitHub username and a Personal Access
Token as the password (GitHub → Settings → Developer settings → Personal access
tokens). GitHub Desktop avoids this entirely.

---

## What NOT to commit (already handled)

The `.gitignore` in the folder keeps the large dataset out of git — never commit the
`.mat` files, `stimuli/`, or `*.npz`. They're multi-hundred-MB and belong on Zenodo,
not in the repo. Everything else (code, README, PROJECT.md, small result images) is
fine to commit.

---

## Where to put the folder inside the repo?

Anywhere that doesn't collide with existing files. The repo root or a `src/` subfolder
both work. Since it's one uniquely-named folder, there's no risk of clashing with a
teammate's file.
