# Push To GitHub

This folder is set up to push to:

`git@github.com:naufalkmd/PAnDa.git`

## First push

1. Create an empty GitHub repository named `PAnDa` under `naufalkmd`.
2. Put your project files into this folder.
3. Run:

```bash
git add .
git commit -m "Initial commit"
git push -u origin main
```

## If you prefer HTTPS instead of SSH

```bash
git remote set-url origin https://github.com/naufalkmd/PAnDa.git
git push -u origin main
```
