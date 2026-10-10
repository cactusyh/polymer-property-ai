# GitHub and data governance

The local repository has an `origin` remote at `git@github.com:cactusyh/polymer-property-ai.git`.
This phase does not push automatically. The feature branch is
`codex/phase-6a-unified-benchmark`.

Safe commits contain source code, tests, YAML/JSON configuration, and concise
reports. Raw property files, Parquet registries, trajectories/topologies, logs,
results, caches, model checkpoints, credentials, and private datasets remain
outside GitHub through `.gitignore` and must be transferred through an approved
private data channel. Review the staged file list before every commit.

If authenticated access is available, push manually after review:

```bash
git push -u origin codex/phase-6a-unified-benchmark
```

To create a new private GitHub repository instead of using the configured remote,
create it in GitHub's UI, then run (with the actual private URL):

```bash
git remote add private git@github.com:ORG/REPO.git
git push -u private codex/phase-6a-unified-benchmark
```
