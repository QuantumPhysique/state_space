Prepare a release: version, changelog, PR, and after the merge the tag that publishes to pub.dev.

Usage: `/release` or `/release <X.Y.Z>` — version: `$ARGUMENTS` if given, else proposed in step 2.

## Step 1 — Check

1. `git fetch origin main --tags`, `git switch main`, `git pull --ff-only` — stop on uncommitted changes to tracked files.
2. `## Unreleased` in `CHANGELOG.md` exists and is not empty — otherwise there is nothing to release; stop.
3. `version:` in `pubspec.yaml`, the latest version heading in `CHANGELOG.md` and the latest tag (`git tag --sort=-v:refname | head -1`) must name the same version.

## Step 2 — Version

Propose the next version from `## Unreleased`, by [SemVer](https://semver.org) as pub reads it: below 1.0, the minor version is the one that breaks.

| `## Unreleased` holds | Below 1.0 | From 1.0 |
|---|---|---|
| A change that breaks callers | minor | major |
| New API | minor | minor |
| Only fixes | patch | patch |

Show the proposal with the line that decided it and ask: this version, or a different one? Also ask whether the release gets an intro line, as 0.1.0 has `First release.` — never invent one.

## Step 3 — Prepare

Only after the version is confirmed. On a branch `release/prepare_vX.Y.Z` off `main`:

1. `pubspec.yaml`: `version: X.Y.Z`.
2. `CHANGELOG.md`: bring every bullet under `## Unreleased` in line with the CLAUDE.md Changelog section, then rename the heading to `## X.Y.Z`, the intro line, if any, directly below it. No new `## Unreleased`; the next change adds it.
3. Every `state_space: ^<previous>` that `grep -rn "state_space: ^" README.md doc/` finds becomes `state_space: ^X.Y.Z`.
4. The gate from CLAUDE.md, and `dart pub publish --dry-run`, which must end in `Package has 0 warnings.`
5. Show the new changelog section; after approval commit everything as `chore: prepare vX.Y.Z` (via `/commit`) and open the PR (via `/pr`).

## Step 4 — Publish

Only once the PR is merged, and after asking:

```bash
git switch main
git pull --ff-only
git tag vX.Y.Z            # lightweight, on the merge commit, like v0.1.0
git push origin vX.Y.Z
```

The tag push runs `publish.yaml`, which publishes to pub.dev with a token from GitHub; nothing is published from this machine. Follow the run with `gh run watch <id>` (`gh run list --workflow publish.yaml --limit 1` for the id), then check that `curl -s https://pub.dev/api/packages/state_space` lists `X.Y.Z` as `latest`.

If the run fails, show its log and stop. A version pub.dev did not accept can be tagged again once the cause is fixed; one it accepted is final.

Never create or push a version tag any other way, and never run `dart pub publish` by hand.

## Step 5 — After publishing

If `../trale/app/pubspec.yaml` takes `state_space` by git, say that trale can switch to `^X.Y.Z` now — a change in trale, not here.
