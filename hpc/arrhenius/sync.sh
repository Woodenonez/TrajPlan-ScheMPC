#!/usr/bin/env bash
# Sync this working tree to and from the Arrhenius project storage.
#
# Arrhenius asks for a password and a one-time verification code on every SSH
# connection. This script avoids repeating that by using OpenSSH connection
# multiplexing: the first command opens one authenticated master connection,
# which the ~/.ssh/config entry for "arrhenius" keeps open for eight hours, and
# every later transfer reuses it without prompting.
#
#   ./hpc/arrhenius/sync.sh connect   Open the master connection. One password
#                                     and one code, once per working session.
#   ./hpc/arrhenius/sync.sh push      This machine -> Arrhenius, whole tree.
#   ./hpc/arrhenius/sync.sh pull      Arrhenius -> this machine, data/results/ only.
#   ./hpc/arrhenius/sync.sh submodules
#                                     Force the cluster's external/ checkouts to
#                                     the commits checked out here.
#   ./hpc/arrhenius/sync.sh shell     Interactive shell on the login node.
#   ./hpc/arrhenius/sync.sh status    Report whether the master is alive.
#   ./hpc/arrhenius/sync.sh stop      Close the master connection.
#
# external/ is not synced at all: the two solvers are submodules, so the cluster
# builds them from its own copy of .git instead. That copy includes
# .git/modules/, the submodules' object stores, which is what lets the checkout
# happen with no network and no GitHub credentials on the cluster -- AOC-CBS is
# a private repository and the login node cannot clone it.
#
# "submodules" mirrors the commit each submodule has checked out here, not the
# commit the superproject records. The recorded pointer is only updated by a
# commit, and the pointer is routinely left behind while a solver moves, so
# "git submodule update" on the cluster would check out a stale version. The
# checkout is forced: tracked files under external/ on the cluster are
# overwritten, which is what keeps the two machines running the same solver.
# Untracked files are left alone, so the AOC-CBS run directories survive it.
#
# Uncommitted changes inside a submodule never reach the cluster -- only commits
# travel, through the object store. The command says so when it finds any.
#
# Neither direction deletes. data/results/ accumulates records on both machines, so a
# mirror would destroy work. Pass --delete to push if an exact mirror is what you
# want; it is still safe for the run directories, which live under the excluded
# external/.
#
# Override the host alias with ARRHENIUS_HOST and the remote directory with
# ARRHENIUS_DIR.

set -euo pipefail

HOST=${ARRHENIUS_HOST:-arrhenius}
REMOTE=${ARRHENIUS_DIR:?set ARRHENIUS_DIR to the project directory on the cluster}

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LOCAL=$(cd -- "$HERE/../.." && pwd)
EXCLUDES="$HERE/rsync-exclude.txt"

master_is_up() {
  ssh -O check "$HOST" >/dev/null 2>&1
}

ensure_master() {
  if master_is_up; then
    return
  fi
  echo "Opening a master connection to $HOST (password, then verification code)."
  ssh -fN "$HOST"
  master_is_up || { echo "Master connection did not come up." >&2; exit 1; }
}

case "${1:-}" in
  connect)
    ensure_master
    echo "Connected. Transfers will not prompt again until it expires."
    ;;

  push)
    shift
    ensure_master
    ssh "$HOST" "mkdir -p '$REMOTE'"
    rsync -az --partial --progress --exclude-from="$EXCLUDES" "$@" \
      "$LOCAL/" "$HOST:$REMOTE/"
    ;;

  pull)
    ensure_master
    mkdir -p "$LOCAL/data/results"
    rsync -az --partial --progress \
      "$HOST:$REMOTE/data/results/" "$LOCAL/data/results/"
    ;;

  submodules)
    ensure_master
    while read -r _ path; do
      sha=$(git -C "$LOCAL/$path" rev-parse HEAD)
      branch=$(git -C "$LOCAL/$path" rev-parse --abbrev-ref HEAD)
      [ "$branch" = HEAD ] && branch=detached
      echo "$path -> $sha ($branch)"

      if ! git -C "$LOCAL/$path" diff --quiet HEAD 2>/dev/null; then
        echo "  warning: uncommitted changes here; they stay on this machine." >&2
      fi

      ssh -n "$HOST" "
        set -e
        cd '$REMOTE/$path'
        if ! git cat-file -e '$sha^{commit}' 2>/dev/null; then
          echo '  $sha is not on the cluster. Run \"sync.sh push\" first.' >&2
          exit 1
        fi
        if [ '$branch' = detached ]; then
          git checkout --force --detach '$sha' >/dev/null
        else
          git checkout --force -B '$branch' '$sha' >/dev/null
        fi
        echo \"  now at \$(git log --oneline -1)\"
      "
    done < <(git config -f "$LOCAL/.gitmodules" --get-regexp '^submodule\..*\.path$')
    ;;

  shell)
    ensure_master
    ssh -t "$HOST" "cd '$REMOTE' 2>/dev/null; exec \$SHELL -l"
    ;;

  status)
    if master_is_up; then
      ssh -O check "$HOST"
    else
      echo "No master connection to $HOST."
    fi
    ;;

  stop)
    ssh -O exit "$HOST" 2>/dev/null || echo "No master connection to close."
    ;;

  *)
    awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"
    exit 1
    ;;
esac
