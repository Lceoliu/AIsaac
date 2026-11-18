# Sourced by the evaluation wrappers. The permanent claim prevents a second job for the same tag.
ev_status() {
  python3 "$BM/evtools/ev_complete.py" --dir "$HOST/$tag" --seeds "$seeds" "$@"
}
ev_existing() {
  [ -e "$HOST/$tag" ] || [ -e "$HOST/$tag.log.request" ] || [ -e "$HOST/$tag.log.recorded" ] || [ -e "$HOST/$tag.log.rc" ]
}
ev_skip() {
  if status=$(ev_status); then
    printf 'skip %s (done): %s\n' "$tag" "$status"
  else
    ev_rc=$?
    printf 'not done %s: %s\n' "$tag" "$status"
    return "$ev_rc"
  fi
}
ev_claim() (
  set -C
  printf '%s\n' "$seeds" > "$HOST/$tag.log.request"
)
ev_finish() {
  # Keep pending on any publication failure: a subsequent invocation must not
  # take a summary or a stale .rc as evidence that this attempt completed.
  if [ -z "$1" ] || [ ! -f "$1.rc" ] ||
     [ "$(cat "$1.rc")" != "$2" ] ||
     ! cp "$1" "$HOST/$tag.log" ||
     ! cp "$1.rc" "$HOST/$tag.log.rc.new" ||
     ! mv -fT "$HOST/$tag.log.rc.new" "$HOST/$tag.log.rc" ||
     ! cp "$HOST/$tag.log.request" "$HOST/$tag.log.recorded.new" ||
     ! mv -fT "$HOST/$tag.log.recorded.new" "$HOST/$tag.log.recorded"; then
    echo "summary INCOMPLETE ($tag: could not publish current job exit status)" >&2
    return 1
  fi
  status=$(ev_status --require-rc) && check=0 || check=$?
  printf '%s %s %s: %s s, %s (%s)\n' "$(date '+%F %T')" "$tag" "$ck" "$seeds" "$(( $(date +%s) - t0 ))" "$status" "$1"
  [ "$2" = 0 ] || return 1
  return "$check"
}
