# burnlings — bash completion for `cargo run|test --example <name>`
#
# Stable cargo has no completion for `--example` (cargo's own native completion
# is nightly-only, tracked in https://github.com/rust-lang/cargo/issues/14520),
# so this reads the exercise names straight out of burnlings' Cargo.toml.
#
# Use it:
#     source scripts/burnlings-completion.bash
#
# Then:
#     cargo run --example ten<TAB>      ->  tensors1 … tensors9
#     cargo test --example <TAB>        ->  all 53 exercises
#
# It only adds completion after `--example`; everything else falls back to
# bash's default (filenames). To remove it again:  complete -r cargo

_burnlings_manifest() {
    local dir="$PWD"
    while [[ "$dir" != "/" ]]; do
        if [[ -f "$dir/Cargo.toml" ]] && grep -q '^\[\[example\]\]' "$dir/Cargo.toml"; then
            printf '%s\n' "$dir/Cargo.toml"
            return 0
        fi
        dir="$(dirname "$dir")"
    done
    return 1
}

_burnlings_example_names() {
    local manifest
    manifest="$(_burnlings_manifest)" || return 1
    awk '
        /^\[\[example\]\]/ { in_example = 1; next }
        /^\[/              { in_example = 0 }
        in_example && /^[[:space:]]*name[[:space:]]*=/ {
            sub(/^[^"]*"/, ""); sub(/".*$/, ""); print
        }
    ' "$manifest"
}

_burnlings_cargo_complete() {
    local cur prev
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    if [[ "$prev" == "--example" ]]; then
        local names
        names="$(_burnlings_example_names)" || return 0
        mapfile -t COMPREPLY < <(compgen -W "$names" -- "$cur")
        return 0
    fi

    COMPREPLY=()
    return 0
}

complete -o bashdefault -o default -F _burnlings_cargo_complete cargo
