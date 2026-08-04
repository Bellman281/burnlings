# burnlings — zsh completion for `cargo run|test --example <name>`
#
# Stable cargo has no completion for `--example` (cargo's own native completion
# is nightly-only, tracked in https://github.com/rust-lang/cargo/issues/14520),
# so this reads the exercise names straight out of burnlings' Cargo.toml.
#
# Use it (after `autoload -U compinit && compinit`):
#     source scripts/burnlings-completion.zsh
#
# Then:
#     cargo run --example ten<TAB>      ->  tensors1 … tensors9
#     cargo test --example <TAB>        ->  all 53 exercises
#
# It only adds completion after `--example`; everything else falls back to
# whatever completion zsh already had for cargo.

_burnlings_manifest() {
    local dir="$PWD"
    while [[ "$dir" != "/" ]]; do
        if [[ -f "$dir/Cargo.toml" ]] && grep -q '^\[\[example\]\]' "$dir/Cargo.toml"; then
            print -r -- "$dir/Cargo.toml"
            return 0
        fi
        dir="${dir:h}"
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

_burnlings_cargo() {
    if [[ "${words[CURRENT-1]}" == "--example" ]]; then
        local -a names
        names=(${(f)"$(_burnlings_example_names)"})
        if (( ${#names} )); then
            _describe -t examples 'burnlings exercise' names
            return
        fi
    fi

    # Not an --example position: defer to cargo's own completion if present.
    if (( $+functions[_cargo] )); then
        _cargo "$@"
    else
        _default
    fi
}

compdef _burnlings_cargo cargo
