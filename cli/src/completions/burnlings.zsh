# burnlings — zsh completion
#
#   source <(burnlings completions zsh)
#
# Completes subcommands, and exercise names for `run` and `hint`.

_burnlings() {
    local -a subcommands
    subcommands=(
        'list:list every exercise with its status'
        'next:run the first unsolved exercise'
        'run:run one exercise by name or prefix'
        'hint:print the hint for an exercise'
        'watch:re-run the current exercise on save'
        'completions:emit a shell completion script'
    )

    if (( CURRENT == 2 )); then
        _describe -t commands 'burnlings command' subcommands
        return
    fi

    case "${words[2]}" in
        run|hint)
            local -a names
            names=(${(f)"$(burnlings list --names 2>/dev/null)"})
            (( ${#names} )) && _describe -t exercises 'exercise' names
            ;;
        completions)
            local -a shells
            shells=(bash zsh)
            _describe -t shells 'shell' shells
            ;;
    esac
}

compdef _burnlings burnlings
