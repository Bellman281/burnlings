# burnlings — bash completion
#
#   source <(burnlings completions bash)
#
# Completes subcommands, and exercise names for `run` and `hint`.

_burnlings_complete() {
    local cur prev subcommands
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    subcommands="list next run hint watch completions help"

    if [[ "$COMP_CWORD" -eq 1 ]]; then
        mapfile -t COMPREPLY < <(compgen -W "$subcommands" -- "$cur")
        return 0
    fi

    case "$prev" in
        run|hint)
            local names
            names="$(burnlings list --names 2>/dev/null)" || return 0
            mapfile -t COMPREPLY < <(compgen -W "$names" -- "$cur")
            ;;
        completions)
            mapfile -t COMPREPLY < <(compgen -W "bash zsh" -- "$cur")
            ;;
        *)
            COMPREPLY=()
            ;;
    esac
    return 0
}

complete -F _burnlings_complete burnlings
