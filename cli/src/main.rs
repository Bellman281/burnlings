//! burnlings — a small runner for the exercises, in the spirit of `rustlings`.
//!
//! It reads `info.toml` (order, hints, whether an exercise has a test) and the
//! `// I AM NOT DONE` markers, then shells out to cargo. The point is that you
//! never have to look up or retype an exercise name:
//!
//! ```text
//! burnlings next     # run the first unsolved exercise
//! burnlings watch    # re-run it every time you save
//! ```

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, Subcommand};
use serde::Deserialize;

const MARKER: &str = "I AM NOT DONE";
const POLL: Duration = Duration::from_millis(500);

#[derive(Parser)]
#[command(
    name = "burnlings",
    about = "Run the burnlings exercises without typing their names",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// List every exercise with its chapter and status
    List {
        /// Print bare names only, one per line (used by shell completion)
        #[arg(long)]
        names: bool,
    },
    /// Run the first unsolved exercise
    Next,
    /// Run one exercise by name, or by any unambiguous prefix
    Run {
        /// Exercise name, e.g. `tensors3` or just `ten3`
        name: String,
    },
    /// Print the hint for an exercise (defaults to the current one)
    Hint {
        /// Exercise name; omit for the first unsolved exercise
        name: Option<String>,
    },
    /// Re-run the current exercise every time you save it
    Watch,
    /// Emit a shell completion script (bash or zsh)
    Completions {
        /// Shell to generate for: bash or zsh
        shell: String,
    },
}

#[derive(Debug, Deserialize)]
struct Info {
    #[serde(default)]
    welcome_message: String,
    #[serde(default)]
    final_message: String,
    exercises: Vec<Exercise>,
}

#[derive(Debug, Deserialize, Clone)]
struct Exercise {
    name: String,
    dir: String,
    #[serde(default)]
    test: bool,
    #[serde(default)]
    hint: String,
}

impl Exercise {
    fn path(&self, root: &Path) -> PathBuf {
        root.join("exercises").join(&self.dir).join(format!("{}.rs", self.name))
    }

    /// An exercise counts as done once the `I AM NOT DONE` marker is gone.
    fn is_done(&self, root: &Path) -> Result<bool> {
        let path = self.path(root);
        let source = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        Ok(!source.contains(MARKER))
    }
}

/// Walk up from the current directory until we find the repo (`info.toml` next
/// to a `Cargo.toml`), so the runner works from any subdirectory.
fn find_root() -> Result<PathBuf> {
    let mut dir = std::env::current_dir()?;
    loop {
        if dir.join("info.toml").is_file() && dir.join("Cargo.toml").is_file() {
            return Ok(dir);
        }
        if !dir.pop() {
            bail!("not inside a burnlings checkout (no info.toml found in any parent directory)");
        }
    }
}

fn load_info(root: &Path) -> Result<Info> {
    let path = root.join("info.toml");
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("reading {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

/// `ten3` matches `tensors3`: every character of the query appears in order.
fn is_subsequence(query: &str, name: &str) -> bool {
    let mut chars = name.chars();
    query.chars().all(|q| chars.any(|c| c == q))
}

/// Exact name first, then a unique prefix, then a unique loose match, then a
/// clear error.
fn resolve<'a>(info: &'a Info, query: &str) -> Result<&'a Exercise> {
    if let Some(found) = info.exercises.iter().find(|e| e.name == query) {
        return Ok(found);
    }
    let mut matches: Vec<&Exercise> = info
        .exercises
        .iter()
        .filter(|e| e.name.starts_with(query))
        .collect();
    if matches.is_empty() {
        matches = info
            .exercises
            .iter()
            .filter(|e| is_subsequence(query, &e.name))
            .collect();
    }
    match matches.len() {
        1 => Ok(matches[0]),
        0 => Err(anyhow!(
            "no exercise matches `{query}` — run `burnlings list` to see them all"
        )),
        _ => {
            let names: Vec<&str> = matches.iter().map(|e| e.name.as_str()).collect();
            Err(anyhow!(
                "`{query}` is ambiguous: {} — be more specific",
                names.join(", ")
            ))
        }
    }
}

fn first_unsolved<'a>(info: &'a Info, root: &Path) -> Result<Option<&'a Exercise>> {
    for exercise in &info.exercises {
        if !exercise.is_done(root)? {
            return Ok(Some(exercise));
        }
    }
    Ok(None)
}

/// `cargo test --example <name>` when the exercise has a test, else `cargo run`.
fn run_exercise(root: &Path, exercise: &Exercise) -> Result<bool> {
    let verb = if exercise.test { "test" } else { "run" };
    println!("\n== {} ({}) — cargo {verb} --example {}", exercise.name, exercise.dir, exercise.name);

    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let status = Command::new(cargo)
        .current_dir(root)
        .args([verb, "--example", &exercise.name])
        .status()
        .with_context(|| format!("running cargo {verb} --example {}", exercise.name))?;

    if status.success() {
        if exercise.is_done(root)? {
            println!("\n✅ {} passes. `burnlings next` for the next one.", exercise.name);
        } else {
            println!(
                "\n✅ {} passes — remove the `// {MARKER}` comment to move on.",
                exercise.name
            );
        }
    } else {
        println!("\n❌ {} is not there yet.", exercise.name);
        if !exercise.hint.trim().is_empty() {
            println!("   hint: burnlings hint {}", exercise.name);
        }
    }
    Ok(status.success())
}

fn cmd_list(info: &Info, root: &Path, names_only: bool) -> Result<()> {
    if names_only {
        for exercise in &info.exercises {
            println!("{}", exercise.name);
        }
        return Ok(());
    }

    let width = info.exercises.iter().map(|e| e.name.len()).max().unwrap_or(8);
    let mut done_count = 0;
    let mut current_chapter = "";
    for exercise in &info.exercises {
        if exercise.dir != current_chapter {
            current_chapter = &exercise.dir;
            println!("\n{current_chapter}");
        }
        let done = exercise.is_done(root)?;
        if done {
            done_count += 1;
        }
        println!(
            "  [{}] {:width$}  {}",
            if done { "x" } else { " " },
            exercise.name,
            if exercise.test { "test" } else { "run " },
        );
    }
    println!("\n{done_count}/{} done", info.exercises.len());
    Ok(())
}

fn cmd_next(info: &Info, root: &Path) -> Result<bool> {
    match first_unsolved(info, root)? {
        Some(exercise) => run_exercise(root, exercise),
        None => {
            println!("All {} exercises are done. 🔥", info.exercises.len());
            if !info.final_message.trim().is_empty() {
                println!("{}", info.final_message.trim());
            }
            Ok(true)
        }
    }
}

fn cmd_hint(info: &Info, root: &Path, name: Option<&str>) -> Result<()> {
    let exercise = match name {
        Some(query) => resolve(info, query)?,
        None => match first_unsolved(info, root)? {
            Some(exercise) => exercise,
            None => {
                println!("Nothing left to hint at — every exercise is done.");
                return Ok(());
            }
        },
    };
    println!("== hint · {}\n", exercise.name);
    if exercise.hint.trim().is_empty() {
        println!("(no hint recorded for this one — the solution is in solutions/{}/{}.rs)", exercise.dir, exercise.name);
    } else {
        println!("{}", exercise.hint.trim());
    }
    Ok(())
}

/// Poll the current exercise's mtime and re-run on save. When it is solved,
/// move on to the next one automatically.
fn cmd_watch(info: &Info, root: &Path) -> Result<()> {
    if !info.welcome_message.trim().is_empty() {
        println!("{}", info.welcome_message.trim());
    }
    println!("\nWatching for changes. Ctrl-C to stop.");

    let mut current: Option<String> = None;
    let mut last_seen: Option<SystemTime> = None;

    loop {
        let exercise = match first_unsolved(info, root)? {
            Some(exercise) => exercise,
            None => {
                println!("\nAll {} exercises are done. 🔥", info.exercises.len());
                if !info.final_message.trim().is_empty() {
                    println!("{}", info.final_message.trim());
                }
                return Ok(());
            }
        };

        let path = exercise.path(root);
        let mtime = std::fs::metadata(&path).and_then(|m| m.modified()).ok();
        let switched = current.as_deref() != Some(exercise.name.as_str());
        let touched = !switched && mtime != last_seen;

        if switched || touched {
            current = Some(exercise.name.clone());
            run_exercise(root, exercise)?;
            // The run may have taken long enough for another save to land.
            last_seen = std::fs::metadata(&path).and_then(|m| m.modified()).ok();
        }

        std::thread::sleep(POLL);
    }
}

fn cmd_completions(shell: &str) -> Result<()> {
    match shell {
        "bash" => print!("{}", include_str!("completions/burnlings.bash")),
        "zsh" => print!("{}", include_str!("completions/burnlings.zsh")),
        other => bail!("unsupported shell `{other}` — try bash or zsh"),
    }
    Ok(())
}

fn run() -> Result<bool> {
    let cli = Cli::parse();
    let root = find_root()?;

    // `completions` is the one command that doesn't need info.toml parsed.
    if let Some(Cmd::Completions { shell }) = &cli.command {
        cmd_completions(shell)?;
        return Ok(true);
    }

    let info = load_info(&root)?;

    match cli.command {
        None | Some(Cmd::Next) => cmd_next(&info, &root),
        Some(Cmd::List { names }) => cmd_list(&info, &root, names).map(|()| true),
        Some(Cmd::Run { name }) => {
            let exercise = resolve(&info, &name)?;
            run_exercise(&root, exercise)
        }
        Some(Cmd::Hint { name }) => cmd_hint(&info, &root, name.as_deref()).map(|()| true),
        Some(Cmd::Watch) => cmd_watch(&info, &root).map(|()| true),
        Some(Cmd::Completions { .. }) => unreachable!("handled above"),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        // An exercise that doesn't pass is not a runner error, but it should
        // still be a non-zero exit so CI and `&&` chains behave.
        Ok(false) => ExitCode::FAILURE,
        Err(err) => {
            eprintln!("error: {err:#}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info() -> Info {
        let ex = |name: &str, dir: &str| Exercise {
            name: name.into(),
            dir: dir.into(),
            test: true,
            hint: String::new(),
        };
        Info {
            welcome_message: String::new(),
            final_message: String::new(),
            exercises: vec![
                ex("tensors1", "01_tensors"),
                ex("tensors3", "01_tensors"),
                ex("ops3", "02_ops"),
                ex("attn1", "12_attention"),
            ],
        }
    }

    #[test]
    fn exact_beats_prefix() {
        assert_eq!(resolve(&info(), "ops3").unwrap().name, "ops3");
    }

    #[test]
    fn unique_prefix_resolves() {
        assert_eq!(resolve(&info(), "att").unwrap().name, "attn1");
    }

    #[test]
    fn loose_match_resolves() {
        assert_eq!(resolve(&info(), "ten3").unwrap().name, "tensors3");
    }

    #[test]
    fn ambiguous_prefix_is_an_error() {
        let err = resolve(&info(), "tensors").unwrap_err().to_string();
        assert!(err.contains("ambiguous"), "{err}");
    }

    #[test]
    fn unknown_name_is_an_error() {
        let err = resolve(&info(), "nope").unwrap_err().to_string();
        assert!(err.contains("no exercise matches"), "{err}");
    }

    #[test]
    fn real_info_toml_parses_and_matches_the_manifest() {
        // info.toml and Cargo.toml must agree, or `burnlings run` and
        // `cargo run --example` would disagree about what exists.
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
        let info = load_info(&root).unwrap();
        let manifest = std::fs::read_to_string(root.join("Cargo.toml")).unwrap();

        for exercise in &info.exercises {
            assert!(
                manifest.contains(&format!("name = \"{}\"", exercise.name)),
                "{} is in info.toml but not registered as a [[example]]",
                exercise.name
            );
            assert!(
                exercise.path(&root).is_file(),
                "missing {}",
                exercise.path(&root).display()
            );
        }
    }
}
