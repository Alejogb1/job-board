---
title: "How do I install Rust in Termux?"
date: "2025-01-30"
id: "how-do-i-install-rust-in-termux"
---
My experience deploying embedded systems and tinkering with cross-compilation has often required setting up development environments in constrained or unconventional locations. This included, on several occasions, utilizing Android devices running Termux, which, while not a typical development target, offers a surprisingly robust environment for basic tooling like Rust. Installing Rust in Termux requires navigating some platform specifics, primarily the architecture and package management considerations inherent to the environment. It's not a direct, one-command process like on a traditional Linux distribution, but it’s achievable with a clear understanding of the necessary steps.

The core challenge lies in the fact that Termux, being a user-space application running on Android, does not adhere to the standard Linux directory structure, and does not have direct access to the system's package manager in the same way a desktop Linux system does. Instead, Termux uses its own package management system based on `pkg`. Further, architecture compatibility is essential. Android devices typically utilize ARM architectures (armv7l, aarch64), and thus the appropriate Rust toolchain for these targets needs to be installed. The standard `rustup` installer, while functional on many Linux distros, needs to be specifically configured within the Termux context.

The first step involves updating the Termux package repository. This ensures access to the most recent versions of required dependencies. I generally execute this by running:

```bash
pkg update && pkg upgrade
```

This command updates the package lists and then upgrades any installed packages to their latest versions. This is crucial as many subsequent steps depend on updated versions of core tools within the Termux ecosystem. Following this, it's important to install `curl` and `git`. `curl` will be necessary to download the Rust installation script and `git` will be helpful for working with projects that are maintained through the `git` system:

```bash
pkg install curl git
```

With the prerequisites established, the next stage involves downloading the `rustup` installation script directly from the official Rust website using `curl`. This is usually done with a single command piped to `sh`. The command may appear complex, but each part is important for a successful install:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Here, `--proto '=https'` enforces the use of the secure `https` protocol, while `--tlsv1.2` enforces a minimum TLS version ensuring a secure connection. The `-sSf` options instruct `curl` to run in silent mode and follow redirects while failing if any problems occur, and the downloaded content is passed to the `sh` command, which will attempt to execute it. I’ve found that this step can be prone to issues if the environment variables are not set correctly, particularly in the context of a non-standard shell environment such as Termux.

Upon completion of the `rustup` script, which typically takes a few minutes, `rustup` will be located at the `$HOME/.cargo/bin` directory. However, this directory is not on your default `PATH`. To rectify this, you'll need to add the `bin` folder to your path. I add the following to my `.bashrc` file, if using `bash` which is the default shell in Termux:

```bash
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
```

This line appends the necessary directory to your `PATH` variable and the command is appended to the `.bashrc` so that it persists for future sessions. After this change, you need to source the `.bashrc` file or restart the terminal for the changes to take effect. You can verify the path with `echo $PATH`. Once this is done, `rustc`, the Rust compiler, and `cargo`, the Rust build system, should be accessible.

To verify the installation, you can try compiling and running a basic Rust program. The following demonstrates this process. First, create a file named `main.rs` with the following content:

```rust
fn main() {
    println!("Hello, Termux!");
}
```

This is a very simple Rust program which will print the text "Hello, Termux!" to the terminal screen. Once that is saved, in the same directory run the following:

```bash
rustc main.rs
./main
```

The `rustc` command will compile the `main.rs` file, producing an executable. The second command executes the generated binary, and if successful will result in "Hello, Termux!" being displayed in the terminal.

A more practical example involves using `cargo` to manage dependencies. Create a new project with `cargo new hello_termux`. Then enter that directory with `cd hello_termux`. Open the `src/main.rs` file and modify it to contain the following content:

```rust
use rand::Rng;

fn main() {
    let mut rng = rand::thread_rng();
    let random_number: u32 = rng.gen();
    println!("Random number: {}", random_number);
}
```

This program will print a random number. This also shows that you can add external dependencies to the `Cargo.toml` file. In the `Cargo.toml` file, add the following under `[dependencies]`:
```
rand = "0.8"
```
This declaration tells `cargo` that you need the `rand` crate as part of the project and that you want version `0.8` (or newer). Now, run `cargo build`. The first time this runs, `cargo` will download the declared dependencies. Once downloaded and compiled, the final binary can be run with `cargo run`.

A final common task is cross-compilation for other devices. This usually requires setting up target-specific toolchains, and in the context of Termux, you would use `rustup target add` to install these toolchains. For example, if targeting `aarch64-linux-android` (a common architecture for Android devices) you would install this target:

```bash
rustup target add aarch64-linux-android
```

Then, you could build code targeting `aarch64-linux-android` using `cargo build --target aarch64-linux-android`. While I have demonstrated usage of a specific ARM based target, other targets may be installed this way allowing cross-compilation for a variety of architectures if needed.

In summary, the process of installing Rust in Termux, while requiring some modifications to the standard installation process, is achievable by understanding the unique environment it provides. You must update packages, install prerequisites, and utilize the `rustup` tool effectively.

For further reading and more advanced topics, I recommend checking the official Rust documentation, the Cargo book, and the rustup book. These resources detail not only the installation process but also explain more advanced build features, cross-compilation, and package management, which are crucial for further development. Additionally, researching best practices for developing on resource constrained devices will assist in ensuring successful projects are created.
