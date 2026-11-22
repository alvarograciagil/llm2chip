# Installing and running llm2chip

This guide covers what to install, how to start the application, the ways to run it, and every option it takes at start-up.

llm2chip has two parts:

- **The engine** (`crates/server`): a local Rust program that reads models, runs the simulations and serves the API. It listens only on `127.0.0.1`.
- **The interface** (`web/`): a React application. Its estimator is compiled to WebAssembly and runs in the browser; everything with real weights goes through the engine.

During development, Vite serves the interface on port 5173 and forwards `/api` to the engine on port 7878. The engine can also serve the built interface by itself, and the desktop application wraps both in a window.

## Requirements

| What | Needed for | Notes |
|---|---|---|
| [Rust](https://rustup.rs) through rustup | Everything | A recent stable toolchain; the project is developed with Rust 1.99. |
| Rust target `wasm32-unknown-unknown` | The interface | `rustup target add wasm32-unknown-unknown` |
| [wasm-pack](https://rustwasm.github.io/wasm-pack/) | The interface | Builds the estimator into `web/src/wasm`. |
| [Node.js](https://nodejs.org) 22.12 or newer, with npm | The interface | Vite needs 20.19 or newer; the tests need 22.12 or newer. |
| A GPU driver: Metal (macOS) or Vulkan (Linux, Windows) | Faster simulation | Optional. Without a usable GPU, the CPU does the work, more slowly. |
| Rust target `riscv32im-unknown-none-elf` | Rebuilding the firmware | Optional. A built firmware is included in `firmware/prebuilt/`; the tests that rebuild it are skipped without the target. |
| [Docker](https://www.docker.com) | Laying out a cell with OpenROAD | Optional. Pulls the `openroad/orfs` image the first time. |
| [Icarus Verilog](https://steveicarus.github.io/iverilog/) | Simulating the exported Verilog | Optional. Used by the exported projects and the `LLM2CHIP_ICARUS` test. |
| [OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build) (Yosys, nextpnr) | FPGA synthesis | Optional. Expected in `~/.llm2chip/tools/oss-cad-suite`, or where `OSS_CAD_SUITE` points. |
| [Tauri CLI](https://tauri.app) 2 | Building the desktop application | Optional. `cargo install tauri-cli --version "^2"` |
| [uv](https://docs.astral.sh/uv/) with Python | The floating-point references in `tools/golden` | Optional. Only to regenerate reference data. |

Memory: the engine maps weights from disk instead of copying them, but running a model still needs roughly its size in RAM. Qwen3.8-27B at 4 bits (about 20 GB) runs on a 64 GB machine; small models such as Qwen3-0.6B or SmolLM2-135M run anywhere.

### macOS

```bash
brew install rustup node wasm-pack
rustup-init -y
rustup target add wasm32-unknown-unknown riscv32im-unknown-none-elf
```

Homebrew's `rustup` is keg-only: the scripts add `/opt/homebrew/opt/rustup/bin` to `PATH` themselves. The GPU backend is Metal and needs nothing else.

### Linux

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup target add wasm32-unknown-unknown riscv32im-unknown-none-elf
cargo install wasm-pack
```

Install Node 22.12 or newer from your distribution or from nodejs.org. For the GPU backend, install your card's Vulkan driver and the Vulkan loader (`libvulkan1` on Debian and Ubuntu, `vulkan-icd-loader` on Arch).

### Windows

Install Rust with `rustup-init.exe` from [rustup.rs](https://rustup.rs) (with the MSVC build tools it asks for), Node 22.12 or newer from [nodejs.org](https://nodejs.org), and then, in PowerShell:

```powershell
rustup target add wasm32-unknown-unknown riscv32im-unknown-none-elf
cargo install wasm-pack
```

The GPU backend is Vulkan, which comes with current NVIDIA, AMD and Intel drivers. Use the `.ps1` scripts in `scripts/` instead of the `.sh` ones.

## Installing

```bash
git clone https://github.com/alvarograciagil/llm2chip.git
cd llm2chip
npm --prefix web ci
```

That is all: the first start builds the rest.

## Starting

```bash
scripts/dev.sh
```

On Windows, `scripts\dev.ps1`. The script:

1. Builds the estimator to WebAssembly (`scripts/build-wasm.sh`).
2. Builds the engine in release mode (`cargo build --release -p server`). The first build takes several minutes; later ones only rebuild what changed.
3. Starts the engine on `127.0.0.1:7878`.
4. Starts Vite on `127.0.0.1:5173` and stops the engine when Vite stops.

Then open `http://localhost:5173`. If the Models view says the local engine is not running, the engine failed to start: its error is in the terminal.

### A first model

Weights are not included. The Models view lists every model found in:

- `~/.llm2chip/models`, where downloads go;
- `~/.mtplx/models`;
- the Hugging Face cache (`$HF_HUB_CACHE`, `$HF_HOME/hub` or `~/.cache/huggingface/hub`);
- the folders in `LLM2CHIP_MODEL_DIRS`;
- the folders you add in the Models view.

Each of the usual places can be turned off in that view; the download folder is always searched. Folders are searched up to three levels deep, and a model is any folder with a `config.json` and `.safetensors` weights.

To get one, open **Models → Hugging Face**, search for a small model such as `Qwen/Qwen3-0.6B` or `HuggingFaceTB/SmolLM2-135M`, and download it. Gated models need their license accepted on Hugging Face and a read token (see `HF_TOKEN` below, or the field in the same view). Then use **Design chip** to size a chip for it, or **Simulate** to run it.

The tiny llama2.c models (stories260K, 15M, 110M) are converted from their `.bin` files with the standard library alone:

```bash
python3 tools/convert/llama2c.py stories260K.bin tok512.bin ~/.llm2chip/models/stories260K
python3 tools/convert/llama2c.py stories260K.bin tok512.bin ~/.llm2chip/models/stories260K-q8 --affine 8
```

## Other ways to run it

### The engine alone, with the built interface

Without Vite, the engine serves the built interface itself:

```bash
npm --prefix web run build
cargo build --release -p server
target/release/server
```

Then open `http://127.0.0.1:7878`. The engine looks for the interface in `web/dist` next to the repository it was built in; `LLM2CHIP_DIST` points it elsewhere.

### The desktop application

```bash
cargo install tauri-cli --version "^2"
scripts/build-app.sh
```

On macOS it leaves `app/src-tauri/target/release/bundle/macos/llm2chip.app`; elsewhere, the system's packages in `app/src-tauri/target/release/bundle`. The application starts its own engine and closes it on exit; if another engine already uses the port, the window connects to that one instead.

### From other devices on your network

```bash
scripts/dev.sh --lan
```

On Windows, `scripts\dev.ps1 -Lan`. Vite then listens on every interface and prints an address with an access key, also saved in `~/.llm2chip/lan-url`. Other devices must open that address once; a cookie remembers the key. The engine itself still listens only on `127.0.0.1`.

Anyone with that address can load and run models on your machine: do not share it outside your network. Traffic is not encrypted, so type a Hugging Face token only on the machine itself.

### The online demo, without the engine

The interface also runs as static files, with no engine behind it: the estimator, the floorplan, how it works, the schematic simulated gate by gate, the explorer and the export to Verilog and VHDL all run in the browser on WebAssembly. The views that need the engine (Images, Numerics, Optimise and Models) say they need the app installed instead of calling it.

The Simulator runs there too, on a WebAssembly build of the same simulator (`crates/wasm-sim`): it downloads a model into the browser and runs it on the design's integer arithmetic, on one thread and without a GPU, with the engine's framing, sampling and stops. It offers stories260K, served with the demo, and SmolLM2-135M-Instruct from Hugging Face, and takes any public model of a supported architecture up to 300 MB (`DOWNLOAD_LIMIT` in `web/src/simulator/browserModels.ts`); a model takes about three times its size in memory. By default a download lives only in the page's memory and goes when the page closes, so trying a model leaves nothing on disk. With "Keep the downloaded models in this browser" ticked (the choice is remembered), downloads stay in the browser's private storage (OPFS) and load again without the network; a model not loaded for 30 days (`KEEP_DAYS`) is deleted the next time the Simulator opens, and so is any download cut off halfway. The Simulator shows what is kept and deletes it, keeps nothing when the browser lacks room, and warns when a model may need more memory than the device has. The chip's CPU, its firmware and the image load need the app.

A link can open a view: `#schematic`, `#explorer` and the rest by their names. `#simulator` also loads stories260K and runs it, and `#simulator/HuggingFaceTB/SmolLM2-135M-Instruct` chooses that model, which waits for a click because it is a larger download.

```bash
scripts/build-wasm.sh
scripts/demo-models.sh
cd web && VITE_DEMO=1 LLM2CHIP_BASE=/llm2chip/ npx vite build
```

`scripts/demo-models.sh` downloads stories260K from [karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas/tree/main/stories260K) at a fixed commit, checks it and converts it; without it the demo offers only the models on Hugging Face.

`LLM2CHIP_BASE` is the path the files are served under (`/` by default). The workflow in `.github/workflows/demo.yml` builds it on every push and publishes it to GitHub Pages from `main`.

#### On several threads, on a Hugging Face Space

WebAssembly threads share their memory, which browsers allow only to pages served with two headers (`Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy`). GitHub Pages cannot send them; a static Hugging Face Space can, with `custom_headers` in its README (`space/README.md`). There, models of 32 MB or more run on as many threads as the device has: SmolLM2-135M took 0.14–0.24 s per token against 0.29–0.39 on one thread, measured with other work running on the machine. Smaller ones stay on one thread, where they are faster.

`scripts/build-wasm.sh --threads` builds the threaded simulator as well, with a pinned nightly toolchain (installed if missing) into `web/src/wasm-sim-threads`. The page picks it only when it is cross-origin isolated, so the same build runs everywhere, on one thread where the headers are missing.

To publish the Space from `main`:

1. Create a Space on Hugging Face with the Static SDK (for example `your-name/llm2chip`).
2. Create a Hugging Face access token with write access to it.
3. In the GitHub repository, add the token as the secret `HF_TOKEN`, and the Space's name as the variable `HF_SPACE` (Settings → Secrets and variables → Actions).

The `space` job then builds the demo with threads and uploads it on every push to `main`. Share the Space's own address (`https://<owner>-<name>.static.hf.space`): embedded in the Hugging Face page the browser may not isolate it, and the simulator falls back to one thread.

## Start-up options

### Scripts

| Command | What it does |
|---|---|
| `scripts/dev.sh` | Builds and starts the engine and the interface. |
| `scripts/dev.sh --lan` | The same, reachable from the local network with a key. On Windows, `dev.ps1 -Lan`. |
| `scripts/build-wasm.sh` | Rebuilds only the WebAssembly estimator and the browser's simulator. |
| `scripts/build-wasm.sh --threads` | Also the simulator on several threads, for a Hugging Face Space. |
| `scripts/build-app.sh` | Builds the desktop application. |
| `scripts/check.sh` | The quick tests and lints, a few minutes. |
| `scripts/check.sh --full` | Also the whole chips in gates and the slow tests; for changes to the hardware. |

### Environment variables

Set them before starting, for example `LLM2CHIP_ENGINE_PORT=7900 scripts/dev.sh`.

| Variable | Default | Read by | What it does |
|---|---|---|---|
| `LLM2CHIP_ENGINE_PORT` | `7878` | Engine, Vite, desktop app | The engine's port on `127.0.0.1`. Use it when 7878 is taken or to run two engines. |
| `LLM2CHIP_WEB_PORT` | `5173` | Vite | The interface's port. |
| `LLM2CHIP_LAN` | unset | Vite | `1` opens the interface to the local network; `--lan` sets it. |
| `LLM2CHIP_LAN_KEY` | generated | Vite | The access key for other devices. Without it, one is generated and kept in `~/.llm2chip/lan-key`. |
| `LLM2CHIP_MODEL_DIRS` | unset | Engine | Extra model folders, separated as in `PATH` (`:` on macOS and Linux, `;` on Windows). They can be turned off in the Models view like the usual places. |
| `LLM2CHIP_DIST` | `web/dist` | Engine | The built interface the engine serves on its own port. |
| `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` | unset | Engine | Hugging Face read token, for gated models. A token saved in the Models view or by `huggingface-cli login` also works. |
| `HF_ENDPOINT` | `https://huggingface.co` | Engine | Another Hugging Face server or mirror. |
| `HF_HUB_CACHE`, `HF_HOME` | `~/.cache/huggingface` | Engine | Where the Hugging Face cache is, to search it for models. |

Variables used only by tests and tools:

| Variable | What it does |
|---|---|
| `LLM2CHIP_ICARUS=1` | Runs the test that simulates the whole chip in Icarus Verilog (close to an hour). |
| `LLM2CHIP_STORIES` | The stories260K folder for the chip tests; default `~/.llm2chip/models/stories260K`. |
| `LLM2CHIP_SHOWO` | The Show-o folder for the image tests; default `~/.llm2chip/models/Show-o`. |
| `LLM2CHIP_TOKENIZERS` | A folder of extra tokenizers for the tokenizer tests, each in its own folder. |
| `OSS_CAD_SUITE` | Where OSS CAD Suite is, for `tools/synth/families.sh`. |

The examples take their own variables and arguments; see [examples](examples.md).

## Files it writes

Everything goes under `~/.llm2chip` (`%USERPROFILE%\.llm2chip` on Windows):

| Path | What |
|---|---|
| `models/` | Downloaded models, unless another download folder is set. |
| `settings.json` | Model folders, the download folder and the usual places turned off. |
| `session-token` | A new token on each engine start; the interface and Vite present it to the engine. |
| `lan-key`, `lan-url` | The local-network key and address, with `--lan`. |
| `images/` | The chips' boot images built from the Models view. |
| `asic/` | OpenROAD layouts of the cell. |
| `tools/` | Optional tools such as OSS CAD Suite. |

The interface keeps its own preferences (language, appearance, saved designs) in the browser.

## Troubleshooting

- **"cannot listen on 127.0.0.1:7878"**: another engine is running, often a forgotten `dev.sh` or the desktop application. Stop it, or start this one with another `LLM2CHIP_ENGINE_PORT`.
- **Vite says port 5173 is already in use**: another Vite is running; stop it or set `LLM2CHIP_WEB_PORT`.
- **The Models view says the engine is not running**: the engine did not start or stopped; its error is in the terminal where `dev.sh` runs.
- **`wasm-pack: command not found`**: install it as above, and check that `~/.cargo/bin` is on your `PATH`.
- **Simulation is slow**: when a model loads, the Simulator shows the accelerator it uses (for example `Metal (Apple M2 Max)`). With none, the CPU does the work; check the Metal or Vulkan driver. Large models run at seconds per token in any case.
- **The interface shows a 403 from the engine**: the engine restarted with a new session token; reload the page.
- **A model is missing from the list**: check that its folder has a `config.json` and `.safetensors` files, that it is at most three levels inside a searched folder, and that the folder is not turned off; then press **Search again**.
