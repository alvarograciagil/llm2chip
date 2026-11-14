import { useEffect, useRef, useState, type FormEvent } from "react";
import { Segmented } from "../components/controls";
import { useSimulator, type ServerModel } from "../simulator/simulatorStore";
import { formatNumber, locale } from "../format";
import { useDesign } from "../store";
import { formatBytes, formatCount, formatRemaining, useModels, type Adaptation, type Details, type Job } from "./modelsStore";

/** With `family`, only the architecture is known: the repository's config may refuse it later. */
function Tier({ adaptation, known = true, family = false }: { adaptation: Adaptation; known?: boolean; family?: boolean }) {
  const [label, tone] = !known ? ["No data", "none"] : adaptation.simulate ? [family ? "Runnable family" : "Runnable", "full"] : adaptation.design ? [adaptation.approximate ? "Approximate design" : "Design only", "part"] : ["Not adaptable", "none"];
  return <span className={`tier ${tone}`}>{label}</span>;
}

function useToDesigner() {
  const addModel = useDesign((s) => s.addModel);
  const setParam = useDesign((s) => s.setParam);
  const setView = useDesign((s) => s.setView);
  return (name: string, config: string, adaptation: Adaptation, path?: string) => {
    addModel(name, config, path);
    // The chip is sized for this model, with a sliver so that rounding never leaves it short.
    if (adaptation.parameters) setParam("max_params_g", (adaptation.parameters / 1e9) * 1.001);
    setView("floorplan");
  };
}

function LocalList() {
  const models = useSimulator((s) => s.models);
  const setSimulator = useSimulator((s) => s.set);
  const setView = useDesign((s) => s.setView);
  const toDesigner = useToDesigner();
  const startImage = useModels((s) => s.startImage);
  const writing = useModels((s) => s.image.state === "writing");
  // One image boots on every chip count up to what the current design needs.
  const makeImage = (m: ServerModel) => {
    const { params, report } = useDesign.getState();
    const dies = Math.max(1, Math.round(report?.dies ?? 1));
    // The gates share by tensor as rows of every matrix; the mixed split is not generated yet.
    const sharing = params.chip_split === "tensor" ? "rows" : params.chip_split === "mixed" ? { mixed: Number(params.tensor_group) } : "layers";
    void startImage(m.path, Array.from({ length: dies }, (_, i) => i + 1), Number(params.context_tokens), Number(params.macs_per_die), Number(params.activation_bits), Number(params.vector_lanes_per_die), Number(params.vector_units_per_die), sharing);
  };
  const measureSpeed = useModels((s) => s.measureSpeed);
  const measuring = useModels((s) => s.measuring);
  const measure = (m: ServerModel) => {
    const { params } = useDesign.getState();
    void measureSpeed(m.path, Number(params.context_tokens), Number(params.macs_per_die), Number(params.activation_bits), Number(params.vector_lanes_per_die), Number(params.vector_units_per_die));
  };
  const simulate = (m: ServerModel) => {
    setSimulator({ path: m.path });
    setView("simulator");
  };

  if (!models.length) {
    return <p className="placeholder models-empty">No models with safetensors weights in the folders. Add a folder in the right panel or search Hugging Face.</p>;
  }
  return (
    <ul className="model-list">
      {models.map((m) => (
        <li key={m.path} className="model-row">
          <div className="model-main">
            <div className="model-title">
              <strong>{m.name}</strong>
              <Tier adaptation={m.adaptation} />
              {m.images?.reads && <span className="tier" title="Its image encoder runs in the simulator: an image can be attached to the question.">Reads images</span>}
              {m.images?.makes && <span className="tier" title="Its image decoder runs in the simulator: it can generate images from text.">Generates images</span>}
            </div>
            <p className="muted-text">
              {m.architecture} · {formatBytes(m.size_gb * 1e9)}
              {m.adaptation.parameters ? ` · ${formatCount(m.adaptation.parameters)} parameters · ${m.adaptation.layers} layers` : ""}
            </p>
            {m.adaptation.reason && <p className="model-reason">{m.adaptation.reason}</p>}
            {m.images?.note && <p className="model-reason">{m.images.note}</p>}
            <p className="model-path" title={m.path}>{m.path}</p>
          </div>
          <div className="model-actions">
            <button className="pill" disabled={!m.adaptation.design} onClick={() => toDesigner(m.name, m.config, m.adaptation, m.path)}>Design chip</button>
            <button className="pill" disabled={!m.supported} onClick={() => simulate(m)}>Simulate</button>
            <button className="pill" disabled={!m.supported || writing} onClick={() => makeImage(m)} title="Prepares the chip's boot image: microcode, weights and tables the firmware loads at power-on">Boot image</button>
            <button className="pill" disabled={!m.supported || measuring} onClick={() => measure(m)}>Measure speed</button>
          </div>
        </li>
      ))}
    </ul>
  );
}

function RepoDetails({ details }: { details: Details }) {
  const jobs = useModels((s) => s.jobs);
  const download = useModels((s) => s.download);
  const discard = useModels((s) => s.discard);
  const toDesigner = useToDesigner();
  const running = jobs.some((j) => j.id === details.id && (j.state === "queued" || j.state === "downloading"));
  const { adaptation, config } = details;
  const short = details.free_bytes !== null && details.free_bytes < details.total_bytes - details.resumable_bytes;

  return (
    <div className="repo-details">
      <p className="muted-text">
        {details.architecture ?? "unknown architecture"}
        {adaptation.parameters ? ` · ${formatCount(adaptation.parameters)} parameters · ${adaptation.layers} layers` : ""}
        {details.gated ? " · gated access" : ""}
      </p>
      {adaptation.approximate && adaptation.design && <p className="model-reason">The circuit is estimated from the shape of its layers. Its activation or extra norms are not modelled.</p>}
      {adaptation.reason && <p className="model-reason">{adaptation.reason}</p>}
      {details.files.length > 0 && (
        <p className="muted-text">
          {details.files.filter((f) => f.name.endsWith(".safetensors")).length} weight file(s) · {formatBytes(details.total_bytes)} in total
          {details.free_bytes !== null ? ` · ${formatBytes(details.free_bytes)} libres` : ""}
        </p>
      )}
      {details.resumable_bytes > 0 && <p className="muted-text">{formatBytes(details.resumable_bytes)} from an earlier download will be reused.</p>}
      <div className="model-actions inline">
        <button className="pill" disabled={!adaptation.design || !config} onClick={() => config && toDesigner(details.id, config, adaptation)}>Design chip without downloading</button>
        <button className="pill primary-pill" disabled={details.present || running || !details.files.length || short} onClick={() => void download(details.id)}>
          {details.present ? "Already downloaded" : running ? "Downloading…" : short ? "Does not fit on the disk" : `Download ${formatBytes(details.total_bytes)}`}
        </button>
        {details.resumable_bytes > 0 && !running && <button className="pill" onClick={() => void discard(details.id)}>Delete the partial download</button>}
      </div>
    </div>
  );
}

function HubSearch() {
  const { query, searched, hits, searching, selected, details, error, set, search, select } = useModels();
  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (query.trim()) void search();
  };
  const direct = selected !== null && !hits.some((h) => h.id === selected);

  return (
    <>
      <form className="model-search" onSubmit={submit}>
        <input type="search" aria-label="Search Hugging Face" placeholder="Search, or type author/model" value={query} onChange={(e) => set({ query: e.target.value })} />
        <button className="pill primary-pill" disabled={!query.trim() || searching}>{searching ? "Searching…" : "Search"}</button>
      </form>
      {error && (
        <div className="issue error" role="alert">
          <span className="issue-icon" aria-hidden="true">!</span>
          <div><p>{error}</p></div>
        </div>
      )}
      {searched !== null && !searching && !hits.length && !selected && !error && <p className="placeholder models-empty">No results for “{searched}” with safetensors weights.</p>}
      {direct && (
        <ul className="model-list">
          <li className="model-row open">
            <div className="model-main">
              <div className="model-title">
                <strong>{selected}</strong>
                {details && <Tier adaptation={details.adaptation} />}
              </div>
              {details ? <RepoDetails details={details} /> : !error && <p className="muted-text">Looking it up…</p>}
            </div>
          </li>
        </ul>
      )}
      {hits.length > 0 && (
        <ul className="model-list">
          {hits.map((hit) => {
            const open = hit.id === selected;
            return (
              <li key={hit.id} className={open ? "model-row open" : "model-row"}>
                <div className="model-main">
                  <button className="model-title model-toggle" aria-expanded={open} onClick={() => void select(open ? null : hit.id)}>
                    <strong>{hit.id}</strong>
                    {!(open && details) && <Tier adaptation={hit.adaptation} known={hit.architecture !== null} family />}
                    {open && details && <Tier adaptation={details.adaptation} />}
                  </button>
                  <p className="muted-text">{hit.architecture ?? "unknown architecture"} · {formatCount(hit.downloads)} descargas · {formatCount(hit.likes)} me gusta</p>
                  {open && (details ? <RepoDetails details={details} /> : !error && <p className="muted-text">Looking it up…</p>)}
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </>
  );
}

export function ModelsView() {
  const source = useModels((s) => s.source);
  const set = useModels((s) => s.set);
  const loadSettings = useModels((s) => s.loadSettings);
  const pollJobs = useModels((s) => s.pollJobs);
  const pollImage = useModels((s) => s.pollImage);
  const connect = useSimulator((s) => s.connect);
  const online = useSimulator((s) => s.online);
  const finished = useRef(new Set<string>());

  useEffect(() => {
    void connect();
    void loadSettings();
    const tick = async () => {
      void pollImage();
      const jobs = await pollJobs();
      const done = jobs.filter((j) => j.state === "done" && !finished.current.has(j.id));
      if (done.length) {
        done.forEach((j) => finished.current.add(j.id));
        void connect();
      }
    };
    void tick();
    const timer = setInterval(() => void tick(), 1000);
    return () => clearInterval(timer);
  }, [connect, loadSettings, pollJobs, pollImage]);

  return (
    <section className="stage models-stage">
      <div className="models-column">
        <Segmented
          label="Source"
          value={source}
          options={[
            ["local", "On this machine"],
            ["hub", "Hugging Face"],
          ]}
          onChange={(value) => set({ source: value })}
        />
        {online === false ? <p className="placeholder models-empty">The local engine is not running. Start it with scripts/dev.sh.</p> : source === "local" ? <LocalList /> : <HubSearch />}
      </div>
    </section>
  );
}

function JobRow({ job }: { job: Job }) {
  const speed = useModels((s) => s.speeds[job.id]);
  const cancel = useModels((s) => s.cancel);
  const discard = useModels((s) => s.discard);
  const share = job.total_bytes ? Math.min(100, (100 * job.done_bytes) / job.total_bytes) : 0;
  const status = {
    queued: "Queued",
    downloading: [formatBytes(job.done_bytes), "of", formatBytes(job.total_bytes), speed ? `· ${formatBytes(speed)}/s` : "", formatRemaining(job, speed) ? `· ${formatRemaining(job, speed)}` : ""].filter(Boolean).join(" "),
    done: "Finished and checked",
    cancelled: "Stopped; it resumes if you request it again",
    error: job.error ?? "Error",
  }[job.state];
  const live = job.state === "queued" || job.state === "downloading";
  return (
    <div className="job">
      <div className="job-head">
        <strong title={job.id}>{job.id}</strong>
        {live ? <button className="pill" onClick={() => void cancel(job.id)}>Stop</button> : job.state !== "done" && <button className="pill" onClick={() => void discard(job.id)}>Delete</button>}
      </div>
      {live && <progress max={100} value={share} aria-label={`Progress of ${job.id}`} />}
      <p className={job.state === "error" ? "model-reason" : "muted-text"}>{status}</p>
    </div>
  );
}

export function ModelsPanel() {
  const { settings, tokenSource, jobs, settingsError, saveSettings, saveToken, forgetToken, image, cancelImage } = useModels();
  const connect = useSimulator((s) => s.connect);
  const [folder, setFolder] = useState("");
  const [download, setDownload] = useState<string | null>(null);
  const [token, setToken] = useState("");
  const local = ["localhost", "127.0.0.1", "[::1]"].includes(location.hostname);

  const refresh = async () => {
    await fetch("/api/models?refresh=1");
    await connect();
  };
  const replaceFolders = async (dirs: string[]) => {
    if (await saveSettings(dirs, settings?.download_is_default ? null : (settings?.download_dir ?? null))) {
      setFolder("");
      await refresh();
    }
  };
  const addFolder = (e: FormEvent) => {
    e.preventDefault();
    if (settings && folder.trim()) void replaceFolders([...settings.model_dirs, folder]);
  };
  const saveDownload = async (e: FormEvent) => {
    e.preventDefault();
    if (settings && download !== null && (await saveSettings(settings.model_dirs, download.trim() || null))) {
      setDownload(null);
      await refresh();
    }
  };
  const submitToken = async (e: FormEvent) => {
    e.preventDefault();
    if (token.trim() && (await saveToken(token))) setToken("");
  };

  return (
    <aside className="inspector">
      <h2 className="detail-title">Models</h2>
      <p className="detail-sub">Where they are searched for and where they are saved.</p>

      {settingsError && (
        <div className="issue error" role="alert">
          <span className="issue-icon" aria-hidden="true">!</span>
          <div><p>{settingsError}</p></div>
        </div>
      )}

      <h3>Folders</h3>
      <div className="inset padded form">
        {settings?.model_dirs.map((dir) => (
          <div className="folder" key={dir}>
            <span title={dir}>{dir}</span>
            <button className="pill" aria-label={`Remove ${dir}`} onClick={() => void replaceFolders(settings.model_dirs.filter((d) => d !== dir))}>Remove</button>
          </div>
        ))}
        <form className="inline-form" onSubmit={addFolder}>
          <input type="text" aria-label="Model folder" placeholder="/path/to/my/models" value={folder} onChange={(e) => setFolder(e.target.value)} />
          <button className="pill" disabled={!folder.trim()}>Add</button>
        </form>
        <p className="muted-text">Searched up to three levels inside each. Also always in:</p>
        <ul className="plain-paths">
          {settings?.default_dirs.map((dir) => <li key={dir} title={dir}>{dir}</li>)}
        </ul>
        <button className="pill" onClick={() => void refresh()}>Search again</button>
      </div>

      <h3>Download to</h3>
      <form className="inset padded form" onSubmit={saveDownload}>
        <input type="text" aria-label="Download folder" value={download ?? settings?.download_dir ?? ""} onChange={(e) => setDownload(e.target.value)} />
        {download !== null && <button className="pill">Save</button>}
      </form>

      <h3>Hugging Face account</h3>
      <div className="inset padded form">
        <p className="muted-text">
          {tokenSource === null && "No token. Enough for public models; gated ones need their license accepted and a read token."}
          {tokenSource === "app" && "Using the token saved on this machine."}
          {tokenSource === "environment" && "Using the token from the HF_TOKEN environment variable."}
          {tokenSource === "huggingface-cli" && "Using the token you signed in to huggingface-cli with."}
        </p>
        <form className="inline-form" onSubmit={submitToken}>
          <input type="password" autoComplete="off" aria-label="Hugging Face token" placeholder="Read token" value={token} onChange={(e) => setToken(e.target.value)} />
          <button className="pill" disabled={!token.trim()}>Save</button>
        </form>
        {tokenSource === "app" && <button className="pill" onClick={() => void forgetToken()}>Delete token</button>}
        {!local && <p className="model-reason">You are connected over the local network without encryption: type the token on the machine itself.</p>}
      </div>

      <SpeedPanel />

      {image.state !== "idle" && (
        <>
          <h3>Chip image</h3>
          <div className="inset padded form job">
            <div className="job-head">
              <strong title={image.name}>{image.name}</strong>
              {image.state === "writing" && <button className="pill" onClick={() => void cancelImage()}>Stop</button>}
            </div>
            {image.state === "writing" && <progress max={image.total || 1} value={image.done ?? 0} aria-label="Image progress" />}
            {image.state === "writing" && <p className="muted-text">{formatBytes(image.done ?? 0)} of {formatBytes(image.total ?? 0)}</p>}
            {image.state === "done" && image.report && <p className="muted-text">{formatBytes(image.report.bytes)}, for {image.report.chips.join(", ")} chips. Saved in {image.report.path}</p>}
            {image.state === "error" && <p className="model-reason">{image.error}</p>}
          </div>
        </>
      )}

      {jobs.length > 0 && (
        <>
          <h3>Downloads</h3>
          <div className="inset padded form">
            {jobs.map((job) => <JobRow key={job.id} job={job} />)}
          </div>
        </>
      )}
    </aside>
  );
}

const OPERATION_NAMES: Record<string, string> = {
  AttnSum: "Attention",
  DeltaStep: "DeltaNet",
  Dconv: "Convolution",
  Matvec: "Matrices",
  RmsGated: "Gated normalization",
  RmsNorm: "Normalization",
  Rope: "RoPE",
  Silu: "SiLU",
  Gelu: "GELU",
  Moe: "Experts",
  SigmoidMoe: "Experts",
  Mla: "Latent attention",
};

/** The tokens per second of the generated circuit, counted cycle by cycle, against the target. */
function SpeedPanel() {
  const speed = useModels((s) => s.speed);
  const measuring = useModels((s) => s.measuring);
  const params = useDesign((s) => s.params);
  if (measuring) return <p className="muted-text">Counting the circuit's cycles…</p>;
  if (!speed) return null;
  const ghz = Number(params.freq_ghz);
  const measured = (ghz * 1e9) / speed.total;
  const target = Number(params.target_tok_s);
  const top = Math.max(...speed.ops.map((o) => o.cycles), 1);
  return (
    <>
      <h3>Circuit speed</h3>
      <div className="inset padded form job">
        <div className="job-head">
          <strong title={speed.name}>{speed.name}</strong>
        </div>
        <p className="muted-text">
          {formatNumber(measured)} tok/s at {formatNumber(ghz)} GHz with {speed.cells.toLocaleString(locale())} cells, {speed.units} units and {speed.lanes} lanes, at position {speed.position.toLocaleString(locale())}
          {target > 0 ? `. Target: ${formatNumber(target)} tok/s (${formatNumber(target / measured)} times more).` : "."}
        </p>
        <div className="lab-bars" role="table" aria-label="Cycles per operation">
          {speed.ops.slice(0, 8).map((o) => (
            <div className="lab-row" role="row" key={o.name} title={`${o.name}: ${o.count} times`}>
              <span role="cell" className="lab-label">{OPERATION_NAMES[o.name] ?? o.name}</span>
              <span role="cell" className="lab-track"><i style={{ width: `${(100 * o.cycles) / top}%` }} /></span>
              <span role="cell" className="lab-value">{(o.cycles / 1e6).toLocaleString(locale(), { maximumFractionDigits: 1 })} M</span>
            </div>
          ))}
        </div>
      </div>
    </>
  );
}
