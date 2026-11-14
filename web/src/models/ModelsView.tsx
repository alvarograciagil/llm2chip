import { useEffect, useRef, useState, type FormEvent } from "react";
import { Segmented } from "../components/controls";
import { useSimulator, type ServerModel } from "../simulator/simulatorStore";
import { formatNumber, locale } from "../format";
import { useDesign } from "../store";
import { useT } from "../i18n";
import { formatBytes, formatCount, formatRemaining, useModels, type Adaptation, type Details, type Job } from "./modelsStore";

/** With `family`, only the architecture is known: the repository's config may refuse it later. */
function Tier({ adaptation, known = true, family = false }: { adaptation: Adaptation; known?: boolean; family?: boolean }) {
  const t = useT();
  const [label, tone] = !known ? ["No data", "none"] : adaptation.simulate ? [family ? "Runnable family" : "Runnable", "full"] : adaptation.design ? [adaptation.approximate ? "Approximate design" : "Design only", "part"] : ["Not adaptable", "none"];
  return <span className={`tier ${tone}`}>{t(label)}</span>;
}

/** A path as the person reads it: under their home folder it starts with `~`. */
export function shortPath(path: string, home: string | undefined): string {
  if (!home || !path.startsWith(home)) return path;
  const rest = path.slice(home.length);
  return rest === "" || rest.startsWith("/") || rest.startsWith("\\") ? `~${rest}` : path;
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
  const t = useT();
  const home = useModels((s) => s.settings?.home);
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
    return <p className="placeholder models-empty">{t("No models with safetensors weights in the folders. Add a folder in the right panel or search Hugging Face.")}</p>;
  }
  return (
    <ul className="model-list">
      {models.map((m) => (
        <li key={m.path} className="model-row">
          <div className="model-main">
            <div className="model-title">
              <strong>{m.name}</strong>
              <Tier adaptation={m.adaptation} />
              {m.images?.reads && <span className="tier" title={t("Its image encoder runs in the simulator: an image can be attached to the question.")}>{t("Reads images")}</span>}
              {m.images?.makes && <span className="tier" title={t("Its image decoder runs in the simulator: it can generate images from text.")}>{t("Generates images")}</span>}
            </div>
            <p className="muted-text">
              {m.architecture} · {formatBytes(m.size_gb * 1e9)}
              {m.adaptation.parameters ? ` · ${t(`${formatCount(m.adaptation.parameters)} parameters · ${m.adaptation.layers} layers`)}` : ""}
            </p>
            {m.adaptation.reason && <p className="model-reason">{t(m.adaptation.reason)}</p>}
            {m.images?.note && <p className="model-reason">{t(m.images.note)}</p>}
            <p className="model-path" title={shortPath(m.path, home)}>{shortPath(m.path, home)}</p>
          </div>
          <div className="model-actions">
            <button className="pill" disabled={!m.adaptation.design} onClick={() => toDesigner(m.name, m.config, m.adaptation, m.path)}>{t("Design chip")}</button>
            <button className="pill" disabled={!m.supported} onClick={() => simulate(m)}>{t("Simulate")}</button>
            <button className="pill" disabled={!m.supported || writing} onClick={() => makeImage(m)} title={t("Prepares the chip's boot image: microcode, weights and tables the firmware loads at power-on")}>{t("Boot image")}</button>
            <button className="pill" disabled={!m.supported || measuring} onClick={() => measure(m)}>{t("Measure speed")}</button>
          </div>
        </li>
      ))}
    </ul>
  );
}

function RepoDetails({ details }: { details: Details }) {
  const t = useT();
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
        {details.architecture ?? t("unknown architecture")}
        {adaptation.parameters ? ` · ${t(`${formatCount(adaptation.parameters)} parameters · ${adaptation.layers} layers`)}` : ""}
        {details.gated ? ` · ${t("gated access")}` : ""}
      </p>
      {adaptation.approximate && adaptation.design && <p className="model-reason">{t("The circuit is estimated from the shape of its layers. Its activation or extra norms are not modelled.")}</p>}
      {adaptation.reason && <p className="model-reason">{t(adaptation.reason)}</p>}
      {details.files.length > 0 && (
        <p className="muted-text">
          {t(`${details.files.filter((f) => f.name.endsWith(".safetensors")).length} weight file(s) · ${formatBytes(details.total_bytes)} in total`)}
          {details.free_bytes !== null ? ` · ${t(`${formatBytes(details.free_bytes)} free`)}` : ""}
        </p>
      )}
      {details.resumable_bytes > 0 && <p className="muted-text">{t(`${formatBytes(details.resumable_bytes)} from an earlier download will be reused.`)}</p>}
      <div className="model-actions inline">
        <button className="pill" disabled={!adaptation.design || !config} onClick={() => config && toDesigner(details.id, config, adaptation)}>{t("Design chip without downloading")}</button>
        <button className="pill primary-pill" disabled={details.present || running || !details.files.length || short} onClick={() => void download(details.id)}>
          {details.present ? t("Already downloaded") : running ? t("Downloading…") : short ? t("Does not fit on the disk") : t(`Download ${formatBytes(details.total_bytes)}`)}
        </button>
        {details.resumable_bytes > 0 && !running && <button className="pill" onClick={() => void discard(details.id)}>{t("Delete the partial download")}</button>}
      </div>
    </div>
  );
}

function HubSearch() {
  const t = useT();
  const { query, searched, hits, searching, selected, details, error, set, search, select } = useModels();
  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (query.trim()) void search();
  };
  const direct = selected !== null && !hits.some((h) => h.id === selected);

  return (
    <>
      <form className="model-search" onSubmit={submit}>
        <input type="search" aria-label={t("Search Hugging Face")} placeholder={t("Search, or type author/model")} value={query} onChange={(e) => set({ query: e.target.value })} />
        <button className="pill primary-pill" disabled={!query.trim() || searching}>{searching ? t("Searching…") : t("Search")}</button>
      </form>
      {error && (
        <div className="issue error" role="alert">
          <span className="issue-icon" aria-hidden="true">!</span>
          <div><p>{t(error)}</p></div>
        </div>
      )}
      {searched !== null && !searching && !hits.length && !selected && !error && <p className="placeholder models-empty">{t(`No results for “${searched}” with safetensors weights.`)}</p>}
      {direct && (
        <ul className="model-list">
          <li className="model-row open">
            <div className="model-main">
              <div className="model-title">
                <strong>{selected}</strong>
                {details && <Tier adaptation={details.adaptation} />}
              </div>
              {details ? <RepoDetails details={details} /> : !error && <p className="muted-text">{t("Looking it up…")}</p>}
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
                  <p className="muted-text">{hit.architecture ?? t("unknown architecture")} · {t(`${formatCount(hit.downloads)} downloads · ${formatCount(hit.likes)} likes`)}</p>
                  {open && (details ? <RepoDetails details={details} /> : !error && <p className="muted-text">{t("Looking it up…")}</p>)}
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
  const t = useT();
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
          label={t("Source")}
          value={source}
          options={[
            ["local", t("On this machine")],
            ["hub", "Hugging Face"],
          ]}
          onChange={(value) => set({ source: value })}
        />
        {online === false ? <p className="placeholder models-empty">{t("The local engine is not running. Start it with scripts/dev.sh.")}</p> : source === "local" ? <LocalList /> : <HubSearch />}
      </div>
    </section>
  );
}

function JobRow({ job }: { job: Job }) {
  const t = useT();
  const speed = useModels((s) => s.speeds[job.id]);
  const cancel = useModels((s) => s.cancel);
  const discard = useModels((s) => s.discard);
  const share = job.total_bytes ? Math.min(100, (100 * job.done_bytes) / job.total_bytes) : 0;
  const status = {
    queued: t("Queued"),
    downloading: [`${formatBytes(job.done_bytes)} ${t("of")} ${formatBytes(job.total_bytes)}`, speed ? `· ${formatBytes(speed)}/s` : "", formatRemaining(job, speed) ? `· ${t(formatRemaining(job, speed))}` : ""].filter(Boolean).join(" "),
    done: t("Finished and checked"),
    cancelled: t("Stopped; it resumes if you request it again"),
    error: job.error ? t(job.error) : "Error",
  }[job.state];
  const live = job.state === "queued" || job.state === "downloading";
  return (
    <div className="job">
      <div className="job-head">
        <strong title={job.id}>{job.id}</strong>
        {live ? <button className="pill" onClick={() => void cancel(job.id)}>{t("Stop")}</button> : job.state !== "done" && <button className="pill" onClick={() => void discard(job.id)}>{t("Delete")}</button>}
      </div>
      {live && <progress max={100} value={share} aria-label={t(`Progress of ${job.id}`)} />}
      <p className={job.state === "error" ? "model-reason" : "muted-text"}>{status}</p>
    </div>
  );
}

export function ModelsPanel() {
  const t = useT();
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
  const keptDownload = () => (settings?.download_is_default ? null : (settings?.download_dir ?? null));
  const replaceFolders = async (dirs: string[]) => {
    if (await saveSettings(dirs, keptDownload())) {
      setFolder("");
      await refresh();
    }
  };
  const toggleDefault = async (dir: string, on: boolean) => {
    if (!settings) return;
    const disabled = on ? settings.disabled_defaults.filter((d) => d !== dir) : [...settings.disabled_defaults, dir];
    if (await saveSettings(settings.model_dirs, keptDownload(), disabled)) await refresh();
  };
  const home = settings?.home;
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
      <h2 className="detail-title">{t("Models")}</h2>
      <p className="detail-sub">{t("Where they are searched for and where they are saved.")}</p>

      {settingsError && (
        <div className="issue error" role="alert">
          <span className="issue-icon" aria-hidden="true">!</span>
          <div><p>{t(settingsError)}</p></div>
        </div>
      )}

      <h3>{t("Folders")}</h3>
      <div className="inset padded form">
        <p className="muted-text">{t("Searched up to three levels inside each. The usual places can be turned off; the download folder is always searched.")}</p>
        {settings?.default_dirs.map((dir) => {
          const on = !settings.disabled_defaults.includes(dir);
          const isDownload = dir === settings.download_dir;
          return (
            <label className="check toggle-folder" key={dir} title={shortPath(dir, home)}>
              <input type="checkbox" checked={on || isDownload} disabled={isDownload} onChange={(e) => void toggleDefault(dir, e.target.checked)} />
              <span>{shortPath(dir, home)}</span>
            </label>
          );
        })}
        {settings?.model_dirs.map((dir) => (
          <div className="folder" key={dir}>
            <span title={shortPath(dir, home)}>{shortPath(dir, home)}</span>
            <button className="pill" aria-label={t(`Remove ${shortPath(dir, home)}`)} onClick={() => void replaceFolders(settings.model_dirs.filter((d) => d !== dir))}>{t("Remove")}</button>
          </div>
        ))}
        <form className="inline-form" onSubmit={addFolder}>
          <input type="text" aria-label={t("Model folder")} placeholder={t("~/path/to/my/models")} value={folder} onChange={(e) => setFolder(e.target.value)} />
          <button className="pill" disabled={!folder.trim()}>{t("Add")}</button>
        </form>
        <p className="muted-text">{t("Folders can also be given before starting the engine, in LLM2CHIP_MODEL_DIRS, separated as in PATH.")}</p>
        <button className="pill" onClick={() => void refresh()}>{t("Search again")}</button>
      </div>

      <h3>{t("Download to")}</h3>
      <form className="inset padded form" onSubmit={saveDownload}>
        <input type="text" aria-label={t("Download folder")} value={download ?? shortPath(settings?.download_dir ?? "", home)} onChange={(e) => setDownload(e.target.value)} />
        {download !== null && <button className="pill">{t("Save")}</button>}
      </form>

      <h3>{t("Hugging Face account")}</h3>
      <div className="inset padded form">
        <p className="muted-text">
          {tokenSource === null && t("No token. Enough for public models; gated ones need their license accepted and a read token.")}
          {tokenSource === "app" && t("Using the token saved on this machine.")}
          {tokenSource === "environment" && t("Using the token from the HF_TOKEN environment variable.")}
          {tokenSource === "huggingface-cli" && t("Using the token you signed in to huggingface-cli with.")}
        </p>
        <form className="inline-form" onSubmit={submitToken}>
          <input type="password" autoComplete="off" aria-label={t("Hugging Face token")} placeholder={t("Read token")} value={token} onChange={(e) => setToken(e.target.value)} />
          <button className="pill" disabled={!token.trim()}>{t("Save")}</button>
        </form>
        {tokenSource === "app" && <button className="pill" onClick={() => void forgetToken()}>{t("Delete token")}</button>}
        {!local && <p className="model-reason">{t("You are connected over the local network without encryption: type the token on the machine itself.")}</p>}
      </div>

      <SpeedPanel />

      {image.state !== "idle" && (
        <>
          <h3>{t("Chip image")}</h3>
          <div className="inset padded form job">
            <div className="job-head">
              <strong title={image.name}>{image.name}</strong>
              {image.state === "writing" && <button className="pill" onClick={() => void cancelImage()}>{t("Stop")}</button>}
            </div>
            {image.state === "writing" && <progress max={image.total || 1} value={image.done ?? 0} aria-label={t("Image progress")} />}
            {image.state === "writing" && <p className="muted-text">{`${formatBytes(image.done ?? 0)} ${t("of")} ${formatBytes(image.total ?? 0)}`}</p>}
            {image.state === "done" && image.report && <p className="muted-text">{t(`${formatBytes(image.report.bytes)}, for ${image.report.chips.join(", ")} chips. Saved in ${shortPath(image.report.path, home)}`)}</p>}
            {image.state === "error" && <p className="model-reason">{t(image.error ?? "")}</p>}
          </div>
        </>
      )}

      {jobs.length > 0 && (
        <>
          <h3>{t("Downloads")}</h3>
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
  const t = useT();
  const speed = useModels((s) => s.speed);
  const measuring = useModels((s) => s.measuring);
  const params = useDesign((s) => s.params);
  if (measuring) return <p className="muted-text">{t("Counting the circuit's cycles…")}</p>;
  if (!speed) return null;
  const ghz = Number(params.freq_ghz);
  const measured = (ghz * 1e9) / speed.total;
  const target = Number(params.target_tok_s);
  const top = Math.max(...speed.ops.map((o) => o.cycles), 1);
  return (
    <>
      <h3>{t("Circuit speed")}</h3>
      <div className="inset padded form job">
        <div className="job-head">
          <strong title={speed.name}>{speed.name}</strong>
        </div>
        <p className="muted-text">
          {t(`${formatNumber(measured)} tok/s at ${formatNumber(ghz)} GHz with ${speed.cells.toLocaleString(locale())} cells, ${speed.units} units and ${speed.lanes} lanes, at position ${speed.position.toLocaleString(locale())}`)}
          {target > 0 ? `. ${t(`Target: ${formatNumber(target)} tok/s (${formatNumber(target / measured)} times more).`)}` : "."}
        </p>
        <div className="lab-bars" role="table" aria-label={t("Cycles per operation")}>
          {speed.ops.slice(0, 8).map((o) => (
            <div className="lab-row" role="row" key={o.name} title={t(`${o.name}: ${o.count} times`)}>
              <span role="cell" className="lab-label">{t(OPERATION_NAMES[o.name] ?? o.name)}</span>
              <span role="cell" className="lab-track"><i style={{ width: `${(100 * o.cycles) / top}%` }} /></span>
              <span role="cell" className="lab-value">{(o.cycles / 1e6).toLocaleString(locale(), { maximumFractionDigits: 1 })} M</span>
            </div>
          ))}
        </div>
      </div>
    </>
  );
}
