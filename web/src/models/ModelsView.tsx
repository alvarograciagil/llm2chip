import { useEffect, useRef, useState, type FormEvent } from "react";
import { Segmented } from "../components/controls";
import { useSimulator, type ServerModel } from "../simulator/simulatorStore";
import { formatNumber } from "../format";
import { useDesign } from "../store";
import { formatBytes, formatCount, formatRemaining, useModels, type Adaptation, type Details, type Job } from "./modelsStore";

/** With `family`, only the architecture is known: the repository's config may refuse it later. */
function Tier({ adaptation, known = true, family = false }: { adaptation: Adaptation; known?: boolean; family?: boolean }) {
  const [label, tone] = !known ? ["Sin datos", "none"] : adaptation.simulate ? [family ? "Familia simulable" : "Simulable", "full"] : adaptation.design ? [adaptation.approximate ? "Diseño aproximado" : "Solo diseño", "part"] : ["No adaptable", "none"];
  return <span className={`tier ${tone}`}>{label}</span>;
}

function useToDesigner() {
  const addModel = useDesign((s) => s.addModel);
  const setParam = useDesign((s) => s.setParam);
  const setView = useDesign((s) => s.setView);
  return (name: string, config: string, adaptation: Adaptation) => {
    addModel(name, config);
    // The chip is sized for this model, with a sliver so that rounding never leaves it short.
    if (adaptation.parameters) setParam("max_params_g", (adaptation.parameters / 1e9) * 1.001);
    setView("plano");
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
    void startImage(m.path, Array.from({ length: dies }, (_, i) => i + 1), Number(params.context_tokens), Number(params.macs_per_die), Number(params.activation_bits), Number(params.vector_lanes_per_die), Number(params.vector_units_per_die));
  };
  const measureSpeed = useModels((s) => s.measureSpeed);
  const measuring = useModels((s) => s.measuring);
  const measure = (m: ServerModel) => {
    const { params } = useDesign.getState();
    void measureSpeed(m.path, Number(params.context_tokens), Number(params.macs_per_die), Number(params.activation_bits), Number(params.vector_lanes_per_die), Number(params.vector_units_per_die));
  };
  const simulate = (m: ServerModel) => {
    setSimulator({ path: m.path });
    setView("simulador");
  };

  if (!models.length) {
    return <p className="placeholder models-empty">No hay modelos con pesos safetensors en las carpetas. Añade una carpeta en el panel derecho o busca en Hugging Face.</p>;
  }
  return (
    <ul className="model-list">
      {models.map((m) => (
        <li key={m.path} className="model-row">
          <div className="model-main">
            <div className="model-title">
              <strong>{m.name}</strong>
              <Tier adaptation={m.adaptation} />
              {m.images?.reads && <span className="tier" title="Su codificador de imágenes corre en el simulador: se le puede adjuntar una imagen a la pregunta.">Lee imágenes</span>}
              {m.images?.makes && <span className="tier" title="Su decodificador de imágenes corre en el simulador: puede generar imágenes desde texto.">Genera imágenes</span>}
            </div>
            <p className="muted-text">
              {m.architecture} · {formatBytes(m.size_gb * 1e9)}
              {m.adaptation.parameters ? ` · ${formatCount(m.adaptation.parameters)} parámetros · ${m.adaptation.layers} capas` : ""}
            </p>
            {m.adaptation.reason && <p className="model-reason">{m.adaptation.reason}</p>}
            {m.images?.note && <p className="model-reason">{m.images.note}</p>}
            <p className="model-path" title={m.path}>{m.path}</p>
          </div>
          <div className="model-actions">
            <button className="pill" disabled={!m.adaptation.design} onClick={() => toDesigner(m.name, m.config, m.adaptation)}>Diseñar chip</button>
            <button className="pill" disabled={!m.supported} onClick={() => simulate(m)}>Simular</button>
            <button className="pill" disabled={!m.supported || writing} onClick={() => makeImage(m)} title="Prepara la imagen de arranque del chip: microcódigo, pesos y tablas que el firmware carga al encender">Imagen de arranque</button>
            <button className="pill" disabled={!m.supported || measuring} onClick={() => measure(m)}>Medir velocidad</button>
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
        {details.architecture ?? "arquitectura desconocida"}
        {adaptation.parameters ? ` · ${formatCount(adaptation.parameters)} parámetros · ${adaptation.layers} capas` : ""}
        {details.gated ? " · acceso restringido" : ""}
      </p>
      {adaptation.approximate && adaptation.design && <p className="model-reason">El circuito se estima con su forma de capas. Su activación o sus normas extra no se modelan.</p>}
      {adaptation.reason && <p className="model-reason">{adaptation.reason}</p>}
      {details.files.length > 0 && (
        <p className="muted-text">
          {details.files.filter((f) => f.name.endsWith(".safetensors")).length} archivo(s) de pesos · {formatBytes(details.total_bytes)} en total
          {details.free_bytes !== null ? ` · ${formatBytes(details.free_bytes)} libres` : ""}
        </p>
      )}
      {details.resumable_bytes > 0 && <p className="muted-text">Hay {formatBytes(details.resumable_bytes)} de una descarga anterior que se aprovecharán.</p>}
      <div className="model-actions inline">
        <button className="pill" disabled={!adaptation.design || !config} onClick={() => config && toDesigner(details.id, config, adaptation)}>Diseñar chip sin descargar</button>
        <button className="pill primary-pill" disabled={details.present || running || !details.files.length || short} onClick={() => void download(details.id)}>
          {details.present ? "Ya descargado" : running ? "Descargando…" : short ? "No cabe en el disco" : `Descargar ${formatBytes(details.total_bytes)}`}
        </button>
        {details.resumable_bytes > 0 && !running && <button className="pill" onClick={() => void discard(details.id)}>Borrar lo descargado a medias</button>}
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
        <input type="search" aria-label="Buscar en Hugging Face" placeholder="Buscar, o escribir autor/modelo" value={query} onChange={(e) => set({ query: e.target.value })} />
        <button className="pill primary-pill" disabled={!query.trim() || searching}>{searching ? "Buscando…" : "Buscar"}</button>
      </form>
      {error && (
        <div className="issue error" role="alert">
          <span className="issue-icon" aria-hidden="true">!</span>
          <div><p>{error}</p></div>
        </div>
      )}
      {searched !== null && !searching && !hits.length && !selected && !error && <p className="placeholder models-empty">Sin resultados para «{searched}» con pesos safetensors.</p>}
      {direct && (
        <ul className="model-list">
          <li className="model-row open">
            <div className="model-main">
              <div className="model-title">
                <strong>{selected}</strong>
                {details && <Tier adaptation={details.adaptation} />}
              </div>
              {details ? <RepoDetails details={details} /> : !error && <p className="muted-text">Consultando…</p>}
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
                  <p className="muted-text">{hit.architecture ?? "arquitectura desconocida"} · {formatCount(hit.downloads)} descargas · {formatCount(hit.likes)} me gusta</p>
                  {open && (details ? <RepoDetails details={details} /> : !error && <p className="muted-text">Consultando…</p>)}
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
          label="Origen"
          value={source}
          options={[
            ["local", "En este equipo"],
            ["hub", "Hugging Face"],
          ]}
          onChange={(value) => set({ source: value })}
        />
        {online === false ? <p className="placeholder models-empty">El motor local no está en marcha. Ábrelo con scripts/dev.sh.</p> : source === "local" ? <LocalList /> : <HubSearch />}
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
    queued: "En cola",
    downloading: [formatBytes(job.done_bytes), "de", formatBytes(job.total_bytes), speed ? `· ${formatBytes(speed)}/s` : "", formatRemaining(job, speed) ? `· ${formatRemaining(job, speed)}` : ""].filter(Boolean).join(" "),
    done: "Terminada y comprobada",
    cancelled: "Detenida; se reanuda si la vuelves a pedir",
    error: job.error ?? "Error",
  }[job.state];
  const live = job.state === "queued" || job.state === "downloading";
  return (
    <div className="job">
      <div className="job-head">
        <strong title={job.id}>{job.id}</strong>
        {live ? <button className="pill" onClick={() => void cancel(job.id)}>Detener</button> : job.state !== "done" && <button className="pill" onClick={() => void discard(job.id)}>Borrar</button>}
      </div>
      {live && <progress max={100} value={share} aria-label={`Progreso de ${job.id}`} />}
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
      <h2 className="detail-title">Modelos</h2>
      <p className="detail-sub">Dónde se buscan y dónde se guardan.</p>

      {settingsError && (
        <div className="issue error" role="alert">
          <span className="issue-icon" aria-hidden="true">!</span>
          <div><p>{settingsError}</p></div>
        </div>
      )}

      <h3>Carpetas</h3>
      <div className="inset padded form">
        {settings?.model_dirs.map((dir) => (
          <div className="folder" key={dir}>
            <span title={dir}>{dir}</span>
            <button className="pill" aria-label={`Quitar ${dir}`} onClick={() => void replaceFolders(settings.model_dirs.filter((d) => d !== dir))}>Quitar</button>
          </div>
        ))}
        <form className="inline-form" onSubmit={addFolder}>
          <input type="text" aria-label="Carpeta de modelos" placeholder="/ruta/a/mis/modelos" value={folder} onChange={(e) => setFolder(e.target.value)} />
          <button className="pill" disabled={!folder.trim()}>Añadir</button>
        </form>
        <p className="muted-text">Se busca hasta tres niveles dentro de cada una. También siempre en:</p>
        <ul className="plain-paths">
          {settings?.default_dirs.map((dir) => <li key={dir} title={dir}>{dir}</li>)}
        </ul>
        <button className="pill" onClick={() => void refresh()}>Buscar de nuevo</button>
      </div>

      <h3>Descargar en</h3>
      <form className="inset padded form" onSubmit={saveDownload}>
        <input type="text" aria-label="Carpeta de descargas" value={download ?? settings?.download_dir ?? ""} onChange={(e) => setDownload(e.target.value)} />
        {download !== null && <button className="pill">Guardar</button>}
      </form>

      <h3>Cuenta de Hugging Face</h3>
      <div className="inset padded form">
        <p className="muted-text">
          {tokenSource === null && "Sin token. Basta para modelos públicos; los restringidos piden aceptar su licencia y un token de lectura."}
          {tokenSource === "app" && "Usando el token guardado en este equipo."}
          {tokenSource === "entorno" && "Usando el token de la variable de entorno HF_TOKEN."}
          {tokenSource === "huggingface-cli" && "Usando el token con el que iniciaste sesión en huggingface-cli."}
        </p>
        <form className="inline-form" onSubmit={submitToken}>
          <input type="password" autoComplete="off" aria-label="Token de Hugging Face" placeholder="Token de lectura" value={token} onChange={(e) => setToken(e.target.value)} />
          <button className="pill" disabled={!token.trim()}>Guardar</button>
        </form>
        {tokenSource === "app" && <button className="pill" onClick={() => void forgetToken()}>Borrar token</button>}
        {!local && <p className="model-reason">Estás conectado por la red local sin cifrar: escribe el token desde el propio equipo.</p>}
      </div>

      <SpeedPanel />

      {image.state !== "idle" && (
        <>
          <h3>Imagen del chip</h3>
          <div className="inset padded form job">
            <div className="job-head">
              <strong title={image.name}>{image.name}</strong>
              {image.state === "writing" && <button className="pill" onClick={() => void cancelImage()}>Detener</button>}
            </div>
            {image.state === "writing" && <progress max={image.total || 1} value={image.done ?? 0} aria-label="Progreso de la imagen" />}
            {image.state === "writing" && <p className="muted-text">{formatBytes(image.done ?? 0)} de {formatBytes(image.total ?? 0)}</p>}
            {image.state === "done" && image.report && <p className="muted-text">{formatBytes(image.report.bytes)}, para {image.report.chips.join(", ")} chips. Guardada en {image.report.path}</p>}
            {image.state === "error" && <p className="model-reason">{image.error}</p>}
          </div>
        </>
      )}

      {jobs.length > 0 && (
        <>
          <h3>Descargas</h3>
          <div className="inset padded form">
            {jobs.map((job) => <JobRow key={job.id} job={job} />)}
          </div>
        </>
      )}
    </aside>
  );
}

const OPERATION_NAMES: Record<string, string> = {
  AttnSum: "Atención",
  DeltaStep: "DeltaNet",
  Dconv: "Convolución",
  Matvec: "Matrices",
  RmsGated: "Normalización con puerta",
  RmsNorm: "Normalización",
  Rope: "RoPE",
  Silu: "SiLU",
  Gelu: "GELU",
  Moe: "Expertos",
  SigmoidMoe: "Expertos",
  Mla: "Atención latente",
};

/** The tokens per second of the generated circuit, counted cycle by cycle, against the target. */
function SpeedPanel() {
  const speed = useModels((s) => s.speed);
  const measuring = useModels((s) => s.measuring);
  const params = useDesign((s) => s.params);
  if (measuring) return <p className="muted-text">Contando los ciclos del circuito…</p>;
  if (!speed) return null;
  const ghz = Number(params.freq_ghz);
  const measured = (ghz * 1e9) / speed.total;
  const target = Number(params.target_tok_s);
  const top = Math.max(...speed.ops.map((o) => o.cycles), 1);
  return (
    <>
      <h3>Velocidad del circuito</h3>
      <div className="inset padded form job">
        <div className="job-head">
          <strong title={speed.name}>{speed.name}</strong>
        </div>
        <p className="muted-text">
          {formatNumber(measured)} tok/s a {formatNumber(ghz)} GHz con {speed.cells.toLocaleString("es-ES")} celdas, {speed.units} unidades y {speed.lanes} carriles, en la posición {speed.position.toLocaleString("es-ES")}
          {target > 0 ? `. Objetivo: ${formatNumber(target)} tok/s (${formatNumber(target / measured)} veces más).` : "."}
        </p>
        <div className="lab-bars" role="table" aria-label="Ciclos por operación">
          {speed.ops.slice(0, 8).map((o) => (
            <div className="lab-row" role="row" key={o.name} title={`${o.name}: ${o.count} veces`}>
              <span role="cell" className="lab-label">{OPERATION_NAMES[o.name] ?? o.name}</span>
              <span role="cell" className="lab-track"><i style={{ width: `${(100 * o.cycles) / top}%` }} /></span>
              <span role="cell" className="lab-value">{(o.cycles / 1e6).toLocaleString("es-ES", { maximumFractionDigits: 1 })} M</span>
            </div>
          ))}
        </div>
      </div>
    </>
  );
}
