import { DEMO, REPOSITORY } from "../demo";
import { useEffect, useRef, useState, type ChangeEvent } from "react";
import { useDesign, type Appearance } from "../store";
import { datasheet } from "../datasheet";
import { download } from "../download";
import { IconButton } from "./controls";
import { ProjectWizard } from "../wizard/ProjectWizard";
import { Icon } from "./icons";
import { useT, type Language } from "../i18n";
import type { View } from "../store";

const VIEWS: [View, string][] = [
  ["optimize", "Optimise"],
  ["floorplan", "Floorplan"],
  ["logic", "How it works"],
  ["schematic", "Schematic"],
  ["simulator", "Simulator"],
  ["images", "Images"],
  ["explorer", "Explorer"],
  ["numeric", "Numerics"],
  ["models", "Models"],
  ["parameters", "Parameters"],
];

const stroke = { fill: "none", stroke: "currentColor", strokeWidth: 1.6, strokeLinecap: "round", strokeLinejoin: "round" } as const;

function SavePopover({ onClose }: { onClose: () => void }) {
  const designName = useDesign((s) => s.designName);
  const saveDesign = useDesign((s) => s.saveDesign);
  const t = useT();
  const [name, setName] = useState(designName === "Untitled" ? "" : designName);
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => input.current?.select(), []);

  const submit = () => {
    if (name.trim()) saveDesign(name.trim());
    onClose();
  };
  return (
    <div className="popover" role="dialog" aria-label={t("Save design")}>
      <label>
        {t("Design name")}
        <input
          ref={input}
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") submit();
            if (e.key === "Escape") onClose();
          }}
        />
      </label>
      <div className="popover-actions">
        <button onClick={onClose}>{t("Cancel")}</button>
        <button className="primary" disabled={!name.trim()} onClick={submit}>
          {t("Save")}
        </button>
      </div>
    </div>
  );
}

function DesignMenu() {
  const t = useT();
  const presets = useDesign((s) => s.presets);
  const saved = useDesign((s) => s.saved);
  const designName = useDesign((s) => s.designName);
  const applyPreset = useDesign((s) => s.applyPreset);
  const loadDesign = useDesign((s) => s.loadDesign);

  const onChange = (e: ChangeEvent<HTMLSelectElement>) => {
    const [kind, id] = [e.target.value.slice(0, 2), e.target.value.slice(2)];
    if (kind === "p:") applyPreset(id);
    if (kind === "s:") loadDesign(id);
    e.target.value = "";
  };
  return (
    <label className="design-menu">
      <span>{designName === "Untitled" ? t(designName) : designName}</span>
      <svg viewBox="0 0 10 14" aria-hidden="true">
        <path d="M2 5l3-3 3 3M2 9l3 3 3-3" {...stroke} strokeWidth={1.4} />
      </svg>
      <select aria-label={t("Open design")} value="" onChange={onChange}>
        <option value="" disabled hidden />
        <optgroup label={t("Templates")}>
          {presets.map((p) => (
            <option key={p.id} value={`p:${p.id}`}>
              {t(p.name)}
            </option>
          ))}
        </optgroup>
        {saved.length > 0 && (
          <optgroup label={t("My designs")}>
            {saved.map((d) => (
              <option key={d.name} value={`s:${d.name}`}>
                {d.name}
              </option>
            ))}
          </optgroup>
        )}
      </select>
    </label>
  );
}

function AppearanceMenu() {
  const t = useT();
  const appearance = useDesign((s) => s.appearance);
  const setAppearance = useDesign((s) => s.setAppearance);
  return (
    <label className="icon-select" title={t("Appearance")}>
      <svg viewBox="0 0 20 20" aria-hidden="true">
        {appearance === "light" && (
          <path d="M10 6.5a3.5 3.5 0 1 0 0 7 3.5 3.5 0 0 0 0-7ZM10 2.5v1.6M10 15.9v1.6M2.5 10h1.6M15.9 10h1.6M4.7 4.7l1.1 1.1M14.2 14.2l1.1 1.1M15.3 4.7l-1.1 1.1M5.8 14.2l-1.1 1.1" {...stroke} />
        )}
        {appearance === "dark" && <path d="M16 12.4A6.5 6.5 0 0 1 7.6 4a6.5 6.5 0 1 0 8.4 8.4Z" {...stroke} />}
        {appearance === "auto" && (
          <>
            <circle cx="10" cy="10" r="6.5" {...stroke} />
            <path d="M10 3.5a6.5 6.5 0 0 1 0 13Z" fill="currentColor" />
          </>
        )}
      </svg>
      <select aria-label={t("Appearance")} value={appearance} onChange={(e) => setAppearance(e.target.value as Appearance)}>
        <option value="auto">{t("Automatic")}</option>
        <option value="light">{t("Light")}</option>
        <option value="dark">{t("Dark")}</option>
      </select>
    </label>
  );
}

function LanguageMenu() {
  const language = useDesign((s) => s.language);
  const setLanguage = useDesign((s) => s.setLanguage);
  const t = useT();
  return (
    <label className="icon-select language" title={t("Language")}>
      <span aria-hidden="true">{language.toUpperCase()}</span>
      <select aria-label={t("Language")} value={language} onChange={(e) => setLanguage(e.target.value as Language)}>
        <option value="en">English</option>
        <option value="es">Español</option>
      </select>
    </label>
  );
}

export function Toolbar() {
  const t = useT();
  const undo = useDesign((s) => s.undo);
  const redo = useDesign((s) => s.redo);
  const canUndo = useDesign((s) => s.past.length > 0);
  const canRedo = useDesign((s) => s.future.length > 0);
  const view = useDesign((s) => s.view);
  const setView = useDesign((s) => s.setView);
  const baseline = useDesign((s) => s.baseline);
  const pinBaseline = useDesign((s) => s.pinBaseline);
  const clearBaseline = useDesign((s) => s.clearBaseline);
  const [saving, setSaving] = useState(false);
  const [creating, setCreating] = useState(false);
  const exportDatasheet = () => {
    const { report, designName, model, params } = useDesign.getState();
    if (!report || !model) return;
    download(datasheet(designName, model.name, report, params as unknown as Record<string, unknown>), `${designName.replace(/[^\p{L}\p{N}]+/gu, "_")}_datasheet.md`, "text/markdown");
  };

  return (
    <header className="toolbar">
      <div className="toolbar-side">
        <strong className="app-name">llm2chip</strong>
        <IconButton label={t("New project")} onClick={() => setCreating(true)}>
          <path d="M10 4.5v11M4.5 10h11" {...stroke} />
        </IconButton>
        <DesignMenu />
        {creating && <ProjectWizard onClose={() => setCreating(false)} />}
      </div>
      <div className="segmented views" role="radiogroup" aria-label={t("View")}>
        {VIEWS.map(([id, label]) => (
          <button key={id} role="radio" aria-checked={view === id} aria-label={t(label)} title={t(label)} className={view === id ? "on" : ""} onClick={() => setView(id)}>
            <Icon name={id} />
            <span className="view-label">{t(label)}</span>
          </button>
        ))}
      </div>
      <div className="toolbar-side end">
        <IconButton label={t("Export datasheet")} onClick={exportDatasheet}>
          <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2h6l3 3v9H4zM10 2v3h3M6 8.5h5M6 11h5" {...stroke} /></svg>
        </IconButton>
        <IconButton label={t("Undo")} disabled={!canUndo} onClick={undo}>
          <path d="M7.5 5 4 8.5 7.5 12M4.5 8.5H12a4 4 0 0 1 0 8h-1.5" {...stroke} />
        </IconButton>
        <IconButton label={t("Redo")} disabled={!canRedo} onClick={redo}>
          <path d="M12.5 5 16 8.5 12.5 12M15.5 8.5H8a4 4 0 0 0 0 8h1.5" {...stroke} />
        </IconButton>
        <AppearanceMenu />
        <LanguageMenu />
        {DEMO && (
          <a className="pill" href={REPOSITORY} target="_blank" rel="noreferrer" title={t("The source, and the app that runs real models on your computer")}>
            {t("GitHub")}
          </a>
        )}
        <button className={baseline ? "pill on" : "pill"} onClick={baseline ? clearBaseline : pinBaseline}>
          {t(baseline ? "Remove" : "Pin")}<span className="wide-label">{t(baseline ? " baseline" : " as baseline")}</span>
        </button>
        <div className="anchor">
          <button className="pill" onClick={() => setSaving(!saving)}>
            {t("Save")}
          </button>
          {saving && <SavePopover onClose={() => setSaving(false)} />}
        </div>
      </div>
    </header>
  );
}
