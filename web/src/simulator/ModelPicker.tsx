import { useState } from "react";
import { formatNumber } from "../format";
import { useT } from "../i18n";
import { useDesign } from "../store";
import { DOWNLOAD_LIMIT, memoryWarning } from "./browserModels";
import { useSimulator } from "./simulatorStore";

const MB = 2 ** 20;

/** The online demo's models: choosing one makes it the design's model, so the chip simulated
 * is one built for it. */
export function ModelPicker() {
  const t = useT();
  const models = useSimulator((s) => s.browserModels);
  const refusal = useSimulator((s) => s.browserRefusal);
  const path = useSimulator((s) => s.path);
  const loading = useSimulator((s) => s.loading);
  const addBrowserModel = useSimulator((s) => s.addBrowserModel);
  const designPath = useDesign((s) => s.model?.path);
  const choose = useSimulator((s) => s.chooseBrowserModel);
  const forget = useSimulator((s) => s.forgetBrowserModel);
  const kept = useSimulator((s) => s.kept);
  const noRoom = useSimulator((s) => s.noRoom);
  const keptTotal = useSimulator((s) => s.keptTotal);
  const chosen = models.find((m) => m.id === path);
  const warning = chosen ? memoryWarning(chosen.bytes) : null;
  const [repo, setRepo] = useState("");
  const [checking, setChecking] = useState(false);

  const check = async () => {
    setChecking(true);
    const model = await addBrowserModel({ repo });
    setChecking(false);
    if (model) {
      setRepo("");
      choose(model);
    }
  };

  return (
    <>
      <p className="muted-text">{t(`The online demo runs models of up to ${formatNumber(DOWNLOAD_LIMIT / MB)} MB in your browser, on one thread and without a GPU; larger ones need the app on your computer.`)}</p>
      {models.length === 0 && <p className="muted-text">{t("Finding the models…")}</p>}
      <div className="browser-models">
        {models.map((model) => (
          <div key={model.id} className="browser-model-row">
            <button className={designPath === model.id ? "browser-model on" : "browser-model"} disabled={loading} onClick={() => choose(model)}>
              <strong>{model.name}</strong>
              <span className="muted-text">
                {formatNumber(model.bytes / MB)} MB · {model.architecture}
                {model.credit ? ` · ${model.credit}` : ""}
                {(kept[model.id] ?? 0) > 0 ? ` · ${t("kept in this browser")}` : ""}
              </span>
            </button>
            {(kept[model.id] ?? 0) > 0 && (
              <button className="browser-forget" disabled={loading} title={t("Delete the copy kept in this browser")} aria-label={t("Delete the copy kept in this browser")} onClick={() => void forget(model)}>
                ×
              </button>
            )}
          </div>
        ))}
      </div>
      {warning && <p className="note-warning">{t(warning)}</p>}
      <form
        className="browser-repo"
        onSubmit={(e) => {
          e.preventDefault();
          if (repo.trim()) void check();
        }}
      >
        <input type="text" aria-label={t("Another model on Hugging Face")} placeholder={t("Another model on Hugging Face: owner/name")} value={repo} onChange={(e) => setRepo(e.target.value)} />
        <button className="pill" disabled={checking || !repo.trim()}>
          {t(checking ? "Checking…" : "Try it")}
        </button>
      </form>
      {refusal && <p className="note-warning">{t(refusal)}</p>}
      {noRoom && <p className="muted-text">{t("There was no room to keep a copy in this browser: it will download again next time.")}</p>}
      {keptTotal > 0 && (
        <p className="muted-text browser-kept">
          {t(`Models kept in this browser: ${formatNumber(keptTotal / MB)} MB.`)}{" "}
          <button className="link-button" disabled={loading} onClick={() => void forget(null)}>
            {t("Delete them all")}
          </button>
        </p>
      )}
      {path && !models.some((m) => m.id === path) && <p className="muted-text">{t("Choose one of these models to simulate it.")}</p>}
    </>
  );
}
