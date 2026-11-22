import { DEMO, INSTALL_GUIDE, REPOSITORY } from "../demo";
import { useT } from "../i18n";
import type { View } from "../store";

/** What a part that needs the engine says when there is none: how to start it, or, in the
 * online demo, that it runs on the app installed on one's own computer. */
export function EngineMissing() {
  const t = useT();
  return (
    <div className="issue warning" role="status">
      <span className="issue-icon" aria-hidden="true">i</span>
      {DEMO ? (
        <div>
          <strong>{t("This needs the app on your computer.")}</strong>
          <p>
            {t("It runs real models with their weights, which the online demo does not download or run.")}{" "}
            <a href={INSTALL_GUIDE} target="_blank" rel="noreferrer">{t("How to install it")}</a>
          </p>
        </div>
      ) : (
        <div>
          <strong>{t("The engine is not running.")}</strong>
          <p>{t("Start it with this command in the project folder")}: scripts/dev.sh</p>
        </div>
      )}
    </div>
  );
}

/** The views that only work with the engine, in the online demo: what they do and where to get them. */
export const ENGINE_VIEWS = new Set<View>(["images", "viewer", "optimize", "numeric", "models"]);

function useWhat(view: View): [string, string] {
  const t = useT();
  switch (view) {
    case "images":
      return [t("Images"), t("Draws an image from a text, or retouches a region of one, with Show-o running on the chip's arithmetic.")];
    case "viewer":
      return [t("Chip at work"), t("Plays back, in 3D and cycle by cycle, a run of the chip that the engine simulated gate by gate: the dies, the tokens going through the layers, the KV cache and the links between chips.")];
    case "optimize":
      return [t("Automatic optimisation"), t("Searches numbers of chips, ways of sharing, units, lanes and cells, measuring each candidate on its generated circuit with the model's weights, and recommends one.")];
    case "numeric":
      return [t("Numerics lab"), t("Runs a real model with the chip's integer arithmetic and compares it with floating point, layer by layer and token by token.")];
    default:
      return [t("Models"), t("Finds the models on your computer and downloads others from Hugging Face. In the online demo, the sidebar can still load any model's config.json.")];
  }
}

export function DemoStage({ view }: { view: View }) {
  const t = useT();
  const [title, what] = useWhat(view);
  return (
    <section className="stage demo-stage">
      <div className="demo-card">
        <h2>{title}</h2>
        <p>{what}</p>
        <p className="muted-text">{t("It needs llm2chip's engine on your computer, which reads the model's weights and simulates the whole circuit. This online demo runs only what fits in the browser: the floorplan, how it works, the schematic simulated gate by gate, the explorer, the export to Verilog and VHDL, and the simulator with small models.")}</p>
        <div className="demo-links">
          <a className="pill primary-pill" href={INSTALL_GUIDE} target="_blank" rel="noreferrer">{t("How to install it")}</a>
          <a className="pill" href={REPOSITORY} target="_blank" rel="noreferrer">{t("Source on GitHub")}</a>
        </div>
      </div>
    </section>
  );
}
