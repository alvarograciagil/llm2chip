import { useEffect, useRef, useState } from "react";
import { formatNumber } from "../format";
import { useDesign } from "../store";
import { useT } from "../i18n";
import { readSquare, regionBetween, regionIsUsable, useImages, type Region } from "./imagesStore";

/** The picture with the region marked on it: drag across it to mark another. */
function Marker({ preview, side, region, disabled, onRegion }: { preview: string; side: number; region: Region | null; disabled: boolean; onRegion: (r: Region | null) => void }) {
  const t = useT();
  const box = useRef<HTMLDivElement>(null);
  const [from, setFrom] = useState<[number, number] | null>(null);
  const [draft, setDraft] = useState<Region | null>(null);
  const point = (e: React.PointerEvent): [number, number] => {
    const rect = box.current!.getBoundingClientRect();
    return [((e.clientX - rect.left) / rect.width) * side, ((e.clientY - rect.top) / rect.height) * side];
  };
  const shown = draft ?? region;
  return (
    <div
      ref={box}
      className={disabled ? "marker disabled" : "marker"}
      onPointerDown={(e) => {
        if (disabled) return;
        e.currentTarget.setPointerCapture(e.pointerId);
        const p = point(e);
        setFrom(p);
        setDraft(regionBetween(p, p, side));
      }}
      onPointerMove={(e) => {
        if (from) setDraft(regionBetween(from, point(e), side));
      }}
      onPointerUp={(e) => {
        if (!from) return;
        const r = regionBetween(from, point(e), side);
        setFrom(null);
        setDraft(null);
        onRegion(regionIsUsable(r, side) ? r : null);
      }}
    >
      <img src={preview} alt={t("Image being retouched")} draggable={false} />
      {shown && (
        <div
          className="marker-region"
          style={{ left: `${(shown[0] / side) * 100}%`, top: `${(shown[1] / side) * 100}%`, width: `${((shown[2] - shown[0]) / side) * 100}%`, height: `${((shown[3] - shown[1]) / side) * 100}%` }}
        />
      )}
    </div>
  );
}

export function ImagesView() {
  const t = useT();
  const s = useImages();
  const usable = s.picture !== null && regionIsUsable(s.region, s.picture.side);
  const retouching = s.picture !== null;
  const ready = s.path !== "" && s.prompt.trim() !== "" && (!retouching || usable) && !s.running;
  const how = s.result?.chip_bits ? (s.result.magvit_on_chip ? `, ${t("all in the chip's arithmetic")} (${s.result.chip_bits} bits)` : `, ${t("the transformer and the sampling as on the chip")} (${s.result.chip_bits} bits)`) : `, ${t("in floating point")}`;
  return (
    <section className="stage sim-stage">
      <div className="images-column">
        <div className="images-pair">
          <figure className="picture">
            {s.picture ? (
              <Marker preview={s.picture.preview} side={s.picture.side} region={s.region} disabled={s.running} onRegion={(region) => s.set({ region })} />
            ) : (
              <div className="images-empty placeholder">{t("No image: Show-o draws a new one from the text. Open one to retouch a region.")}</div>
            )}
            <figcaption>
              {s.picture
                ? usable
                  ? `${t("Marked region")}: ${s.region![2] - s.region![0]} × ${s.region![3] - s.region![1]} px. ${t("Drag over the image to change it.")}`
                  : t("Drag over the image to mark the region that changes.")
                : " "}
            </figcaption>
          </figure>
          <figure className="picture">
            {s.result ? <img src={s.result.data} width={s.result.side} height={s.result.side} alt={`${t("Result")}: ${s.prompt}`} /> : <div className="images-empty placeholder">{s.running ? t(s.phase) : t("The result will appear here.")}</div>}
            <figcaption>{s.result ? `${s.result.side} × ${s.result.side} px, ${formatNumber(s.result.seconds)} s, ${t("seed")} ${s.result.seed}${how}` : " "}</figcaption>
          </figure>
        </div>
        {s.running && (
          <div className="images-progress">
            <progress max={Math.max(1, s.total)} value={s.step} />
            <span className="muted-text">
              {t(s.phase)}
              {s.total > 0 && s.step < s.total ? `: ${t("step")} ${s.step + 1} / ${s.total}` : ""}
            </span>
          </div>
        )}
        {s.error && (
          <div className="issue error" role="alert">
            <span className="issue-icon" aria-hidden="true">!</span>
            <div><strong>{t(s.error)}</strong></div>
          </div>
        )}
        <div className="composer">
          <label className={s.running ? "pill attach disabled" : "pill attach"} title={t("Open an image to retouch a region: it is cropped to the central square")}>
            {t("Image")}
            <input
              type="file"
              accept="image/*"
              disabled={s.running}
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) void readSquare(file).then((picture) => s.set({ picture, region: null }));
                e.target.value = "";
              }}
            />
          </label>
          {s.picture && !s.running && (
            <button className="pill" onClick={() => s.set({ picture: null, region: null })}>
              {t("Remove")}
            </button>
          )}
          {s.result && !s.running && (
            <button className="pill" title={t("Use the result as the starting image, to retouch another region")} onClick={() => void fetch(s.result!.data).then((r) => r.blob()).then((b) => readSquare(new File([b], "result.bmp"))).then((picture) => s.set({ picture, region: null }))}>
              {t("Keep retouching")}
            </button>
          )}
          <textarea
            aria-label={t("What it should show")}
            rows={2}
            value={s.prompt}
            placeholder={retouching ? t("What should appear in the region, in English: “a green apple”") : t("What to draw, in English: “a red apple on a wooden table”")}
            onChange={(e) => s.set({ prompt: e.target.value })}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && ready) void s.start();
            }}
          />
          {s.running ? (
            <button className="pill" onClick={() => void s.cancel()}>
              {t("Stop")}
            </button>
          ) : (
            <button className="pill primary-pill" disabled={!ready} onClick={() => void s.start()}>
              {t(retouching ? "Retouch" : "Draw")}
            </button>
          )}
        </div>
      </div>
    </section>
  );
}

export function ImagesPanel() {
  const t = useT();
  const s = useImages();
  const setView = useDesign((d) => d.setView);
  const { connect } = s;
  useEffect(() => {
    void connect();
  }, [connect]);
  const selected = s.models.find((m) => m.path === s.path);
  const minutes = formatNumber(Math.ceil((25 + s.steps * (s.guidance > 0 ? 2 : 1) * 42) / 60));
  const magvit = s.magvitOnChip ? t(s.picture ? "MAGVIT-v2 on the chip adds about 7 and a half minutes." : "MAGVIT-v2 on the chip adds about 3 minutes.") : t("MAGVIT-v2 stays on the host.");

  return (
    <aside className="inspector">
      <h2 className="detail-title">{t("Images")}</h2>
      <p className="detail-sub">{t("Draw an image from a text, or retouch a region of another, with Show-o.")}</p>

      {s.online === false ? (
        <div className="issue warning" role="status">
          <span className="issue-icon" aria-hidden="true">i</span>
          <div>
            <strong>{t("The engine is not running.")}</strong>
            <p>{t("Start it with this command in the project folder")}: scripts/dev.sh</p>
          </div>
        </div>
      ) : (
        <>
          <h3>{t("Model")}</h3>
          <div className="inset padded form">
            {s.models.length === 0 ? (
              <>
                <p className="muted-text">{t("There is no Show-o in the model folders. It needs Show-o's folder with MAGVIT-v2 in “magvitv2” and Phi-1.5's tokenizer in “phi”.")}</p>
                <button className="link-button" onClick={() => setView("models")}>
                  {t("See the models…")}
                </button>
              </>
            ) : (
              <>
                {s.models.length > 1 && (
                  <label>
                    {t("Folder")}
                    <select value={s.path} disabled={s.running} onChange={(e) => s.set({ path: e.target.value })}>
                      {s.models.map((m) => (
                        <option key={m.path} value={m.path}>
                          {m.name}
                        </option>
                      ))}
                    </select>
                  </label>
                )}
                {selected && (
                  <p className="muted-text">
                    {selected.name} · {formatNumber(selected.size_gb)} GB
                  </p>
                )}
                <label className="check" title={t("Show-o runs as on the chip: the die with its cells and 32-bit addresses, 8-bit affine weights in the cells, 16-bit vectors, and the firmware's loop, with MaskGIT's sampling in integers and the guidance in the die. MAGVIT-v2 stays in floating point on the host.")}>
                  <input type="checkbox" checked={s.onChip} disabled={s.running} onChange={(e) => s.set({ onChip: e.target.checked })} />
                  {t("As on the chip")}
                </label>
                {s.onChip && (
                  <label className="check" title={t("MAGVIT-v2 in the die's arithmetic too: it encodes the image (about 4 minutes) and decodes the result (about 3).")}>
                    <input type="checkbox" checked={s.magvitOnChip} disabled={s.running} onChange={(e) => s.set({ magvitOnChip: e.target.checked })} />
                    {t("Also MAGVIT-v2 on the chip")}
                  </label>
                )}
                <p className="muted-text">
                  {s.onChip
                    ? `${t("Each pass through the die takes about 42 s, against 6 s in floating point, and preparing it about 25 s. With guidance there are two per step")}: ${s.steps} ${t("steps")}, ~${minutes} min. ${magvit}`
                    : t("Runs in floating point on the host, outside the chip. The first time it takes a few seconds to read the weights.")}
                </p>
              </>
            )}
          </div>

          <h3>{t("Generation")}</h3>
          <div className="inset padded form">
            <label title={t("Each step fixes the codes the model is surest of; more steps give more detail and take longer.")}>
              {t("Steps")}
              <input type="text" inputMode="numeric" value={s.steps} disabled={s.running} onChange={(e) => s.set({ steps: Math.max(1, Math.min(64, Number(e.target.value) || 1)) })} />
            </label>
            <label title={t("How far it moves from the result without the text: higher follows the text more, at the cost of naturalness. With guidance, each step makes two passes.")}>
              {t("Guidance")}
              <input type="text" inputMode="decimal" value={s.guidance} disabled={s.running} onChange={(e) => s.set({ guidance: Math.max(0, Number(e.target.value.replace(",", ".")) || 0) })} />
            </label>
            <label title={t("The same seed, with the same text and region, gives the same result.")}>
              {t("Seed (empty for a new one each time)")}
              <input type="text" inputMode="numeric" value={s.seed} disabled={s.running} onChange={(e) => s.set({ seed: e.target.value.replace(/[^0-9]/g, "").slice(0, 15) })} />
            </label>
          </div>

          <h3>{t("How it works")}</h3>
          <div className="inset padded">
            <p className="muted-text">
              {t("MAGVIT-v2 turns the image into a 16 × 16 grid of codes. The codes of the marked region are masked, and the transformer fills them in over several steps, guided by the text and by the rest of the image. At the end, MAGVIT-v2 turns the codes back into pixels.")}
            </p>
          </div>
        </>
      )}
    </aside>
  );
}
