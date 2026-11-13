import { useEffect, useRef, useState } from "react";
import { formatNumber } from "../format";
import { useDesign } from "../store";
import { readSquare, regionBetween, regionIsUsable, useImages, type Region } from "./imagesStore";

/** The picture with the region marked on it: drag across it to mark another. */
function Marker({ preview, side, region, disabled, onRegion }: { preview: string; side: number; region: Region | null; disabled: boolean; onRegion: (r: Region | null) => void }) {
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
      <img src={preview} alt="Imagen que se retoca" draggable={false} />
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
  const s = useImages();
  const usable = s.picture !== null && regionIsUsable(s.region, s.picture.side);
  const retouching = s.picture !== null;
  const ready = s.path !== "" && s.prompt.trim() !== "" && (!retouching || usable) && !s.running;
  return (
    <section className="stage sim-stage">
      <div className="images-column">
        <div className="images-pair">
          <figure className="picture">
            {s.picture ? (
              <Marker preview={s.picture.preview} side={s.picture.side} region={s.region} disabled={s.running} onRegion={(region) => s.set({ region })} />
            ) : (
              <div className="images-empty placeholder">Sin imagen: Show-o dibuja una nueva a partir del texto. Abre una para retocar una zona.</div>
            )}
            <figcaption>
              {s.picture
                ? usable
                  ? `Zona marcada: ${s.region![2] - s.region![0]} × ${s.region![3] - s.region![1]} px. Arrastra sobre la imagen para cambiarla.`
                  : "Arrastra sobre la imagen para marcar la zona que se cambia."
                : " "}
            </figcaption>
          </figure>
          <figure className="picture">
            {s.result ? <img src={s.result.data} width={s.result.side} height={s.result.side} alt={`Resultado: ${s.prompt}`} /> : <div className="images-empty placeholder">{s.running ? s.phase : "El resultado aparecerá aquí."}</div>}
            <figcaption>{s.result ? `${s.result.side} × ${s.result.side} px, en ${formatNumber(s.result.seconds)} s, semilla ${s.result.seed}` : " "}</figcaption>
          </figure>
        </div>
        {s.running && (
          <div className="images-progress">
            <progress max={Math.max(1, s.total)} value={s.step} />
            <span className="muted-text">
              {s.phase}
              {s.total > 0 && s.step < s.total ? `: paso ${s.step + 1} de ${s.total}` : ""}
            </span>
          </div>
        )}
        {s.error && (
          <div className="issue error" role="alert">
            <span className="issue-icon" aria-hidden="true">!</span>
            <div><strong>{s.error}</strong></div>
          </div>
        )}
        <div className="composer">
          <label className={s.running ? "pill attach disabled" : "pill attach"} title="Abrir una imagen para retocar una zona: se recorta al cuadrado central">
            Imagen
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
              Quitar
            </button>
          )}
          {s.result && !s.running && (
            <button className="pill" title="Usar el resultado como imagen de partida, para retocar otra zona" onClick={() => void fetch(s.result!.data).then((r) => r.blob()).then((b) => readSquare(new File([b], "resultado.bmp"))).then((picture) => s.set({ picture, region: null }))}>
              Seguir retocando
            </button>
          )}
          <textarea
            aria-label="Qué debe mostrar"
            rows={2}
            value={s.prompt}
            placeholder={retouching ? "Qué debe aparecer en la zona, en inglés: «a green apple»" : "Qué dibujar, en inglés: «a red apple on a wooden table»"}
            onChange={(e) => s.set({ prompt: e.target.value })}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && ready) void s.start();
            }}
          />
          {s.running ? (
            <button className="pill" onClick={() => void s.cancel()}>
              Detener
            </button>
          ) : (
            <button className="pill primary-pill" disabled={!ready} onClick={() => void s.start()}>
              {retouching ? "Retocar" : "Dibujar"}
            </button>
          )}
        </div>
      </div>
    </section>
  );
}

export function ImagesPanel() {
  const s = useImages();
  const setView = useDesign((d) => d.setView);
  const { connect } = s;
  useEffect(() => {
    void connect();
  }, [connect]);
  const selected = s.models.find((m) => m.path === s.path);

  return (
    <aside className="inspector">
      <h2 className="detail-title">Imágenes</h2>
      <p className="detail-sub">Dibuja una imagen desde un texto o retoca una zona de otra con Show-o.</p>

      {s.online === false ? (
        <div className="issue aviso" role="status">
          <span className="issue-icon" aria-hidden="true">i</span>
          <div>
            <strong>El motor no está en marcha.</strong>
            <p>Ábrelo con este comando en la carpeta del proyecto: scripts/dev.sh</p>
          </div>
        </div>
      ) : (
        <>
          <h3>Modelo</h3>
          <div className="inset padded form">
            {s.models.length === 0 ? (
              <>
                <p className="muted-text">No hay ningún Show-o en las carpetas de modelos. Hace falta la carpeta de Show-o con MAGVIT-v2 en «magvitv2» y el tokenizador de Phi-1.5 en «phi».</p>
                <button className="link-button" onClick={() => setView("modelos")}>
                  Ver los modelos…
                </button>
              </>
            ) : (
              <>
                {s.models.length > 1 && (
                  <label>
                    Carpeta
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
                <p className="muted-text">
                  Corre en coma flotante en el anfitrión, fuera del chip: el chip aún no lleva el transformador de Show-o (Phi-1.5) ni MAGVIT-v2. La primera vez tarda unos segundos en leer los pesos.
                </p>
              </>
            )}
          </div>

          <h3>Generación</h3>
          <div className="inset padded form">
            <label title="Cada paso fija los códigos de los que el modelo está más seguro; más pasos dan más detalle y tardan más.">
              Pasos
              <input type="text" inputMode="numeric" value={s.steps} disabled={s.running} onChange={(e) => s.set({ steps: Math.max(1, Math.min(64, Number(e.target.value) || 1)) })} />
            </label>
            <label title="Cuánto se aleja del resultado sin texto: más alto sigue más el texto, a costa de naturalidad. Con guía, cada paso hace dos pasadas.">
              Guía
              <input type="text" inputMode="decimal" value={s.guidance} disabled={s.running} onChange={(e) => s.set({ guidance: Math.max(0, Number(e.target.value.replace(",", ".")) || 0) })} />
            </label>
            <label title="La misma semilla, con el mismo texto y la misma zona, da el mismo resultado.">
              Semilla (vacía, una nueva cada vez)
              <input type="text" inputMode="numeric" value={s.seed} disabled={s.running} onChange={(e) => s.set({ seed: e.target.value.replace(/[^0-9]/g, "").slice(0, 15) })} />
            </label>
          </div>

          <h3>Cómo funciona</h3>
          <div className="inset padded">
            <p className="muted-text">
              MAGVIT-v2 convierte la imagen en una retícula de 16 × 16 códigos. Los códigos de la zona marcada se tapan, y el transformador los rellena en varios pasos, guiado por el texto y por el resto de la imagen. Al final, MAGVIT-v2 vuelve a convertir los códigos en píxeles.
            </p>
          </div>
        </>
      )}
    </aside>
  );
}
