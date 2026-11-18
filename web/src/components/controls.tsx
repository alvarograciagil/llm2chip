import { useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { useT } from "../i18n";

export function Switch({ checked, label, onChange }: { checked: boolean; label: string; onChange: (v: boolean) => void }) {
  return (
    <label className="switch">
      <input type="checkbox" role="switch" aria-label={label} checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span aria-hidden="true" />
    </label>
  );
}

export function Segmented<T extends string>({
  value,
  options,
  label,
  onChange,
}: {
  value: T;
  options: [T, string][];
  label: string;
  onChange: (v: T) => void;
}) {
  return (
    <div className="segmented" role="radiogroup" aria-label={label}>
      {options.map(([id, text]) => (
        <button key={id} role="radio" aria-checked={value === id} className={value === id ? "on" : ""} onClick={() => onChange(id)}>
          {text}
        </button>
      ))}
    </div>
  );
}

export function Disclosure({
  title,
  summary,
  defaultOpen,
  forceOpen,
  children,
}: {
  title: string;
  summary?: string;
  defaultOpen: boolean;
  forceOpen?: boolean;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const shown = open || forceOpen === true;
  return (
    <section className="disclosure">
      <button className="disclosure-head" aria-expanded={shown} onClick={() => setOpen(!open)}>
        <svg className={shown ? "chevron open" : "chevron"} viewBox="0 0 8 12" aria-hidden="true">
          <path d="M1.5 1 6.5 6l-5 5" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <span>{title}</span>
        {!shown && summary && <small>{summary}</small>}
      </button>
      {shown && <div className="inset">{children}</div>}
    </section>
  );
}

export function IconButton({
  label,
  disabled,
  onClick,
  children,
}: {
  label: string;
  disabled?: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button className="icon-button" aria-label={label} title={label} disabled={disabled} onClick={onClick}>
      <svg viewBox="0 0 20 20" aria-hidden="true">
        {children}
      </svg>
    </button>
  );
}

/** A "?" that explains something while the pointer or the focus is on it, above everything
 * so no scrolling panel cuts it, and opens its documentation when pressed, if it has any. */
export function InfoTip({ id, label, text, onOpen }: { id: string; label: string; text: string; onOpen?: () => void }) {
  const t = useT();
  const [at, setAt] = useState<{ x: number; y: number; above: boolean } | null>(null);
  const show = (e: { currentTarget: HTMLElement }) => {
    const box = e.currentTarget.getBoundingClientRect();
    const above = box.bottom + 160 > window.innerHeight;
    setAt({ x: Math.max(8, Math.min(box.left - 8, window.innerWidth - 276)), y: above ? box.top - 6 : box.bottom + 6, above });
  };
  return (
    <span className="info">
      <button className="info-button" aria-label={t(`What ${label} is`)} aria-describedby={id} onClick={onOpen} onMouseEnter={show} onMouseLeave={() => setAt(null)} onFocus={show} onBlur={() => setAt(null)}>
        ?
      </button>
      {/* The sidebar's blur would make a fixed position its own: the tip lives in the page. */}
      {createPortal(
        <span className={at ? "tooltip shown" : "tooltip"} role="tooltip" id={id} style={at ? { left: at.x, top: at.y, transform: at.above ? "translateY(-100%)" : undefined } : undefined}>
          {text}
          {onOpen && <em>{t("Click to see the full documentation.")}</em>}
        </span>,
        document.body,
      )}
    </span>
  );
}
