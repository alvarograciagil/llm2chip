import { useState, type ReactNode } from "react";

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
