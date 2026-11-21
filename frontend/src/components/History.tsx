import type { Item } from "../App";

type Props = {
  items: Item[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onClear: () => void;
};

export default function History({ items, selectedId, onSelect, onClear }: Props) {
  const done = items.filter((i) => i.result);
  const fakes = done.filter((i) => i.result!.label === "fake").length;

  return (
    <section className="panel">
      <div className="panel-head">
        <h3>This session</h3>
        {items.length > 0 && (
          <button className="ghost" onClick={onClear}>
            Clear
          </button>
        )}
      </div>
      {items.length === 0 ? (
        <p className="muted small">Images you analyse will be listed here. Nothing is stored on the server.</p>
      ) : (
        <>
          <p className="muted small mono">
            {done.length} analysed · {fakes} flagged fake
          </p>
          <ul className="history">
            {items.map((it) => (
              <li key={it.id}>
                <button
                  className={`history-row ${it.id === selectedId ? "is-active" : ""}`}
                  onClick={() => onSelect(it.id)}
                >
                  <img src={it.url} alt="" />
                  <span className="history-name">{it.name}</span>
                  <Chip item={it} />
                </button>
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  );
}

function Chip({ item }: { item: Item }) {
  if (item.status === "pending") return <span className="chip">…</span>;
  if (item.status === "error") return <span className="chip chip-error">error</span>;
  const r = item.result!;
  return (
    <span className={`chip chip-${r.label}`}>
      {r.label} {Math.round(r.confidence * 100)}%
    </span>
  );
}
