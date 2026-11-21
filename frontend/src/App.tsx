import { useCallback, useEffect, useState } from "react";
import { getHealth, predict, type Health, type Prediction } from "./api";
import Dropzone from "./components/Dropzone";
import Verdict from "./components/Verdict";
import History from "./components/History";
import ModelInfo from "./components/ModelInfo";

export type Item = {
  id: string;
  name: string;
  size: number;
  url: string;
  status: "pending" | "done" | "error";
  result?: Prediction;
  error?: string;
};

const ACCEPTED = ["image/jpeg", "image/png", "image/webp", "image/bmp"];

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [items, setItems] = useState<Item[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  useEffect(() => {
    getHealth().then(setHealth).catch((e: Error) => setHealthError(e.message));
  }, []);

  const update = (id: string, patch: Partial<Item>) =>
    setItems((prev) => prev.map((it) => (it.id === id ? { ...it, ...patch } : it)));

  const analyse = useCallback(async (files: File[]) => {
    const images = files.filter((f) => ACCEPTED.includes(f.type));
    if (!images.length) return;
    const fresh: Item[] = images.map((f) => ({
      id: crypto.randomUUID(),
      name: f.name || "pasted-image.png",
      size: f.size,
      url: URL.createObjectURL(f),
      status: "pending",
    }));
    setItems((prev) => [...fresh, ...prev]);
    setSelectedId(fresh[0].id);
    // Sequential: the CPU model is quick, and this keeps the server load sane.
    for (let i = 0; i < images.length; i++) {
      try {
        const result = await predict(images[i]);
        update(fresh[i].id, { status: "done", result });
      } catch (e) {
        update(fresh[i].id, { status: "error", error: (e as Error).message });
      }
    }
  }, []);

  // Paste an image from the clipboard anywhere on the page.
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const files = Array.from(e.clipboardData?.files ?? []);
      if (files.length) analyse(files);
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [analyse]);

  const clear = () => {
    items.forEach((it) => URL.revokeObjectURL(it.url));
    setItems([]);
    setSelectedId(null);
  };

  const selected = items.find((it) => it.id === selectedId) ?? null;

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden />
          <div>
            <h1>Deepfake Detector</h1>
            <p>Image classifier · CNN trained on 140k real &amp; synthetic faces</p>
          </div>
        </div>
        <StatusPill health={health} error={healthError} />
      </header>

      {health?.mode === "demo" && (
        <div className="banner" role="status">
          <strong>Demo mode.</strong> No trained model is loaded, so results are placeholder
          scores, not real predictions. Put <code>{health.model_file}</code> in the{" "}
          <code>models/</code> folder and restart the server.
        </div>
      )}
      {healthError && (
        <div className="banner banner-error" role="alert">
          <strong>Can’t reach the API.</strong> Start the backend on port 8000 ({healthError}).
        </div>
      )}

      <main className="layout">
        <section className="workspace">
          {selected ? (
            <Verdict item={selected} />
          ) : (
            <Dropzone onFiles={analyse} large />
          )}
          {selected && <Dropzone onFiles={analyse} />}
        </section>

        <aside className="sidebar">
          <History items={items} selectedId={selectedId} onSelect={setSelectedId} onClear={clear} />
          <ModelInfo health={health} />
        </aside>
      </main>
    </div>
  );
}

function StatusPill({ health, error }: { health: Health | null; error: string | null }) {
  if (error) return <span className="pill pill-error">API offline</span>;
  if (!health) return <span className="pill">Connecting…</span>;
  return health.mode === "model" ? (
    <span className="pill pill-ok" title={health.model_file}>Model loaded</span>
  ) : (
    <span className="pill pill-warn" title={health.detail ?? ""}>Demo mode</span>
  );
}
