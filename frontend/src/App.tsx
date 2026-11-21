import { useCallback, useEffect, useState } from "react";
import { loadModel, predict, subscribe, type ModelState, type Prediction } from "./model";
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
  const [model, setModel] = useState<ModelState>({ status: "loading", progress: 0 });
  const [items, setItems] = useState<Item[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  useEffect(() => {
    const unsubscribe = subscribe(setModel);
    loadModel().catch(() => {
      /* surfaced through the subscription */
    });
    return unsubscribe;
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
            <p>Image classifier · runs in your browser · images never leave your device</p>
          </div>
        </div>
        <StatusPill model={model} />
      </header>

      {model.status === "error" && (
        <div className="banner banner-error" role="alert">
          <strong>The model couldn’t load.</strong> {model.message}.{" "}
          <button className="inline-link" onClick={() => loadModel().catch(() => {})}>
            Try again
          </button>
        </div>
      )}

      <main className="layout">
        <section className="workspace">
          {selected ? <Verdict item={selected} modelLoading={model.status === "loading"} /> : <Dropzone onFiles={analyse} large />}
          {selected && <Dropzone onFiles={analyse} />}
        </section>

        <aside className="sidebar">
          <History items={items} selectedId={selectedId} onSelect={setSelectedId} onClear={clear} />
          <ModelInfo model={model} />
        </aside>
      </main>
    </div>
  );
}

function StatusPill({ model }: { model: ModelState }) {
  if (model.status === "error") return <span className="pill pill-error">Model failed</span>;
  if (model.status === "loading")
    return <span className="pill">Loading model {Math.round(model.progress * 100)}%</span>;
  return (
    <span className="pill pill-ok" title={`TensorFlow.js · ${model.backend}`}>
      Model ready
    </span>
  );
}
