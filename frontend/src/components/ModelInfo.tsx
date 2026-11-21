import type { Health } from "../api";

export default function ModelInfo({ health }: { health: Health | null }) {
  return (
    <section className="panel">
      <div className="panel-head">
        <h3>About the model</h3>
      </div>
      <dl className="facts">
        <div>
          <dt>Architecture</dt>
          <dd>3× Conv2D/MaxPool → Dense 128 → sigmoid</dd>
        </div>
        <div>
          <dt>Input</dt>
          <dd className="mono">{health?.input_size ?? 128}×{health?.input_size ?? 128} RGB</dd>
        </div>
        <div>
          <dt>Training data</dt>
          <dd>140,002 images, balanced real / fake (Kaggle “Deepfake and Real Images”)</dd>
        </div>
        <div>
          <dt>Test accuracy</dt>
          <dd className="mono">87.27%</dd>
        </div>
        <div>
          <dt>Model file</dt>
          <dd className="mono small">{health?.model_file ?? "—"}</dd>
        </div>
      </dl>
      <p className="muted small">
        Roughly 1 in 8 test images was misclassified. Use the score as one signal among several.
      </p>
    </section>
  );
}
