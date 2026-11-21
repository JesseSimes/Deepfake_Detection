import type { ModelState } from "../model";

export default function ModelInfo({ model }: { model: ModelState }) {
  const meta = model.status === "ready" ? model.meta : null;
  const size = meta?.input.size[0] ?? 128;
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
          <dd className="mono">
            {size}×{size} RGB
          </dd>
        </div>
        <div>
          <dt>Training data</dt>
          <dd>140,002 images, balanced real / fake (Kaggle “Deepfake and Real Images”)</dd>
        </div>
        <div>
          <dt>Test accuracy</dt>
          <dd className="mono">{meta ? `${(meta.test_accuracy * 100).toFixed(2)}%` : "—"}</dd>
        </div>
        <div>
          <dt>Runs on</dt>
          <dd className="mono small">
            {model.status === "ready" ? `TensorFlow.js (${model.backend})` : "—"}
          </dd>
        </div>
      </dl>
      <p className="muted small">
        Roughly 1 in 8 test images was misclassified. Use the score as one signal among several.
      </p>
    </section>
  );
}
