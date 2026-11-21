import type { Item } from "../App";

const pct = (n: number) => `${(n * 100).toFixed(1)}%`;
const kb = (n: number) => (n > 1024 * 1024 ? `${(n / 1024 / 1024).toFixed(1)} MB` : `${Math.round(n / 1024)} KB`);

export default function Verdict({ item, modelLoading }: { item: Item; modelLoading: boolean }) {
  const r = item.result;
  const tone = r ? (r.label === "fake" ? "fake" : "real") : item.status === "error" ? "error" : "pending";
  const lowConfidence = r && r.confidence < 0.65;

  return (
    <article className={`verdict tone-${tone}`}>
      <figure className={`specimen ${item.status === "pending" ? "is-scanning" : ""}`}>
        <img src={item.url} alt={item.name} />
        {item.status === "pending" && <span className="scanline" aria-hidden />}
        <figcaption>
          <span className="mono">{item.name}</span>
          <span className="mono muted">
            {r ? `${r.image.width}×${r.image.height} · ` : ""}
            {kb(item.size)}
          </span>
        </figcaption>
      </figure>

      <div className="readout" aria-live="polite">
        {item.status === "pending" && (
          <>
            <p className="eyebrow">Analysing</p>
            <h2 className="verdict-label">{modelLoading ? "Loading the model…" : "Running the model…"}</h2>
            {modelLoading && <p className="note">First visit only: the 13 MB model downloads once, then runs on this device.</p>}
          </>
        )}

        {item.status === "error" && (
          <>
            <p className="eyebrow">Couldn’t analyse</p>
            <h2 className="verdict-label">Error</h2>
            <p className="note">{item.error}</p>
          </>
        )}

        {r && (
          <>
            <p className="eyebrow">
              Verdict
            </p>
            <h2 className="verdict-label">{r.label === "fake" ? "Likely fake" : "Likely real"}</h2>
            <p className="confidence">
              <span className="mono big">{pct(r.confidence)}</span> confidence
              {lowConfidence && <span className="tag tag-warn">low confidence</span>}
            </p>

            <div className="meter" role="img" aria-label={`Probability fake ${pct(r.fake_probability)}`}>
              <div className="meter-track">
                <div className="meter-fill" style={{ width: pct(r.fake_probability) }} />
                <div className="meter-threshold" title="Decision threshold (50%)" />
              </div>
              <div className="meter-scale mono">
                <span>Real</span>
                <span>50%</span>
                <span>Fake</span>
              </div>
            </div>

            <dl className="stats">
              <div>
                <dt>P(fake)</dt>
                <dd className="mono">{pct(r.fake_probability)}</dd>
              </div>
              <div>
                <dt>P(real)</dt>
                <dd className="mono">{pct(r.real_probability)}</dd>
              </div>
              <div>
                <dt>Inference</dt>
                <dd className="mono">{r.inference_ms.toFixed(0)} ms</dd>
              </div>
            </dl>

            <p className="note">
              {lowConfidence
                ? "The score sits close to the 50% threshold, so treat this one as inconclusive."
                : "A classifier score, not proof. The model sees a 128×128 copy and works best on face crops like its training data."}
            </p>
          </>
        )}
      </div>
    </article>
  );
}
