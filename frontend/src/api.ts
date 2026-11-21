export type Health = {
  status: string;
  mode: "model" | "demo";
  model_file: string;
  detail: string | null;
  input_size: number;
  input_scale: string;
  positive_class: string;
};

export type Prediction = {
  label: "fake" | "real";
  fake_probability: number;
  real_probability: number;
  confidence: number;
  inference_ms: number;
  mode: "model" | "demo";
  image: { width: number; height: number; format: string | null };
};

export async function getHealth(): Promise<Health> {
  const res = await fetch("/api/health");
  if (!res.ok) throw new Error(`Health check failed (${res.status})`);
  return res.json();
}

export async function predict(file: File): Promise<Prediction> {
  const body = new FormData();
  body.append("file", file);
  const res = await fetch("/api/predict", { method: "POST", body });
  if (!res.ok) {
    let msg = `Request failed (${res.status})`;
    try {
      const data = await res.json();
      if (data?.detail) msg = typeof data.detail === "string" ? data.detail : msg;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(msg);
  }
  return res.json();
}
