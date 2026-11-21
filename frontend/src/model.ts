// In-browser inference for the deepfake CNN.
//
// Weights come from tools/export_weights.py. The layers are rebuilt with
// TensorFlow.js ops, and images are prepared the way the notebook trained the
// model (tf.keras image_dataset_from_directory): RGB, bilinear resize to
// 128x128 with half-pixel centres (TF2's tf.image.resize), values left 0-255.
// The sigmoid output is P(Real), because Keras sorts the class folders
// alphabetically: Fake = 0, Real = 1.

import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";
import "@tensorflow/tfjs-backend-cpu";

const MODEL_URL = `${import.meta.env.BASE_URL}model/`;

type WeightRef = { shape: number[]; offset: number };
type LayerSpec = {
  name: string;
  type: "Conv2D" | "MaxPooling2D" | "Flatten" | "Dropout" | "Dense";
  activation?: string;
  pool?: [number, number];
  strides?: [number, number];
  weights?: WeightRef[];
};
export type ModelMeta = {
  source: string;
  input: { size: [number, number] };
  test_accuracy: number;
  weights_bytes: number;
  layers: LayerSpec[];
};

export type ModelState =
  | { status: "loading"; progress: number }
  | { status: "ready"; backend: string; meta: ModelMeta }
  | { status: "error"; message: string };

export type Prediction = {
  label: "fake" | "real";
  fake_probability: number;
  real_probability: number;
  confidence: number;
  inference_ms: number;
  image: { width: number; height: number; format: string | null };
};

type Loaded = { meta: ModelMeta; layers: { spec: LayerSpec; w: tf.Tensor[] }[] };

let loading: Promise<Loaded> | null = null;
const listeners = new Set<(s: ModelState) => void>();
let state: ModelState = { status: "loading", progress: 0 };

function emit(next: ModelState) {
  state = next;
  listeners.forEach((fn) => fn(state));
}

export function subscribe(fn: (s: ModelState) => void): () => void {
  listeners.add(fn);
  fn(state);
  return () => listeners.delete(fn);
}

async function fetchWithProgress(url: string, total: number): Promise<ArrayBuffer> {
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`Couldn't download ${url} (${res.status})`);
  const reader = res.body.getReader();
  const buf = new Uint8Array(total);
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (received + value.length > total) throw new Error("Weights file is larger than expected");
    buf.set(value, received);
    received += value.length;
    emit({ status: "loading", progress: received / total });
  }
  if (received !== total) throw new Error(`Weights download incomplete (${received} of ${total} bytes)`);
  return buf.buffer;
}

async function pickBackend(): Promise<string> {
  for (const name of ["webgl", "cpu"]) {
    try {
      if (await tf.setBackend(name)) {
        await tf.ready();
        return name;
      }
    } catch {
      /* try the next backend */
    }
  }
  throw new Error("No TensorFlow.js backend is available in this browser");
}

export function loadModel(): Promise<Loaded> {
  loading ??= (async () => {
    try {
      const backend = await pickBackend();
      const meta: ModelMeta = await (await fetch(`${MODEL_URL}model.json`)).json();
      const data = new Float32Array(await fetchWithProgress(`${MODEL_URL}weights.bin`, meta.weights_bytes));
      const layers = meta.layers.map((spec) => ({
        spec,
        w: (spec.weights ?? []).map((ref) => {
          const size = ref.shape.reduce((a, b) => a * b, 1);
          return tf.tensor(data.subarray(ref.offset, ref.offset + size), ref.shape, "float32");
        }),
      }));
      const loaded = { meta, layers };
      // Warm-up run so the first real prediction isn't slowed by shader compilation.
      tf.tidy(() => forward(loaded, tf.zeros([1, ...meta.input.size, 3])));
      emit({ status: "ready", backend, meta });
      return loaded;
    } catch (e) {
      loading = null;
      emit({ status: "error", message: (e as Error).message });
      throw e;
    }
  })();
  return loading;
}

function activate(x: tf.Tensor, fn?: string): tf.Tensor {
  if (fn === "relu") return tf.relu(x);
  if (fn === "sigmoid") return tf.sigmoid(x);
  if (!fn || fn === "linear") return x;
  throw new Error(`Unsupported activation ${fn}`);
}

function forward({ layers }: Loaded, input: tf.Tensor): tf.Tensor {
  let x = input;
  for (const { spec, w } of layers) {
    switch (spec.type) {
      case "Conv2D":
        x = activate(tf.add(tf.conv2d(x as tf.Tensor4D, w[0] as tf.Tensor4D, 1, "valid"), w[1]), spec.activation);
        break;
      case "MaxPooling2D":
        x = tf.maxPool(x as tf.Tensor4D, spec.pool!, spec.strides!, "valid");
        break;
      case "Flatten":
        x = tf.reshape(x, [x.shape[0], -1]); // NHWC order, same as Keras channels_last
        break;
      case "Dropout":
        break; // inactive at inference
      case "Dense":
        x = activate(tf.add(tf.matMul(x as tf.Tensor2D, w[0] as tf.Tensor2D), w[1]), spec.activation);
        break;
    }
  }
  return x;
}

export async function predict(file: File): Promise<Prediction> {
  const model = await loadModel();
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(file);
  } catch {
    throw new Error("Could not read that file as an image");
  }
  const { width, height } = bitmap;
  const started = performance.now();
  const output = tf.tidy(() => {
    const pixels = tf.browser.fromPixels(bitmap, 3); // int32 RGB, 0-255
    const resized = tf.image.resizeBilinear(pixels, model.meta.input.size, false, true);
    return forward(model, tf.expandDims(tf.cast(resized, "float32"), 0));
  });
  const pReal = (await output.data())[0];
  output.dispose();
  bitmap.close();
  const inference_ms = performance.now() - started;

  const pFake = 1 - pReal;
  return {
    label: pFake > 0.5 ? "fake" : "real",
    fake_probability: pFake,
    real_probability: pReal,
    confidence: Math.max(pFake, pReal),
    inference_ms,
    image: { width, height, format: file.type.replace("image/", "").toUpperCase() || null },
  };
}
