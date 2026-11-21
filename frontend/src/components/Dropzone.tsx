import { useRef, useState } from "react";

type Props = { onFiles: (files: File[]) => void; large?: boolean };

export default function Dropzone({ onFiles, large }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  return (
    <div
      className={`dropzone ${large ? "dropzone-large" : ""} ${over ? "is-over" : ""}`}
      role="button"
      tabIndex={0}
      onClick={() => input.current?.click()}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        onFiles(Array.from(e.dataTransfer.files));
      }}
    >
      <input
        ref={input}
        type="file"
        accept="image/jpeg,image/png,image/webp,image/bmp"
        multiple
        hidden
        onChange={(e) => {
          onFiles(Array.from(e.target.files ?? []));
          e.target.value = "";
        }}
      />
      {large ? (
        <>
          <svg className="drop-icon" viewBox="0 0 48 48" aria-hidden>
            <path d="M8 16V10a2 2 0 0 1 2-2h6M32 8h6a2 2 0 0 1 2 2v6M40 32v6a2 2 0 0 1-2 2h-6M16 40h-6a2 2 0 0 1-2-2v-6" />
            <circle cx="24" cy="24" r="7" />
          </svg>
          <h2>Drop an image to analyse</h2>
          <p>
            or <span className="link">browse files</span> · paste with <kbd>Ctrl</kbd>+<kbd>V</kbd>
          </p>
          <p className="hint">JPG, PNG, WebP or BMP · several at once is fine · up to 15 MB each</p>
        </>
      ) : (
        <p>
          <span className="link">Add more images</span> · drop or paste anywhere
        </p>
      )}
    </div>
  );
}
