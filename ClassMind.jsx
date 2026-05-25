import { useState, useRef } from "react";

// ── APIs ───────────────────────────────────────────────
const CLAUDE_API = "https://api.anthropic.com/v1/messages";
const WHISPER_API = "https://api.openai.com/v1/audio/transcriptions";
const MODEL = "claude-sonnet-4-20250514";
const BLOCK_MINUTES = 20;

// ── Whisper ────────────────────────────────────────────
async function transcribeAudio(blob, filename, key) {
  const form = new FormData();
  form.append("file", blob, filename);
  form.append("model", "whisper-1");
  form.append("language", "es");
  form.append("response_format", "verbose_json");
  form.append("timestamp_granularities[]", "segment");
  const res = await fetch(WHISPER_API, {
    method: "POST",
    headers: { Authorization: `Bearer ${key}` },
    body: form,
  });
  if (!res.ok) {
    const e = await res.json().catch(() => ({}));
    throw new Error(e?.error?.message || `Error Whisper: ${res.status}`);
  }
  return res.json();
}

// ── Agrupar segmentos de Whisper en bloques de 20 min ──
function groupSegments(segments) {
  const blockSecs = BLOCK_MINUTES * 60;
  const blocks = [];
  let current = { limit: blockSecs, texts: [] };
  for (const seg of segments) {
    if (seg.start >= current.limit && current.texts.length > 0) {
      blocks.push(current.texts.join(" "));
      const idx = Math.floor(seg.start / blockSecs);
      current = { limit: (idx + 1) * blockSecs, texts: [] };
    }
    current.texts.push(seg.text.trim());
  }
  if (current.texts.length > 0) blocks.push(current.texts.join(" "));
  return blocks;
}

// ── Audio: decode + downsample + split (archivos > 24 MB) ──
async function decodeAndChunk(file) {
  const TARGET_SR = 8000;
  const CHUNK_SECS = BLOCK_MINUTES * 60;
  const ab = await file.arrayBuffer();
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const buf = await ctx.decodeAudioData(ab);
  ctx.close();

  const { sampleRate: src, length: srcLen, numberOfChannels: ch } = buf;
  const ratio = src / TARGET_SR;
  const dstLen = Math.floor(srcLen / ratio);
  const mono = new Float32Array(dstLen);

  for (let i = 0; i < dstLen; i++) {
    const si = i * ratio, s0 = Math.floor(si), s1 = Math.min(s0 + 1, srcLen - 1), f = si - s0;
    for (let c = 0; c < ch; c++) {
      const d = buf.getChannelData(c);
      mono[i] += ((1 - f) * d[s0] + f * d[s1]) / ch;
    }
  }

  const chunkSamples = CHUNK_SECS * TARGET_SR;
  const chunks = [];
  for (let i = 0; i * chunkSamples < dstLen; i++) {
    const start = i * chunkSamples;
    chunks.push(encodeWAV(mono.slice(start, Math.min(start + chunkSamples, dstLen)), TARGET_SR));
  }
  return chunks;
}

function encodeWAV(samples, sr) {
  const n = samples.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const v = new DataView(buf);
  const ws = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  ws(0, "RIFF"); v.setUint32(4, 36 + n * 2, true); ws(8, "WAVE"); ws(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  ws(36, "data"); v.setUint32(40, n * 2, true);
  let o = 44;
  for (let i = 0; i < n; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    o += 2;
  }
  return new Blob([buf], { type: "audio/wav" });
}

// ── Claude: analizar bloque ────────────────────────────
async function analyzeBlock(transcript, num) {
  const res = await fetch(CLAUDE_API, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: MODEL, max_tokens: 1000,
      system: "Sos un asistente académico experto. Analizás transcripciones de clases y generás material de estudio detallado en español rioplatense. Respondé ÚNICAMENTE con JSON válido, sin backticks ni texto extra.",
      messages: [{ role: "user", content: `Analizá la transcripción del Bloque ${num} de una clase universitaria y generá material de estudio.

TRANSCRIPCIÓN:
${transcript?.trim() || "(sin contenido detectado en este bloque)"}

Respondé con este JSON:
{
  "titulo": "título descriptivo del tema principal",
  "resumen": "resumen de 3-4 oraciones",
  "temas_principales": ["tema 1", "tema 2"],
  "conceptos_clave": [{"concepto": "nombre", "explicacion": "explicación detallada de 2-3 oraciones"}],
  "ejemplos": ["ejemplo o caso mencionado"],
  "ideas_importantes": ["punto o idea clave"]
}` }]
    })
  });
  const data = await res.json();
  const txt = data.content?.map((b) => b.text || "").join("") || "{}";
  try { return JSON.parse(txt.replace(/```json|```/g, "").trim()); }
  catch { return { titulo: `Bloque ${num}`, resumen: "No se pudo analizar.", temas_principales: [], conceptos_clave: [], ejemplos: [], ideas_importantes: [] }; }
}

async function generateFinalDoc(blocks) {
  const content = blocks.map((b) => {
    const a = b.analysis;
    return `=== BLOQUE ${b.number}: ${a.titulo} ===
Resumen: ${a.resumen}
Temas: ${a.temas_principales?.join(", ")}
Conceptos: ${a.conceptos_clave?.map((c) => `${c.concepto}: ${c.explicacion}`).join(" | ")}
Ideas: ${a.ideas_importantes?.join("; ")}`;
  }).join("\n\n");

  const res = await fetch(CLAUDE_API, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: MODEL, max_tokens: 1000,
      system: "Sos un asistente académico experto. Generás documentos de estudio completos, claros y bien estructurados en español rioplatense.",
      messages: [{ role: "user", content: `Generá un documento de estudio completo y detallado basado en los análisis de bloques de esta clase universitaria.

${content}

Estructurá el documento con estas secciones (usá ## para los títulos):

## RESUMEN EJECUTIVO
## TEMAS DESARROLLADOS
## CONCEPTOS CLAVE
## EJEMPLOS Y CASOS
## PUNTOS IMPORTANTES
## CONCLUSIONES

Sé exhaustivo, detallado y pedagógico.` }]
    })
  });
  const data = await res.json();
  return data.content?.map((b) => b.text || "").join("") || "";
}

// ── Utils ──────────────────────────────────────────────
const fmtSize = (b) => b < 1048576 ? `${(b / 1024).toFixed(0)} KB` : `${(b / 1048576).toFixed(1)} MB`;

// ── Component ──────────────────────────────────────────
export default function ClassMind() {
  const [key, setKey] = useState("");
  const [file, setFile] = useState(null);
  const [phase, setPhase] = useState("setup"); // setup | processing | done
  const [steps, setSteps] = useState([]);
  const [blocks, setBlocks] = useState([]);
  const [finalDoc, setFinalDoc] = useState("");
  const [expanded, setExpanded] = useState(null);
  const [error, setError] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef(null);

  const pushStep = (msg) => setSteps((p) => [...p.map((s) => s.active ? { ...s, active: false } : s), { msg, active: true }]);
  const updateLastStep = (msg) => setSteps((p) => p.map((s, i) => i === p.length - 1 ? { ...s, msg } : s));

  const processFile = async () => {
    if (!file || !key.trim()) return;
    setError(""); setPhase("processing"); setSteps([]); setBlocks([]); setFinalDoc("");
    const MAX_DIRECT = 24 * 1024 * 1024;

    try {
      let transcriptBlocks = [];

      if (file.size <= MAX_DIRECT) {
        pushStep(`Transcribiendo con Whisper (${fmtSize(file.size)})...`);
        const res = await transcribeAudio(file, file.name, key.trim());
        if (res.segments?.length > 0) {
          transcriptBlocks = groupSegments(res.segments);
          updateLastStep(`✓ Transcripción lista · ${transcriptBlocks.length} bloque${transcriptBlocks.length !== 1 ? "s" : ""} de ${BLOCK_MINUTES} min`);
        } else {
          transcriptBlocks = [res.text || ""];
          updateLastStep("✓ Transcripción lista · 1 bloque");
        }
      } else {
        pushStep(`Archivo ${fmtSize(file.size)} · Decodificando audio...`);
        let chunks;
        try { chunks = await decodeAndChunk(file); }
        catch { throw new Error("El archivo es muy grande para procesar en el navegador. Grabá con menor calidad (M4A, 32 kbps) para obtener un archivo bajo 24 MB."); }
        updateLastStep(`✓ Audio dividido en ${chunks.length} bloques`);

        for (let i = 0; i < chunks.length; i++) {
          pushStep(`Transcribiendo bloque ${i + 1}/${chunks.length}...`);
          const res = await transcribeAudio(chunks[i], `bloque_${i + 1}.wav`, key.trim());
          transcriptBlocks.push(res.text || res.segments?.map((s) => s.text).join(" ") || "");
          updateLastStep(`✓ Bloque ${i + 1}/${chunks.length} transcripto`);
        }
      }

      const total = transcriptBlocks.length;
      const analyzed = [];

      for (let i = 0; i < total; i++) {
        pushStep(`Analizando bloque ${i + 1}/${total} con Claude...`);
        const analysis = await analyzeBlock(transcriptBlocks[i], i + 1);
        const block = { number: i + 1, transcript: transcriptBlocks[i], analysis, status: "done" };
        analyzed.push(block);
        setBlocks([...analyzed]);
        updateLastStep(`✓ Bloque ${i + 1}/${total} analizado: ${analysis.titulo}`);
      }

      pushStep("Generando documento final...");
      const doc = await generateFinalDoc(analyzed);
      setFinalDoc(doc);
      updateLastStep("✓ Documento listo");
      setPhase("done");
    } catch (err) {
      setError(err.message || "Ocurrió un error. Verificá tu API key y el formato del archivo.");
      setPhase("setup");
    }
  };

  const exportToPDF = () => {
    const win = window.open("", "_blank");
    const html = finalDoc
      .replace(/^## (.+)$/gm, "<h2>$1</h2>")
      .replace(/^### (.+)$/gm, "<h3>$1</h3>")
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/^- (.+)$/gm, "<li>$1</li>")
      .replace(/\n{2,}/g, "</p><p>")
      .replace(/\n/g, "<br/>");
    win.document.write(`<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8"><title>Apuntes de Clase</title>
<style>body{font-family:Georgia,serif;max-width:820px;margin:40px auto;line-height:1.9;color:#111;padding:0 24px}
h1{font-size:2em;border-bottom:3px solid #111;padding-bottom:10px}
h2{font-size:1.45em;color:#1a3a5c;margin-top:40px;padding-left:14px;border-left:4px solid #1a3a5c}
h3{font-size:1.15em;color:#2c5282;margin-top:24px}ul{padding-left:22px}li{margin:5px 0}p{margin:10px 0}
.meta{color:#666;font-style:italic;margin-bottom:28px}@media print{h2{page-break-before:auto}}</style></head>
<body><h1>📚 Apuntes de Clase</h1>
<p class="meta">Generado el ${new Date().toLocaleDateString("es-UY",{day:"2-digit",month:"long",year:"numeric"})}</p>
<p>${html}</p></body></html>`);
    win.document.close(); win.print();
  };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
        *{box-sizing:border-box;margin:0;padding:0}body{background:#07090f}
        .app{min-height:100vh;background:#07090f;color:#dde3f0;font-family:'Outfit',sans-serif;padding:20px 16px 60px}
        .wrap{max-width:580px;margin:0 auto}

        .hdr{text-align:center;padding:36px 0 40px}
        .hdr-badge{font-size:10px;letter-spacing:7px;text-transform:uppercase;color:#5b8fff;margin-bottom:14px;display:flex;align-items:center;justify-content:center;gap:8px}
        .hdr-dot{width:5px;height:5px;background:#5b8fff;border-radius:50%}
        .hdr-title{font-size:2.5em;font-weight:700;color:#f0f4ff;letter-spacing:-1.5px;line-height:1.05}
        .hdr-sub{font-size:13px;color:#3d4f6b;margin-top:10px;letter-spacing:.3px}

        .card{background:#0c1120;border:1px solid #161f35;border-radius:18px;padding:24px;margin-bottom:16px}
        .stitle{font-size:9px;letter-spacing:4px;text-transform:uppercase;color:#3d4f6b;margin-bottom:16px;padding-bottom:8px;border-bottom:1px solid #161f35}

        .label{font-size:12px;color:#4a5f7a;margin-bottom:7px;letter-spacing:.3px;display:block}
        .inp{width:100%;background:#080c16;border:1px solid #161f35;border-radius:10px;padding:12px 14px;color:#dde3f0;font-family:'DM Mono',monospace;font-size:12px;outline:none;transition:border-color .2s}
        .inp:focus{border-color:#5b8fff}
        .inp::placeholder{color:#1a2540}

        .drop{border:2px dashed #1a2540;border-radius:14px;padding:40px 24px;text-align:center;cursor:pointer;transition:all .2s;margin-top:16px;user-select:none}
        .drop:hover,.drop.over{border-color:#5b8fff;background:#0a0f1e}
        .drop.has{border-color:#22c55e;border-style:solid;background:#052e1615}
        .drop-icon{font-size:36px;margin-bottom:12px}
        .drop-main{font-size:14px;font-weight:600;color:#8294ae;margin-bottom:4px}
        .drop-sub{font-size:12px;color:#2a3b52}
        .file-ok{font-size:12px;color:#22c55e;margin-top:8px;font-family:'DM Mono',monospace}

        .tip{background:#0a0f1e;border:1px solid #161f35;border-radius:10px;padding:14px 16px;font-size:12px;color:#2a3b52;line-height:1.75;margin-top:16px}
        .tip strong{color:#4a7aaf}

        .btn{padding:13px 20px;border-radius:11px;border:none;cursor:pointer;font-family:'Outfit',sans-serif;font-size:14px;font-weight:600;transition:all .2s;width:100%;margin-top:16px;letter-spacing:.2px}
        .btn-blue{background:#5b8fff;color:#000}
        .btn-blue:hover:not(:disabled){background:#4a7eef;transform:translateY(-1px)}
        .btn-blue:disabled{background:#0f1f3d;color:#1a2d4a;cursor:not-allowed}
        .btn-green{background:#052e16;color:#22c55e;margin-top:12px}
        .btn-green:hover{background:#064020;transform:translateY(-1px)}
        .btn-outline{background:transparent;border:1px solid #1a2d4a;color:#4a5f7a;margin-top:10px}
        .btn-outline:hover{border-color:#5b8fff;color:#5b8fff}

        .steps{display:flex;flex-direction:column;gap:6px}
        .step{display:flex;align-items:center;gap:10px;font-size:13px;padding:10px 14px;border-radius:10px;transition:all .3s}
        .step.old{color:#2a3b52}
        .step.cur{color:#dde3f0;background:#0a0f1e;border:1px solid #161f35}
        .step-ic{width:18px;flex-shrink:0;text-align:center;font-size:13px}
        .spin{display:inline-block;animation:spin .9s linear infinite}
        @keyframes spin{to{transform:rotate(360deg)}}

        .bcard{background:#080c16;border:1px solid #161f35;border-radius:12px;padding:14px 16px;cursor:pointer;transition:border-color .2s;margin-bottom:10px}
        .bcard:hover{border-color:#242f45}
        .bcard.open{border-color:#5b8fff}
        .bcard-top{display:flex;align-items:flex-start;gap:10px}
        .bnum{font-size:9px;letter-spacing:3px;text-transform:uppercase;color:#5b8fff;padding-top:2px;flex-shrink:0}
        .btitle{font-size:14px;font-weight:600;color:#dde3f0;flex:1;line-height:1.3}
        .bsummary{font-size:12.5px;color:#3d4f6b;margin-top:8px;line-height:1.65}
        .bdetail{margin-top:14px;padding-top:14px;border-top:1px solid #161f35;display:flex;flex-direction:column;gap:14px}
        .dlbl{font-size:9px;letter-spacing:3px;text-transform:uppercase;color:#5b8fff;margin-bottom:7px}
        .concept{background:#0c1120;border-radius:8px;padding:10px 12px;margin-bottom:6px}
        .cname{font-size:13px;font-weight:600;color:#dde3f0;margin-bottom:4px}
        .cexp{font-size:12px;color:#4a5f7a;line-height:1.55}
        .tags{display:flex;flex-wrap:wrap;gap:5px}
        .tag{background:#161f35;border-radius:6px;padding:3px 9px;font-size:11.5px;color:#4a5f7a}
        .tag.bl{background:#0f1f3d;color:#7db4ff}
        .idea{font-size:12px;color:#4a5f7a;padding:4px 0 4px 12px;border-left:2px solid #1a2540;margin-bottom:4px;line-height:1.5}

        .fdoc{font-size:13px;color:#8294ae;line-height:1.85;white-space:pre-wrap;max-height:400px;overflow-y:auto;padding-right:4px}
        .fdoc::-webkit-scrollbar{width:3px}.fdoc::-webkit-scrollbar-track{background:#0c1120}.fdoc::-webkit-scrollbar-thumb{background:#1a2d4a;border-radius:3px}

        .err{background:#1a040420;border:1px solid #7f1d1d;border-radius:10px;padding:14px;color:#fca5a5;font-size:13px;line-height:1.65;margin-top:12px}
      `}</style>

      <div className="app">
        <div className="wrap">
          <div className="hdr">
            <div className="hdr-badge"><div className="hdr-dot"/>ClassMind<div className="hdr-dot"/></div>
            <h1 className="hdr-title">Subí tu clase.<br/>Claude la explica.</h1>
            <p className="hdr-sub">Grabá con cualquier app · Subís el audio · Exportás a PDF</p>
          </div>

          {/* ── SETUP ── */}
          {phase === "setup" && (
            <>
              <div className="card">
                <div className="stitle">API Key de OpenAI</div>
                <label className="label">Para transcribir el audio con Whisper</label>
                <input className="inp" type="password" placeholder="sk-..." value={key} onChange={(e) => setKey(e.target.value)} />
                <div className="tip">
                  💡 <strong>Para archivos chicos y rápidos:</strong> En tu grabadora, elegí formato M4A/AAC con calidad de voz (16–32 kbps). Una clase de 2 horas queda en ~15–30 MB y Whisper la procesa directo, sin dividir.
                </div>
              </div>

              <div className="card">
                <div className="stitle">Archivo de audio</div>
                <div
                  className={`drop ${dragOver ? "over" : ""} ${file ? "has" : ""}`}
                  onDrop={(e) => { e.preventDefault(); setDragOver(false); const f = e.dataTransfer.files[0]; if (f) setFile(f); }}
                  onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                  onDragLeave={() => setDragOver(false)}
                  onClick={() => fileRef.current?.click()}
                >
                  <div className="drop-icon">{file ? "🎧" : "📁"}</div>
                  {file ? (
                    <>
                      <div className="drop-main">{file.name}</div>
                      <div className="file-ok">{fmtSize(file.size)} · listo para procesar</div>
                    </>
                  ) : (
                    <>
                      <div className="drop-main">Tocá para seleccionar el archivo</div>
                      <div className="drop-sub">M4A · MP3 · WAV · OGG — hasta ~100 MB</div>
                    </>
                  )}
                </div>
                <input ref={fileRef} type="file" accept="audio/*" style={{ display: "none" }} onChange={(e) => { if (e.target.files[0]) setFile(e.target.files[0]); }} />

                <button className="btn btn-blue" onClick={processFile} disabled={!file || !key.trim()}>
                  ✦ Procesar clase
                </button>
                {error && <div className="err">⚠️ {error}</div>}
              </div>
            </>
          )}

          {/* ── PROCESSING / DONE ── */}
          {(phase === "processing" || phase === "done") && (
            <>
              {steps.length > 0 && (
                <div className="card">
                  <div className="stitle">Procesando</div>
                  <div className="steps">
                    {steps.map((s, i) => (
                      <div key={i} className={`step ${s.active ? "cur" : "old"}`}>
                        <span className="step-ic">{s.active ? <span className="spin">⟳</span> : "✓"}</span>
                        {s.msg}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {blocks.length > 0 && (
                <div className="card">
                  <div className="stitle">Bloques analizados</div>
                  {blocks.map((b) => (
                    <div key={b.number} className={`bcard ${expanded === b.number ? "open" : ""}`} onClick={() => setExpanded(expanded === b.number ? null : b.number)}>
                      <div className="bcard-top">
                        <span className="bnum">BLQ {b.number}</span>
                        <span className="btitle">{b.analysis?.titulo || `Bloque ${b.number}`}</span>
                      </div>
                      {b.analysis?.resumen && <p className="bsummary">{b.analysis.resumen}</p>}
                      {expanded === b.number && b.analysis && (
                        <div className="bdetail">
                          {b.analysis.temas_principales?.length > 0 && (
                            <div>
                              <div className="dlbl">Temas</div>
                              <div className="tags">{b.analysis.temas_principales.map((t, i) => <span key={i} className="tag">{t}</span>)}</div>
                            </div>
                          )}
                          {b.analysis.conceptos_clave?.length > 0 && (
                            <div>
                              <div className="dlbl">Conceptos Clave</div>
                              {b.analysis.conceptos_clave.map((c, i) => (
                                <div key={i} className="concept">
                                  <div className="cname">{c.concepto}</div>
                                  <div className="cexp">{c.explicacion}</div>
                                </div>
                              ))}
                            </div>
                          )}
                          {b.analysis.ideas_importantes?.length > 0 && (
                            <div>
                              <div className="dlbl">Ideas Importantes</div>
                              {b.analysis.ideas_importantes.map((idea, i) => <div key={i} className="idea">{idea}</div>)}
                            </div>
                          )}
                          {b.analysis.ejemplos?.length > 0 && (
                            <div>
                              <div className="dlbl">Ejemplos</div>
                              <div className="tags">{b.analysis.ejemplos.map((e, i) => <span key={i} className="tag bl">{e}</span>)}</div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {finalDoc && (
                <div className="card">
                  <div className="stitle">Documento final</div>
                  <div className="fdoc">{finalDoc}</div>
                  <button className="btn btn-green" onClick={exportToPDF}>↓ Exportar a PDF</button>
                  <button className="btn btn-outline" onClick={() => { setPhase("setup"); setFile(null); setSteps([]); setBlocks([]); setFinalDoc(""); }}>
                    + Procesar otra clase
                  </button>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </>
  );
}
