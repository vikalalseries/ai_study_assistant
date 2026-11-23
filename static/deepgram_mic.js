/* Browser side of voice typing with Deepgram. No libraries needed. Works on Vercel (no server WebSocket needed).
 *
 * Two modes, chosen automatically:
 *  1. LIVE streaming: the page asks your server for a short-lived token (POST tokenUrl), then streams the microphone
 *     straight to Deepgram over a WebSocket. Words appear as you speak. Needs a Deepgram key that is allowed to
 *     create temporary tokens (Member role or higher).
 *  2. SEGMENTS (fallback, works with any key): the microphone is recorded in short pieces (about 3.5 s). Each piece is
 *     POSTed to your server (transcribeUrl), which asks Deepgram over plain HTTP. Words appear every few seconds.
 * Your real API key never reaches the browser in either mode.
 *
 *   const mic = createMic({
 *     input: document.querySelector("#topic"),     // <textarea> or <input>
 *     button: document.querySelector("#micBtn"),   // mic toggle button
 *     onError: msg => alert(msg),                  // optional, e.g. a toast
 *     onChange: () => {},                          // optional, e.g. auto-grow the box
 *   });
 *   mic.cancel();   // call when the message is sent: stops and keeps nothing new
 *
 * Tap the button to start, tap again to stop (text is kept). Esc cancels.
 * The button gets class "rec" and aria-pressed="true" while recording; it is hidden
 * if the browser cannot record audio. The mic only works on localhost or HTTPS.
 */
function createMic({ input, button, tokenUrl = "/api/voice-token", transcribeUrl = "/api/transcribe", language = "multi", segmentMs = 2500, maxSegmentMs = 6500, onError = () => {}, onChange = () => {} }) {
  if (!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.MediaRecorder)) {
    button.hidden = true; button.style.display = "none";
    return { cancel() {}, stop() {}, active: () => false };
  }
  let rec = null, stream = null, ws = null, base = "", fin = "", cancelled = false, closer = 0, starting = false;
  let segmented = false, stopping = false, segTimer = 0, pending = 0, noLive = false, errored = false, actx = null, analyser = null;

  const ui = on => {
    button.classList.toggle("rec", on);
    button.setAttribute("aria-pressed", on);
    button.setAttribute("aria-label", on ? "Stop recording" : "Speak instead of typing");
  };
  const show = interim => {
    input.value = [base, fin, interim].filter(Boolean).join(" ");
    onChange(); input.scrollTop = input.scrollHeight;
  };
  const cleanup = () => {
    clearTimeout(closer); clearTimeout(segTimer);
    try { rec && rec.state !== "inactive" && rec.stop(); } catch {}
    if (stream) stream.getTracks().forEach(t => t.stop());   // turns the browser's recording light off
    if (actx) { try { actx.close(); } catch {} }
    rec = stream = actx = analyser = null; ui(false);
  };
  const fail = msg => { if (!errored) { errored = true; onError(msg); } };

  /* ---------- mode 1: live streaming straight to Deepgram ---------- */
  async function startLive(token) {
    base = input.value.trim(); fin = ""; cancelled = false;
    const qs = new URLSearchParams({ model: "nova-3", language, interim_results: "true", smart_format: "true", punctuate: "true" });
    ws = new WebSocket("wss://api.deepgram.com/v1/listen?" + qs, ["bearer", token]);
    ws.onmessage = e => {
      if (cancelled) return;
      let m; try { m = JSON.parse(e.data); } catch { return; }
      if (m.type !== "Results") return;
      const alt = m.channel && m.channel.alternatives && m.channel.alternatives[0];
      const text = ((alt && alt.transcript) || "").trim();
      if (!text) return;
      if (m.is_final) { fin = (fin ? fin + " " : "") + text; show(""); } else show(text);
    };
    ws.onerror = () => { if (!cancelled) { fail("Could not reach the speech service."); cleanup(); } };
    ws.onclose = () => { cleanup(); ws = null; if (!cancelled) { show(""); input.focus(); } };
    ws.onopen = () => {
      try {
        rec = new MediaRecorder(stream);
        rec.ondataavailable = e => { if (e.data.size && ws && ws.readyState === 1) ws.send(e.data); };
        rec.onstop = () => {   // flush the last spoken words, then let Deepgram close the connection
          if (ws && ws.readyState === 1) {
            ws.send(JSON.stringify({ type: "Finalize" }));
            ws.send(JSON.stringify({ type: "CloseStream" }));
            closer = setTimeout(() => { try { ws && ws.close(); } catch {} }, 4000);   // safety net
          }
        };
        rec.start(250);   // 250 ms chunks = live streaming
        ui(true);
      } catch { fail("Could not start recording."); cleanup(); try { ws.close(); } catch {} }
    };
  }

  /* ---------- mode 2: short segments sent to the server over plain HTTP ---------- */
  function finishSegments() {
    segmented = false; stopping = false; cleanup();
    if (!cancelled) { show(""); input.focus(); }
  }
  async function sendSegment(blob) {
    pending++;
    try {
      const r = await fetch(transcribeUrl, { method: "POST", headers: { "Content-Type": blob.type || "audio/webm" }, body: blob });
      if (!r.ok) { let d = ""; try { d = (await r.json()).detail; } catch {} throw new Error(d || "Speech to text failed"); }
      const text = ((await r.json()).text || "").trim();
      if (text && !cancelled) { fin = (fin ? fin + " " : "") + text; show(""); }
    } catch (e) { if (!cancelled) fail(e.message || "Speech to text failed"); }
    pending--;
    if (stopping && pending === 0 && (!rec || rec.state === "inactive")) finishSegments();
  }
  function recordSegment() {
    const r = new MediaRecorder(stream), chunks = [];
    rec = r;
    r.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
    r.onstop = () => {
      if (!stopping && !cancelled && segmented) recordSegment();   // next piece starts at once, so no speech is lost
      const blob = new Blob(chunks, { type: r.mimeType || "audio/webm" });
      if (!cancelled) sendSegment(blob);
      else if (stopping && pending === 0) finishSegments();
    };
    r.start();
    // Cut the recording at a natural pause (so words are not sliced in half): not before segmentMs, never after maxSegmentMs.
    const t0 = Date.now(), buf = analyser ? new Uint8Array(analyser.fftSize) : null;
    let quietSince = 0;
    const poll = () => {
      if (r.state !== "recording") return;
      const age = Date.now() - t0;
      let quiet = false;
      if (buf) { analyser.getByteTimeDomainData(buf); let peak = 0; for (const v of buf) peak = Math.max(peak, Math.abs(v - 128)); quiet = peak < 4; }
      quietSince = quiet ? (quietSince || Date.now()) : 0;
      if (age >= maxSegmentMs || (age >= segmentMs && quietSince && Date.now() - quietSince >= 350)) { r.stop(); return; }
      segTimer = setTimeout(poll, 100);
    };
    segTimer = setTimeout(poll, 100);
  }
  function startSegments() {
    base = input.value.trim(); fin = ""; cancelled = false; stopping = false; errored = false; pending = 0;
    try {   // a level meter, used to find pauses in speech
      actx = new (window.AudioContext || window.webkitAudioContext)();
      analyser = actx.createAnalyser(); analyser.fftSize = 1024;
      actx.createMediaStreamSource(stream).connect(analyser);
    } catch { actx = analyser = null; }
    segmented = true; recordSegment(); ui(true);
  }

  async function start() {
    if (starting || rec) return;
    starting = true; errored = false;
    try {
      try { stream = await navigator.mediaDevices.getUserMedia({ audio: true }); }
      catch { onError("Microphone permission was denied."); return; }
      let token = null;
      if (!noLive) {
        try {
          const r = await fetch(tokenUrl, { method: "POST" });
          if (r.ok) token = (await r.json()).token;
        } catch {}
        if (!token) noLive = true;   // this key cannot make tokens: use segments from now on (until the page is reloaded)
      }
      if (token) await startLive(token); else startSegments();
    } finally { starting = false; }
  }
  function stop(cancel) {
    cancelled = !!cancel;
    if (cancel) { input.value = base; onChange(); }
    if (segmented) {
      stopping = true; clearTimeout(segTimer); ui(false);
      if (rec && rec.state === "recording") rec.stop();   // sends the last piece, then finishSegments() runs when it is transcribed
      else if (pending === 0) finishSegments();
      if (cancel) finishSegments();
      return;
    }
    if (rec && rec.state !== "inactive") { try { rec.stop(); } catch {} }
    else if (ws) ws.close();
    if (cancel && ws) ws.close();
    ui(false);
  }
  button.addEventListener("click", () => (rec ? stop(false) : start()));
  document.addEventListener("keydown", e => { if (e.key === "Escape" && rec) stop(true); });
  return { stop: () => stop(false), cancel: () => stop(true), active: () => !!(rec || ws || segmented) };
}
