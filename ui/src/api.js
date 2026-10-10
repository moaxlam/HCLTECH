// The UI talks to the API under /api: the Vite dev server proxies that to the
// FastAPI service, and a production build may set VITE_API_BASE instead.
const BASE = import.meta.env.VITE_API_BASE || "/api";

async function request(path, options = {}) {
  const response = await fetch(`${BASE}${path}`, options);
  const text = await response.text();
  let body = null;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = { detail: text };
    }
  }
  if (!response.ok) {
    const detail = body && body.detail ? body.detail : `HTTP ${response.status}`;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return body;
}

export function getHealth() {
  return request("/health");
}

export function getSources() {
  return request("/sources");
}

export function getRules() {
  return request("/rules");
}

export function getRecentAudit(limit = 25) {
  return request(`/audit?limit=${limit}`);
}

export function getAudit(traceId) {
  return request(`/audit/${encodeURIComponent(traceId)}`);
}

export function getEvaluation() {
  return request("/evaluation");
}

export function ask(question, studentId, asOfDate) {
  const headers = { "Content-Type": "application/json" };
  if (studentId) headers["X-Student-Id"] = studentId;
  return request("/ask", {
    method: "POST",
    headers,
    body: JSON.stringify({ question, as_of_date: asOfDate || null }),
  });
}

export function ingest(file, metadata) {
  const form = new FormData();
  form.append("file", file);
  form.append("metadata", JSON.stringify(metadata));
  return request("/ingest", { method: "POST", body: form });
}
