import React, { useEffect, useState } from "react";
import { getHealth } from "./api.js";
import { AskPanel, IngestPanel, RulesPanel, SourcesPanel } from "./panels.jsx";

const TABS = [
  { id: "ask", label: "Ask" },
  { id: "ingest", label: "Ingest a document" },
  { id: "sources", label: "Source register" },
  { id: "rules", label: "Rule registry" },
];

export default function App() {
  const [tab, setTab] = useState("ask");
  const [studentId, setStudentId] = useState(() => localStorage.getItem("studentId") || "S2001");
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState("");

  useEffect(() => {
    localStorage.setItem("studentId", studentId);
  }, [studentId]);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const next = await getHealth();
        if (!cancelled) {
          setHealth(next);
          setHealthError("");
        }
      } catch (error) {
        if (!cancelled) {
          setHealth(null);
          setHealthError(error.message);
        }
      }
    };
    load();
    const timer = setInterval(load, 15000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  const status = health ? health.status : healthError ? "down" : "unknown";

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <h1>University Student Services Assistant</h1>
          <p className="subtitle">
            Answers and eligibility decisions from authorised university documents, with citations,
            deterministic tools and an audit record for every response.
          </p>
        </div>
        <div className="identity">
          <span className={`pill pill-${status}`}>API {status}</span>
          <label className="field inline">
            <span>Signed-in student (X-Student-Id)</span>
            <input
              value={studentId}
              onChange={(event) => setStudentId(event.target.value.trim().toUpperCase())}
              placeholder="S1001"
              spellCheck={false}
            />
          </label>
          {health ? (
            <span className="muted small">
              {health.mock_llm
                ? "MOCK_LLM"
                : health.llm?.detail?.toLowerCase().includes("groq")
                ? "Groq API"
                : "local Ollama"} · {health.documents_indexed} documents indexed
            </span>
          ) : null}
        </div>
      </header>

      <nav className="tabs">
        {TABS.map((item) => (
          <button
            key={item.id}
            className={item.id === tab ? "tab tab-active" : "tab"}
            onClick={() => setTab(item.id)}
          >
            {item.label}
          </button>
        ))}
      </nav>

      <main>
        {tab === "ask" && <AskPanel studentId={studentId} />}
        {tab === "ingest" && <IngestPanel />}
        {tab === "sources" && <SourcesPanel />}
        {tab === "rules" && <RulesPanel />}
      </main>

      <footer className="muted small">
        Identity is taken only from the request context, never from the question text. Document content is
        treated as data, never as instructions.
      </footer>
    </div>
  );
}
