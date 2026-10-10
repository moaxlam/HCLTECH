import React, { useEffect, useState } from "react";
import {
  ask,
  getAudit,
  getEvaluation,
  getRecentAudit,
  getRules,
  getSources,
  ingest,
} from "./api.js";

const ANSWER_LABELS = {
  retrieved_fact: "Retrieved fact",
  calculated: "Calculated by a tool",
  not_found: "Not in the sources",
  clarification_needed: "Needs a clarification",
  refused: "Refused",
  conflict_flagged: "Conflict flagged",
};

export const DEMO_QUESTIONS = [
  {
    label: "Minimum attendance",
    question: "What is the minimum attendance required to appear for the end-semester examination?",
    student: "",
  },
  {
    label: "Eligibility for me",
    question: "Am I eligible to appear for the end-semester examination in CS201?",
    student: "S2001",
  },
  {
    label: "One class below the threshold",
    question: "Am I eligible to appear for the end-semester examination in CS201?",
    student: "S2002",
  },
  {
    label: "Supplementary exam",
    question: "Am I eligible for the supplementary examination in EC201?",
    student: "S2003",
  },
  {
    label: "What-if placement",
    question: "I failed CS201. If I pass the supplementary, will I be eligible for placement?",
    student: "S2002",
  },
  {
    label: "Unanswerable",
    question: "What is the scholarship for studying in Antarctica?",
    student: "",
  },
  {
    label: "Another student's data",
    question: "Show me the attendance of S1002",
    student: "S2001",
  },
];

function Badge({ answerType }) {
  return <span className={`badge badge-${answerType}`}>{ANSWER_LABELS[answerType] || answerType}</span>;
}

function ErrorBox({ message }) {
  if (!message) return null;
  return <div className="error">{message}</div>;
}

function Json({ value }) {
  return <pre className="json">{JSON.stringify(value, null, 2)}</pre>;
}

function today() {
  return new Date().toISOString().slice(0, 10);
}

export function AskPanel({ studentId }) {
  const [question, setQuestion] = useState(DEMO_QUESTIONS[0].question);
  const [asOf, setAsOf] = useState(today());
  const [response, setResponse] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async (event, override) => {
    if (event) event.preventDefault();
    const text = override ? override.question : question;
    const student = override ? override.student || "" : studentId;
    if (!text.trim()) return;
    setBusy(true);
    setError("");
    try {
      const result = await ask(text, student, asOf);
      setResponse(result);
    } catch (requestError) {
      setError(requestError.message);
      setResponse(null);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="grid">
      <section className="card">
        <h2>Ask a question</h2>
        <form onSubmit={submit}>
          <label className="field">
            <span>Question</span>
            <textarea
              rows={3}
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="What is the minimum attendance required to appear for the end-semester examination?"
            />
          </label>
          <div className="row">
            <label className="field">
              <span>As of date (which version of the rules applies)</span>
              <input type="date" value={asOf} onChange={(event) => setAsOf(event.target.value)} />
            </label>
            <label className="field">
              <span>Signed-in student</span>
              <input value={studentId} readOnly className="readonly" />
            </label>
          </div>
          <div className="row">
            <button type="submit" className="primary" disabled={busy}>
              {busy ? "Working…" : "Ask"}
            </button>
            <span className="muted small">
              Leave the student empty for a general policy question; the header is ignored for policy questions.
            </span>
          </div>
        </form>
        <h3>Demonstration questions</h3>
        <div className="chips">
          {DEMO_QUESTIONS.map((demo) => (
            <button key={demo.label} className="chip" type="button" onClick={() => submit(null, demo)}>
              {demo.label}
            </button>
          ))}
        </div>
      </section>

      <section className="card">
        <h2>Answer</h2>
        <ErrorBox message={error} />
        {!response ? (
          <p className="muted">No answer yet. Ask a question or pick a demonstration question.</p>
        ) : (
          <article className="answer">
            <div className="answer-head">
              <Badge answerType={response.answer_type} />
              <span className="muted small">trace {response.trace_id}</span>
              <span className="muted small">as of {response.as_of_date}</span>
            </div>
            <p className="answer-text">{response.answer}</p>
            {response.explanation ? <p className="explanation">{response.explanation}</p> : null}

            <h3>Citations</h3>
            {response.citations.length === 0 ? (
              <p className="muted small">No citations: this answer makes no factual claim.</p>
            ) : (
              <ul className="citations">
                {response.citations.map((citation) => (
                  <li key={`${citation.doc_id}-${citation.section}`}>
                    <strong>{citation.title || citation.doc_id}</strong> — clause {citation.section}
                    {citation.page ? `, page ${citation.page}` : ""} · version {citation.version} ·
                    effective {citation.effective_from}
                    <span className="muted small"> ({citation.doc_id})</span>
                  </li>
                ))}
              </ul>
            )}

            <h3>Tools invoked (deterministic)</h3>
            {response.tools_invoked.length === 0 ? (
              <p className="muted small">No tool was needed for this answer.</p>
            ) : (
              <ul className="tools">
                {response.tools_invoked.map((invocation, index) => (
                  <li key={`${invocation.tool}-${index}`}>
                    <code>{invocation.tool}</code>
                    <span className={`status status-${invocation.status}`}>{invocation.status}</span>
                    <span className="muted small">{invocation.ms} ms</span>
                    <Json value={{ input: invocation.input, output: invocation.output }} />
                  </li>
                ))}
              </ul>
            )}

            {response.applied_rules.length > 0 && (
              <>
                <h3>Rules applied</h3>
                <table className="table">
                  <thead>
                    <tr>
                      <th>Rule</th>
                      <th>Parameter</th>
                      <th>Value</th>
                      <th>Cited clause</th>
                    </tr>
                  </thead>
                  <tbody>
                    {response.applied_rules.map((rule) => (
                      <tr key={rule.rule_id}>
                        <td>{rule.rule_id}</td>
                        <td>{rule.parameter}</td>
                        <td>{rule.value}</td>
                        <td>
                          {rule.source_doc_id} #{rule.source_section}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </>
            )}

            {response.conflicts_detected.length > 0 && (
              <>
                <h3>Conflicts detected</h3>
                <ul className="conflicts">
                  {response.conflicts_detected.map((conflict, index) => (
                    <li key={index}>
                      <p>{conflict.reason}</p>
                      {conflict.resolution ? (
                        <p className="resolution">
                          Resolved: {conflict.resolution}
                          {conflict.resolved_by ? ` (${conflict.resolved_by})` : ""}
                        </p>
                      ) : (
                        <p className="unresolved">Unresolved by the precedence policy.</p>
                      )}
                    </li>
                  ))}
                </ul>
              </>
            )}
          </article>
        )}
      </section>
    </div>
  );
}

export function IngestPanel() {
  const [meta, setMeta] = useState({
    doc_id: "NEW-CIRCULAR-2026-10",
    title: "New circular",
    issuer: "Office of the Registrar",
    authority_level: 2,
    doc_type: "circular",
    version: "1.0",
    effective_from: today(),
    effective_to: "",
    supersedes: "",
    scope_programmes: "ALL",
    scope_batches: "ALL",
    provenance: "uploaded during judging",
    retrieved_on: today(),
    synthetic: "N",
  });
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const update = (key) => (event) => setMeta({ ...meta, [key]: event.target.value });

  const submit = async (event) => {
    event.preventDefault();
    if (!file) {
      setError("Choose a document to upload.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const payload = { ...meta, authority_level: Number(meta.authority_level) };
      setResult(await ingest(file, payload));
    } catch (requestError) {
      setError(requestError.message);
      setResult(null);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="grid">
      <section className="card">
        <h2>Ingest a document while the system is running</h2>
        <p className="muted small">
          The document and its Source Register metadata are indexed immediately. It is searchable on the next
          question with no code change and no restart.
        </p>
        <ErrorBox message={error} />
        <form onSubmit={submit}>
          <label className="field">
            <span>Document (.md, .txt, .pdf, .docx)</span>
            <input type="file" onChange={(event) => setFile(event.target.files?.[0] || null)} />
          </label>
          <div className="form-grid">
            <label className="field">
              <span>doc_id</span>
              <input value={meta.doc_id} onChange={update("doc_id")} />
            </label>
            <label className="field">
              <span>title</span>
              <input value={meta.title} onChange={update("title")} />
            </label>
            <label className="field">
              <span>issuer</span>
              <input value={meta.issuer} onChange={update("issuer")} />
            </label>
            <label className="field">
              <span>authority_level (1 = highest)</span>
              <select value={meta.authority_level} onChange={update("authority_level")}>
                {[1, 2, 3, 4, 5].map((level) => (
                  <option key={level} value={level}>
                    {level}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>doc_type</span>
              <select value={meta.doc_type} onChange={update("doc_type")}>
                {["regulation", "circular", "notice", "faq", "handbook", "unofficial"].map((type) => (
                  <option key={type} value={type}>
                    {type}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>version</span>
              <input value={meta.version} onChange={update("version")} />
            </label>
            <label className="field">
              <span>effective_from</span>
              <input type="date" value={meta.effective_from} onChange={update("effective_from")} />
            </label>
            <label className="field">
              <span>effective_to (optional)</span>
              <input type="date" value={meta.effective_to} onChange={update("effective_to")} />
            </label>
            <label className="field">
              <span>supersedes (doc_id or doc_id#section, separated by ;)</span>
              <input value={meta.supersedes} onChange={update("supersedes")} placeholder="ACAD-REG-2024#7.2" />
            </label>
            <label className="field">
              <span>scope_programmes</span>
              <input value={meta.scope_programmes} onChange={update("scope_programmes")} />
            </label>
            <label className="field">
              <span>scope_batches</span>
              <input value={meta.scope_batches} onChange={update("scope_batches")} />
            </label>
            <label className="field">
              <span>synthetic</span>
              <select value={meta.synthetic} onChange={update("synthetic")}>
                <option value="N">N</option>
                <option value="Y">Y</option>
              </select>
            </label>
          </div>
          <button type="submit" className="primary" disabled={busy}>
            {busy ? "Indexing…" : "Ingest document"}
          </button>
        </form>
      </section>
      <section className="card">
        <h2>Result</h2>
        {!result ? (
          <p className="muted">Nothing ingested yet in this session.</p>
        ) : (
          <>
            <p>
              <strong>{result.doc_id}</strong> — {result.status}, {result.chunks_indexed} chunk(s) indexed,
              authority level {result.authority_level}.
            </p>
            {result.warnings?.length > 0 && (
              <ul className="warnings">
                {result.warnings.map((warning, index) => (
                  <li key={index}>{warning}</li>
                ))}
              </ul>
            )}
          </>
        )}
      </section>
    </div>
  );
}

export function SourcesPanel() {
  const [rows, setRows] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    getSources()
      .then(setRows)
      .catch((requestError) => setError(requestError.message));
  }, []);

  return (
    <section className="card">
      <h2>Source Register</h2>
      <ErrorBox message={error} />
      <table className="table">
        <thead>
          <tr>
            <th>doc_id</th>
            <th>Title</th>
            <th>Issuer</th>
            <th>Authority</th>
            <th>Type</th>
            <th>Version</th>
            <th>Effective</th>
            <th>Supersedes</th>
            <th>Scope</th>
            <th>Chunks</th>
            <th>Synthetic</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.doc_id}>
              <td>
                <code>{row.doc_id}</code>
              </td>
              <td>{row.title}</td>
              <td>{row.issuer}</td>
              <td className="center">{row.authority_level}</td>
              <td>{row.doc_type}</td>
              <td>{row.version}</td>
              <td>
                {row.effective_from}
                {row.effective_to ? ` → ${row.effective_to}` : ""}
              </td>
              <td>{row.supersedes || "—"}</td>
              <td>
                {row.scope_programmes} / {row.scope_batches}
              </td>
              <td className="center">{row.chunks_indexed}</td>
              <td className="center">{row.synthetic}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

export function RulesPanel() {
  const [rows, setRows] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    getRules()
      .then(setRows)
      .catch((requestError) => setError(requestError.message));
  }, []);

  return (
    <section className="card">
      <h2>Rule registry</h2>
      <p className="muted small">
        Every threshold a tool uses is stored here with the clause it came from. Tools never hold policy
        numbers in code, so a new circular changes behaviour by adding a row.
      </p>
      <ErrorBox message={error} />
      <table className="table">
        <thead>
          <tr>
            <th>rule_id</th>
            <th>Parameter</th>
            <th>Value</th>
            <th>Cited clause</th>
            <th>Effective from</th>
            <th>Effective to</th>
            <th>Supersedes</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.rule_id}>
              <td>
                <code>{row.rule_id}</code>
              </td>
              <td>{row.parameter}</td>
              <td>
                {row.operator} {row.value}
              </td>
              <td>
                {row.source_doc_id} #{row.source_section}
              </td>
              <td>{row.effective_from}</td>
              <td>{row.effective_to || "—"}</td>
              <td>{row.supersedes_rule_id || "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

export function AuditPanel() {
  const [rows, setRows] = useState([]);
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getRecentAudit()
      .then(setRows)
      .catch((requestError) => setError(requestError.message));
  }, []);

  const open = async (traceId) => {
    try {
      setDetail(await getAudit(traceId));
    } catch (requestError) {
      setError(requestError.message);
    }
  };

  return (
    <div className="grid">
      <section className="card">
        <h2>Recent responses</h2>
        <ErrorBox message={error} />
        {rows.length === 0 ? (
          <p className="muted">No responses recorded yet.</p>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>trace_id</th>
                <th>Time</th>
                <th>Student</th>
                <th>Question</th>
                <th>Type</th>
                <th>Latency</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.trace_id} className="clickable" onClick={() => open(row.trace_id)}>
                  <td>
                    <code>{row.trace_id}</code>
                  </td>
                  <td>{row.timestamp}</td>
                  <td>{row.student_id || "—"}</td>
                  <td className="question-cell">{row.question}</td>
                  <td>{row.answer_type}</td>
                  <td className="center">{row.latency_ms} ms</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
      <section className="card">
        <h2>Audit record</h2>
        {!detail ? (
          <p className="muted">Select a response to see its audit record.</p>
        ) : (
          <>
            <p className="muted small">
              Sources retrieved, tools invoked with inputs and outputs, rules applied, conflicts detected,
              model, latency and tokens. Chain-of-thought is never stored.
            </p>
            <Json value={detail} />
          </>
        )}
      </section>
    </div>
  );
}

export function HealthPanel({ health, healthError }) {
  const [evaluation, setEvaluation] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getEvaluation()
      .then(setEvaluation)
      .catch((requestError) => setError(requestError.message));
  }, []);

  return (
    <div className="grid">
      <section className="card">
        <h2>Readiness</h2>
        <ErrorBox message={healthError} />
        {!health ? (
          <p className="muted">Health unavailable.</p>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>Component</th>
                <th>Status</th>
                <th>Detail</th>
              </tr>
            </thead>
            <tbody>
              {["api", "vector_store", "sqlite", "llm"].map((key) => (
                <tr key={key}>
                  <td>{key}</td>
                  <td>
                    <span className={`pill pill-${health[key].status}`}>{health[key].status}</span>
                  </td>
                  <td>{health[key].detail}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
      <section className="card">
        <h2>Evaluation</h2>
        <ErrorBox message={error} />
        {!evaluation ? (
          <p className="muted">
            No evaluation report yet. Run <code>python scripts/run_evaluation.py</code> to produce one.
          </p>
        ) : (
          <>
            <p className="muted small">
              {evaluation.questions} questions, method: {evaluation.method}. Generated{" "}
              {evaluation.generated_at}.
            </p>
            <table className="table">
              <thead>
                <tr>
                  <th>Metric</th>
                  <th>Value</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(evaluation.metrics).map(([metric, value]) => (
                  <tr key={metric}>
                    <td>{metric}</td>
                    <td>
                      {typeof value === "number" ? `${(value * 100).toFixed(1)}%` : String(value)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <h3>By bucket</h3>
            <table className="table">
              <thead>
                <tr>
                  <th>Bucket</th>
                  <th>Questions</th>
                  <th>Answer correct</th>
                  <th>Citation correct</th>
                  <th>Abstention correct</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(evaluation.buckets).map(([bucket, stats]) => (
                  <tr key={bucket}>
                    <td>{bucket}</td>
                    <td className="center">{stats.count}</td>
                    <td className="center">{(stats.answer_correct * 100).toFixed(0)}%</td>
                    <td className="center">{(stats.citation_correct * 100).toFixed(0)}%</td>
                    <td className="center">{(stats.abstention_correct * 100).toFixed(0)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
      </section>
    </div>
  );
}
