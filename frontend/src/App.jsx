import { useState, useRef } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function App() {
  const [mode, setMode] = useState('single') // 'single' | 'batch'

  // single prediction state
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  // batch prediction state
  const [batchRows, setBatchRows] = useState(null) // [{title, text, label, confidence, raw_score}]
  const [batchLoading, setBatchLoading] = useState(false)
  const [batchError, setBatchError] = useState('')
  const fileRef = useRef(null)

  // ── single prediction ──
  const handleSingle = async (e) => {
    e.preventDefault()
    if (!text.trim()) return
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      })
      if (!res.ok) throw new Error('Prediction failed')
      setResult(await res.json())
    } catch {
      setError('Could not connect to the server. Make sure the backend is running.')
    } finally {
      setLoading(false)
    }
  }

  // ── CSV parsing ──
  const parseCSV = (csvText) => {
    const result = []
    const fields = []
    let current = ''
    let inQuotes = false
    for (let i = 0; i < csvText.length; i++) {
      const ch = csvText[i]
      if (ch === '"') {
        if (inQuotes && csvText[i + 1] === '"') {
          current += '"'
          i++
        } else {
          inQuotes = !inQuotes
        }
      } else if (ch === ',' && !inQuotes) {
        fields.push(current)
        current = ''
      } else if ((ch === '\n' || ch === '\r') && !inQuotes) {
        if (current.length || fields.length) {
          fields.push(current)
          result.push(fields.splice(0))
          current = ''
        }
        if (ch === '\r' && csvText[i + 1] === '\n') i++
      } else {
        current += ch
      }
    }
    if (current.length || fields.length) {
      fields.push(current)
      result.push(fields.splice(0))
    }
    return result
  }

  const readCSV = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = (e) => {
        const text = e.target.result
        const parsed = parseCSV(text)
        if (!parsed.length) return resolve([])
        const headers = parsed[0].map((h) => h.trim().toLowerCase())
        const rows = parsed.slice(1).map((fields) => {
          const obj = {}
          headers.forEach((h, i) => (obj[h] = fields[i] || ''))
          return obj
        })
        resolve(rows)
      }
      reader.onerror = reject
      reader.readAsText(file)
    })
  }

  // ── batch prediction ──
  const handleBatch = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return

    setBatchLoading(true)
    setBatchError('')
    setBatchRows(null)

    try {
      const rows = await readCSV(file)
      if (!rows.length) throw new Error('CSV is empty')

      const textCol = rows[0].text !== undefined ? 'text' : Object.keys(rows[0])[0]
      const articles = rows.map((r) => r[textCol] || '')

      const res = await fetch(`${API_URL}/predict/batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ articles }),
      })
      if (!res.ok) throw new Error('Batch prediction failed')

      const data = await res.json()
      const merged = rows.map((row, i) => ({
        title: row.title || '',
        text: row[textCol] || '',
        label: data.predictions[i].label,
        confidence: data.predictions[i].confidence,
        raw_score: data.predictions[i].raw_score,
      }))
      setBatchRows(merged)
    } catch (err) {
      setBatchError(err.message || 'Failed to process CSV.')
    } finally {
      setBatchLoading(false)
      if (fileRef.current) fileRef.current.value = ''
    }
  }

  // ── download results ──
  const downloadCSV = () => {
    if (!batchRows) return
    const header = 'title,text,label,confidence,raw_score'
    const escape = (s) => `"${String(s).replace(/"/g, '""')}"`
    const lines = batchRows.map(
      (r) => [escape(r.title), escape(r.text), r.label, r.confidence, r.raw_score].join(',')
    )
    const blob = new Blob([header + '\n' + lines.join('\n')], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'predictions.csv'
    a.click()
    URL.revokeObjectURL(url)
  }

  const confidencePercent = result ? (result.confidence * 100).toFixed(1) : 0
  const isFake = result?.label === 'Fake'

  return (
    <div className="app">
      <div className="container">
        <h1 className="title">Fake News Detector</h1>
        <p className="subtitle">Detect fake news using a Bidirectional LSTM model</p>

        {/* Mode toggle */}
        <div className="mode-toggle">
          <button
            className={`mode-btn ${mode === 'single' ? 'active' : ''}`}
            onClick={() => setMode('single')}
          >
            Single Prediction
          </button>
          <button
            className={`mode-btn ${mode === 'batch' ? 'active' : ''}`}
            onClick={() => setMode('batch')}
          >
            Batch Prediction
          </button>
        </div>

        {/* ── Single mode ── */}
        {mode === 'single' && (
          <>
            <form onSubmit={handleSingle} className="form">
              <textarea
                className="input"
                rows={6}
                placeholder="Enter news text here..."
                value={text}
                onChange={(e) => setText(e.target.value)}
              />
              <button className="btn" type="submit" disabled={loading || !text.trim()}>
                {loading ? 'Analyzing...' : 'Check News'}
              </button>
            </form>

            {error && <p className="error">{error}</p>}

            {result && (
              <div className={`result ${isFake ? 'fake' : 'real'}`}>
                <div className="result-label">{result.label}</div>
                <div className="result-confidence">{confidencePercent}% confidence</div>
                <div className="progress-bar">
                  <div
                    className={`progress-fill ${isFake ? 'fake' : 'real'}`}
                    style={{ width: `${confidencePercent}%` }}
                  />
                </div>
              </div>
            )}
          </>
        )}

        {/* ── Batch mode ── */}
        {mode === 'batch' && (
          <>
            <div className="batch-upload">
              <label className="upload-label">
                <span>Upload a CSV file with a <code>text</code> column (and optional <code>title</code> column)</span>
                <input
                  ref={fileRef}
                  type="file"
                  accept=".csv"
                  onChange={handleBatch}
                  className="file-input"
                />
                <div className="upload-btn">
                  {batchLoading ? 'Processing...' : 'Choose CSV File'}
                </div>
              </label>
            </div>

            {batchError && <p className="error">{batchError}</p>}

            {batchRows && (
              <div className="batch-results">
                <div className="batch-header">
                  <h3>{batchRows.length} predictions</h3>
                  <button className="btn btn-sm" onClick={downloadCSV}>
                    Download CSV
                  </button>
                </div>

                <div className="table-wrapper">
                  <table className="results-table">
                    <thead>
                      <tr>
                        <th>#</th>
                        {batchRows[0].title && <th>Title</th>}
                        <th>Text</th>
                        <th>Label</th>
                        <th>Confidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {batchRows.map((row, i) => (
                        <tr key={i}>
                          <td className="cell-num">{i + 1}</td>
                          {row.title && <td className="cell-title">{row.title}</td>}
                          <td className="cell-text">{row.text.slice(0, 120)}...</td>
                          <td>
                            <span className={`badge ${row.label === 'Fake' ? 'badge-fake' : 'badge-real'}`}>
                              {row.label}
                            </span>
                          </td>
                          <td className="cell-conf">{(row.confidence * 100).toFixed(1)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

export default App
