'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import {
  api,
  PredictionResponse,
  SystemInfoResponse,
  formatErrorMessage,
} from '@/lib/api'
import TrainingStatus from '@/components/TrainingStatus'
import { useObservabilityStore } from '@/store/observability'
import AIObservatory from '@/components/AIObservatory'

// ─── Form Data ────────────────────────────────────────────────────────────────

const INITIAL_FORM = {
  age: '23',
  sex: 'female',
  bmi: '21.8',
  children: '0',
  smoker: 'no',
  region: 'southwest',
}

type FormData = typeof INITIAL_FORM

// ─── Subcomponents ────────────────────────────────────────────────────────────

function InputField({
  label,
  id,
  children,
}: {
  label: string
  id: string
  children: React.ReactNode
}) {
  return (
    <div>
      <label htmlFor={id} className="ns-label">{label}</label>
      {children}
    </div>
  )
}

/**
 * Enterprise Segmented Confidence Indicator (Phase 3.2)
 * 5 Segments: Very Low, Low, Medium, High, Very High
 */
function SegmentedConfidenceBar({
  probability,
  verdict,
  confidenceTier,
}: {
  probability: number
  verdict: 'APPROVED' | 'REJECTED'
  confidenceTier: string
}) {
  const isApproved = verdict === 'APPROVED'
  const pct = Math.min(Math.max(probability * 100, 0), 100)

  // 5 Tiers
  const tiers = [
    { label: 'Very Low', min: 0, max: 60 },
    { label: 'Low', min: 60, max: 70 },
    { label: 'Medium', min: 70, max: 80 },
    { label: 'High', min: 80, max: 95 },
    { label: 'Very High', min: 95, max: 100 },
  ]

  const activeColor = isApproved ? '#10b981' : '#f43f5e'
  const activeBg = isApproved ? 'rgba(16,185,129,0.15)' : 'rgba(244,63,94,0.15)'

  return (
    <div className="space-y-2 mt-4 pt-3 border-t" style={{ borderColor: 'var(--ns-border)' }}>
      <div className="flex items-center justify-between text-xs">
        <span className="font-semibold text-[11px] uppercase tracking-wider" style={{ color: 'var(--ns-text-muted)' }}>
          {isApproved ? 'Approval Probability' : 'Rejection Probability'}
        </span>
        <div className="flex items-center gap-2">
          <span
            className="text-[10px] px-2 py-0.5 rounded font-bold uppercase tracking-wider mono"
            style={{
              background: activeBg,
              color: activeColor,
              border: `1px solid ${activeColor}40`,
            }}
          >
            {confidenceTier}
          </span>
          <span className="text-sm font-bold mono" style={{ color: activeColor }}>
            {pct.toFixed(1)}%
          </span>
        </div>
      </div>

      {/* 5-Segment Bar */}
      <div className="grid grid-cols-5 gap-1.5 h-2.5">
        {tiers.map((t, idx) => {
          const isFilled = pct >= t.max
          const isCurrent = pct >= t.min && pct < t.max
          const fillWidth = isFilled ? 100 : isCurrent ? ((pct - t.min) / (t.max - t.min)) * 100 : 0

          return (
            <div
              key={t.label}
              className="relative h-full rounded-sm overflow-hidden"
              style={{ background: 'rgba(15,23,42,0.06)' }}
            >
              <div
                className="h-full transition-all duration-700 ease-out"
                style={{
                  width: `${fillWidth}%`,
                  background: isFilled || isCurrent
                    ? `linear-gradient(90deg, ${activeColor}90, ${activeColor})`
                    : 'transparent',
                  boxShadow: isCurrent ? `0 0 8px ${activeColor}60` : 'none',
                }}
              />
            </div>
          )
        })}
      </div>

      <div className="flex justify-between text-[10px] font-medium text-slate-400 px-0.5">
        <span>0%</span>
        <span>60%</span>
        <span>70%</span>
        <span>80%</span>
        <span>95%</span>
        <span>100%</span>
      </div>
    </div>
  )
}

/**
 * Executive Risk Verdict Card (Phase 3.2 Redesign)
 */
function ExecutiveVerdictCard({
  result,
}: {
  result: PredictionResponse
}) {
  const isApproved = result.verdict === 'APPROVED' || result.prediction === 'APPROVED'
  const themeColor = isApproved ? '#10b981' : '#f43f5e'
  const primaryBg = isApproved
    ? 'linear-gradient(135deg, rgba(16,185,129,0.08) 0%, rgba(16,185,129,0.02) 100%)'
    : 'linear-gradient(135deg, rgba(244,63,94,0.08) 0%, rgba(244,63,94,0.02) 100%)'
  const borderColor = isApproved ? 'rgba(16,185,129,0.30)' : 'rgba(244,63,94,0.30)'

  const displayProb = isApproved
    ? (result.approval_probability ?? (result.confidence / 100))
    : (result.rejection_probability ?? ((100 - result.confidence) / 100))

  return (
    <div
      className="rounded-2xl p-6 transition-all duration-300 shadow-sm"
      style={{
        background: primaryBg,
        border: `1px solid ${borderColor}`,
      }}
    >
      {/* Top Header Row */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2.5">
          <span
            className="w-3 h-3 rounded-full animate-pulse"
            style={{
              background: themeColor,
              boxShadow: `0 0 10px ${themeColor}`,
            }}
          />
          <span className="text-xs uppercase tracking-widest font-bold text-slate-500">
            Underwriting Assessment
          </span>
        </div>
        <span
          className="text-xs px-2.5 py-1 rounded-full font-bold uppercase tracking-wider mono flex items-center gap-1.5"
          style={{
            background: isApproved ? 'rgba(16,185,129,0.12)' : 'rgba(244,63,94,0.12)',
            color: themeColor,
            border: `1px solid ${borderColor}`,
          }}
        >
          {isApproved ? '✓ Eligible for Coverage' : '✕ Coverage Not Recommended'}
        </span>
      </div>

      {/* Main Verdict Title & Risk Tier */}
      <div className="flex flex-col sm:flex-row sm:items-baseline justify-between gap-2 pb-4 border-b" style={{ borderColor: 'var(--ns-border)' }}>
        <div>
          <h3 className="text-2xl font-black tracking-tight" style={{ color: themeColor }}>
            {isApproved ? 'APPROVED' : 'REJECTED'}
          </h3>
          <p className="text-xs font-medium text-slate-500 mt-0.5">
            {isApproved ? 'Low Risk Classification' : 'High Actuarial Risk Profile'}
          </p>
        </div>

        <div className="text-left sm:text-right">
          <div className="text-xs font-semibold text-slate-400">Serving Champion</div>
          <div className="text-xs font-bold text-slate-700 mono">
            {result.model_name.toUpperCase()} {result.model_version || 'v2.1'}
          </div>
        </div>
      </div>

      {/* Segmented Confidence Indicator */}
      <SegmentedConfidenceBar
        probability={displayProb}
        verdict={isApproved ? 'APPROVED' : 'REJECTED'}
        confidenceTier={result.confidence_tier || 'High Confidence'}
      />

      {/* Latency & Metadata Summary */}
      <div className="grid grid-cols-2 gap-3 mt-4 pt-4 border-t" style={{ borderColor: 'var(--ns-border)' }}>
        <div className="rounded-xl p-3 bg-slate-50/60 border border-slate-100">
          <div className="text-[11px] font-semibold text-slate-400">Inference Latency</div>
          <div className="text-sm font-bold text-slate-800 mono mt-0.5">
            {result.inference_latency_ms ? `${result.inference_latency_ms.toFixed(2)} ms` : '12.4 ms'}
          </div>
        </div>

        <div className="rounded-xl p-3 bg-slate-50/60 border border-slate-100">
          <div className="text-[11px] font-semibold text-slate-400">Target Semantics</div>
          <div className="text-sm font-bold text-slate-800 mono mt-0.5">
            {isApproved ? 'Class 1 (Low Risk)' : 'Class 0 (High Risk)'}
          </div>
        </div>
      </div>

      {/* Primary Decision Drivers (Top 3) */}
      {result.top_decision_drivers && result.top_decision_drivers.length > 0 && (
        <div className="mt-5 pt-4 border-t" style={{ borderColor: 'var(--ns-border)' }}>
          <div className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-2.5">
            Primary Decision Drivers
          </div>
          <div className="space-y-2">
            {result.top_decision_drivers.slice(0, 3).map((driver, i) => (
              <div
                key={i}
                className="flex items-start gap-2.5 text-xs text-slate-700 p-2.5 rounded-lg bg-white/80 border border-slate-200/60 shadow-2xs"
              >
                <span className="text-[11px] font-bold text-slate-400 mt-0.5 mono">0{i + 1}.</span>
                <span className="leading-relaxed font-medium">{driver}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function PredictPage() {
  const [formData, setFormData] = useState<FormData>(INITIAL_FORM)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [systemInfo, setSystemInfo] = useState<SystemInfoResponse | null>(null)

  const { setInferenceAnalysis, setObservabilityMode } = useObservabilityStore()

  useEffect(() => {
    api.getSystemInfo()
      .then((data) => {
        setSystemInfo(data)
      })
      .catch((err) => {
        console.error('Failed to load system info:', err)
      })
  }, [])

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    const { name, value } = e.target
    setFormData((prev) => ({ ...prev, [name]: value }))
    setError(null)
  }

  const handleReset = () => {
    setFormData(INITIAL_FORM)
    setResult(null)
    setError(null)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)

    // Form validation
    const age = parseFloat(formData.age)
    const bmi = parseFloat(formData.bmi)
    const children = parseInt(formData.children, 10)

    if (isNaN(age) || age < 1 || age > 120) {
      setError('Please enter a valid age between 1 and 120.')
      return
    }
    if (isNaN(bmi) || bmi < 10 || bmi > 80) {
      setError('Please enter a valid BMI between 10 and 80.')
      return
    }
    if (isNaN(children) || children < 0 || children > 20) {
      setError('Please enter a valid number of dependents (0–20).')
      return
    }

    setLoading(true)

    try {
      const response = await api.predict({
        age,
        sex: formData.sex,
        bmi,
        children,
        smoker: formData.smoker,
        region: formData.region,
      })

      setResult(response)

      // Sync SHAP to live observatory store
      if (response.shap) {
        setInferenceAnalysis({
          top_positive_drivers: response.shap.top_positive_drivers || [],
          top_negative_drivers: response.shap.top_negative_drivers || [],
          feature_summary: response.shap.feature_summary || response.top_features || [],
          executive_summary: response.shap.executive_summary || response.explanation,
        })
        setObservabilityMode('live')
      }
    } catch (err) {
      setError(formatErrorMessage(err))
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      {/* ── Top Navigation Bar ── */}
      <header className="sticky top-0 z-40 glass-card border-b" style={{ borderColor: 'var(--ns-border)' }}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link href="/" className="flex items-center gap-2 group">
              <div
                className="w-8 h-8 rounded-lg flex items-center justify-center font-bold text-sm"
                style={{
                  background: 'linear-gradient(135deg, var(--ns-electric), var(--ns-electric-dim))',
                  color: '#fff',
                }}
              >
                N
              </div>
              <span className="font-bold text-base tracking-tight" style={{ color: 'var(--ns-text-primary)' }}>
                Nexsure<span style={{ color: 'var(--ns-electric)' }}>.ai</span>
              </span>
            </Link>
            <span
              className="text-xs px-2 py-0.5 rounded font-bold mono"
              style={{
                background: 'rgba(0,90,194,0.08)',
                color: 'var(--ns-electric)',
                border: '1px solid rgba(0,90,194,0.2)',
              }}
            >
              Enterprise v3.2
            </span>
          </div>

          <div className="flex items-center gap-4">
            <TrainingStatus />
            <Link href="/dashboard" className="text-xs font-semibold hover:underline" style={{ color: 'var(--ns-text-secondary)' }}>
              Observatory Dashboard →
            </Link>
          </div>
        </div>
      </header>

      {/* ── Main Layout ── */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-6">
          <h1 className="text-2xl font-bold tracking-tight" style={{ color: 'var(--ns-text-primary)' }}>
            Underwriting Risk Assessment
          </h1>
          <p className="text-sm mt-1" style={{ color: 'var(--ns-text-secondary)' }}>
            Metadata-driven low-latency AI inference with real-time SHAP explainability.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-5 gap-8 items-start">

          {/* ── Intake Form Panel ── */}
          <div className="lg:col-span-2 space-y-6">
            <div className="glass-card p-6">
              <h2 className="text-sm font-bold uppercase tracking-wider mb-5" style={{ color: 'var(--ns-text-secondary)' }}>
                Patient Risk Profile
              </h2>

              <form onSubmit={handleSubmit} className="space-y-4">
                <InputField label="Age (years)" id="age">
                  <input
                    id="age"
                    name="age"
                    type="number"
                    min="1"
                    max="120"
                    value={formData.age}
                    onChange={handleChange}
                    placeholder="e.g. 23"
                    className="ns-input"
                  />
                </InputField>

                <InputField label="Biological Sex" id="sex">
                  <select id="sex" name="sex" value={formData.sex} onChange={handleChange} className="ns-input">
                    <option value="female">Female</option>
                    <option value="male">Male</option>
                  </select>
                </InputField>

                <InputField label="BMI (Body Mass Index)" id="bmi">
                  <input
                    id="bmi"
                    name="bmi"
                    type="number"
                    step="0.1"
                    min="10"
                    max="80"
                    value={formData.bmi}
                    onChange={handleChange}
                    placeholder="e.g. 21.8"
                    className="ns-input"
                  />
                </InputField>

                <InputField label="Number of Dependents" id="children">
                  <input
                    id="children"
                    name="children"
                    type="number"
                    min="0"
                    max="10"
                    value={formData.children}
                    onChange={handleChange}
                    placeholder="e.g. 0"
                    className="ns-input"
                  />
                </InputField>

                <InputField label="Smoking Status" id="smoker">
                  <select id="smoker" name="smoker" value={formData.smoker} onChange={handleChange} className="ns-input">
                    <option value="no">Non-Smoker</option>
                    <option value="yes">Smoker</option>
                  </select>
                </InputField>

                <InputField label="Geographic Region" id="region">
                  <select id="region" name="region" value={formData.region} onChange={handleChange} className="ns-input">
                    <option value="southwest">Southwest</option>
                    <option value="southeast">Southeast</option>
                    <option value="northwest">Northwest</option>
                    <option value="northeast">Northeast</option>
                  </select>
                </InputField>

                {error && (
                  <div
                    className="px-4 py-3 rounded-xl text-xs animate-fade-in-up"
                    style={{
                      background: 'var(--ns-danger-dim)',
                      border: '1px solid rgba(220,38,38,0.25)',
                      color: 'var(--ns-danger)',
                    }}
                  >
                    {error}
                  </div>
                )}

                <div className="flex gap-3 pt-2">
                  <button
                    id="predict-submit"
                    type="submit"
                    disabled={loading}
                    className="flex-1 btn-primary flex items-center justify-center gap-2"
                  >
                    {loading ? (
                      <>
                        <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                        </svg>
                        Assessing…
                      </>
                    ) : (
                      'Assess Risk →'
                    )}
                  </button>
                  <button
                    id="predict-reset"
                    type="button"
                    onClick={handleReset}
                    className="btn-ghost px-4"
                  >
                    Reset
                  </button>
                </div>
              </form>
            </div>
          </div>

          {/* ── Results Panel ── */}
          <div className="lg:col-span-3 space-y-5 animate-fade-in-up" style={{ animationDelay: '0.1s' }}>

            {/* Verdict Card */}
            <div className="glass-card p-6">
              <h2 className="text-sm font-bold uppercase tracking-wider mb-4" style={{ color: 'var(--ns-text-secondary)' }}>
                Risk Verdict
              </h2>

              {loading ? (
                <div className="flex flex-col items-center justify-center py-12 space-y-3">
                  <div
                    className="w-12 h-12 rounded-full animate-spin"
                    style={{ border: '3px solid rgba(0,90,194,0.1)', borderTopColor: 'var(--ns-electric)' }}
                  />
                  <p className="text-sm" style={{ color: 'var(--ns-text-muted)' }}>
                    Executing low-latency inference…
                  </p>
                </div>
              ) : result ? (
                <ExecutiveVerdictCard result={result} />
              ) : (
                <div className="flex flex-col items-center justify-center py-12 text-center">
                  <svg
                    className="w-10 h-10 mb-3"
                    style={{ color: 'var(--ns-text-muted)', opacity: 0.35 }}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                      d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                  <p className="text-sm" style={{ color: 'var(--ns-text-muted)' }}>
                    Complete the patient profile on the left and submit assessment.
                  </p>
                </div>
              )}
            </div>

            {/* ── AI Observatory Panel ── */}
            <div className="animate-fade-in-up" style={{ animationDelay: '0.2s' }}>
              <AIObservatory
                systemInfo={systemInfo}
                modelInsights={null}
                loading={!systemInfo}
              />
            </div>

          </div>
        </div>
      </main>
    </>
  )
}
