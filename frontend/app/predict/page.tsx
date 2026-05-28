'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
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
  age: '',
  sex: 'male',
  bmi: '',
  children: '',
  smoker: 'no',
  region: 'southeast',
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

function ConfidenceGauge({ confidence, tier }: { confidence: number; tier: string }) {
  const color =
    tier.includes('Very High') ? '#0ea5e9' :
    tier.includes('High')      ? '#10b981' :
    tier.includes('Moderate')  ? '#f59e0b' : '#ef4444'

  const pct = Math.min(confidence, 100)

  return (
    <div className="mt-4">
      <div className="flex justify-between items-center mb-1.5">
        <span className="text-xs" style={{ color: 'var(--ns-text-muted)' }}>Confidence</span>
        <div className="flex items-center gap-2">
          <span
            className="text-xs px-2 py-0.5 rounded font-bold uppercase tracking-wider"
            style={{ background: `${color}15`, color, border: `1px solid ${color}30` }}
          >
            {tier}
          </span>
          <span className="text-sm font-bold mono" style={{ color }}>
            {confidence.toFixed(1)}%
          </span>
        </div>
      </div>
      <div
        className="h-2 rounded-full overflow-hidden"
        style={{ background: 'rgba(255,255,255,0.06)' }}
      >
        <div
          className="h-full rounded-full transition-all duration-700 ease-out"
          style={{
            width: `${pct}%`,
            background: `linear-gradient(90deg, ${color}80, ${color})`,
            boxShadow: `0 0 8px ${color}50`,
          }}
        />
      </div>
    </div>
  )
}

function VerdictBadge({ verdict }: { verdict: 'APPROVED' | 'REJECTED' }) {
  const approved = verdict === 'APPROVED'
  const color = approved ? '#10b981' : '#ef4444'
  const bg    = approved ? 'rgba(16,185,129,0.08)' : 'rgba(239,68,68,0.08)'
  const border = approved ? 'rgba(16,185,129,0.25)' : 'rgba(239,68,68,0.25)'

  return (
    <div
      className="flex flex-col items-center py-5 px-6 rounded-xl animate-fade-in-up"
      style={{ background: bg, border: `1px solid ${border}` }}
    >
      <svg className="w-10 h-10 mb-2" fill="none" stroke={color} strokeWidth={1.5} viewBox="0 0 24 24">
        {approved ? (
          <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
        ) : (
          <path strokeLinecap="round" strokeLinejoin="round" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
        )}
      </svg>
      <span className="text-2xl font-bold" style={{ color }}>
        {verdict}
      </span>
      <span className="text-xs mt-1" style={{ color: `${color}80` }}>
        Risk Assessment Complete
      </span>
    </div>
  )
}

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function PredictPage() {
  const [formData, setFormData]   = useState<FormData>(INITIAL_FORM)
  const [result, setResult]       = useState<PredictionResponse | null>(null)
  const [loading, setLoading]     = useState(false)
  const [error, setError]         = useState<string | null>(null)
  const [systemInfo, setSystemInfo] = useState<SystemInfoResponse | null>(null)
  const setLatestPrediction       = useObservabilityStore((state) => state.setLatestPrediction)

  useEffect(() => {
    api.getSystemInfo().then(setSystemInfo).catch(() => null)
  }, [])

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    setFormData(prev => ({ ...prev, [e.target.name]: e.target.value }))
  }

  const buildFeatureChart = (importance: Record<string, number>) => {
    return Object.entries(importance)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 6)
      .map(([name, value]) => ({
        name: name.toUpperCase(),
        value: +(value * 100).toFixed(1),
      }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!formData.age || !formData.bmi) {
      setError('Please provide Age and BMI at minimum.')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const payload = {
        age:      Number(formData.age),
        sex:      formData.sex,
        bmi:      Number(formData.bmi),
        children: Number(formData.children) || 0,
        smoker:   formData.smoker,
        region:   formData.region,
      }

      const response = await api.predict(payload)
      setResult(response)
      setLatestPrediction(response)

    } catch (err) {
      setError(formatErrorMessage(err))
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setFormData(INITIAL_FORM)
    setResult(null)
    setError(null)
  }

  const BAR_COLORS = ['#3b82f6', '#6366f1', '#10b981', '#8b5cf6', '#f59e0b', '#ef4444', '#0ea5e9']

  return (
    <>
      <TrainingStatus />

      {/* Nav */}
      <nav
        className="fixed top-0 left-0 right-0 z-40"
        style={{
          background: 'var(--ns-bg-overlay)',
          backdropFilter: 'blur(20px)',
          borderBottom: '1px solid var(--ns-border)',
        }}
      >
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href="/" className="flex items-center gap-2 text-sm transition-colors" style={{ color: 'var(--ns-text-muted)' }}>
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
              </svg>
              Home
            </Link>
            <span style={{ color: 'var(--ns-text-muted)', opacity: 0.5 }}>·</span>
            <span className="text-sm font-bold" style={{ color: 'var(--ns-text-primary)' }}>
              Risk Assessment
            </span>
          </div>

          {/* Model badge */}
          {systemInfo?.model_name && (
            <div
              className="hidden md:flex items-center gap-2 px-3 py-1 rounded-lg text-xs"
              style={{
                background: 'rgba(0,90,194,0.06)',
                border: '1px solid rgba(0,90,194,0.15)',
              }}
            >
              <span style={{ color: 'var(--ns-text-muted)' }}>Engine:</span>
              <span className="font-medium" style={{ color: 'var(--ns-electric)' }}>
                {systemInfo.model_name.replace('_', ' ')}
              </span>
              {systemInfo.model_version && (
                <span className="mono" style={{ color: 'var(--ns-text-muted)' }}>
                  {systemInfo.model_version}
                </span>
              )}
            </div>
          )}

          <Link href="/dashboard" className="btn-ghost text-xs px-3 py-1.5">
            Observatory
          </Link>
        </div>
      </nav>

      <main className="pt-20 pb-16 max-w-6xl mx-auto px-6">

        {/* Header */}
        <div className="mb-8 animate-fade-in-up">
          <h1 className="text-3xl font-bold mb-1" style={{ color: 'var(--ns-text-primary)' }}>
            Health Insurance Risk Assessment
          </h1>
          <p className="text-sm" style={{ color: 'var(--ns-text-muted)' }}>
            Enter patient profile to receive an AI-driven risk verdict with SHAP feature attribution.
            6 inputs required.
          </p>
        </div>

        <div className="grid lg:grid-cols-5 gap-6">

          {/* ── Input Form ── */}
          <div className="lg:col-span-2 animate-fade-in-up" style={{ animationDelay: '0.05s' }}>
            <div className="glass-card p-6">
              <h2 className="text-sm font-bold uppercase tracking-wider mb-5" style={{ color: 'var(--ns-text-secondary)' }}>
                Patient Profile
              </h2>

              <form id="predict-form" onSubmit={handleSubmit} className="space-y-4">
                <InputField label="Age (years)" id="age">
                  <input
                    id="age"
                    name="age"
                    type="number"
                    min="1"
                    max="120"
                    value={formData.age}
                    onChange={handleChange}
                    placeholder="e.g. 34"
                    className="ns-input"
                  />
                </InputField>

                <InputField label="Biological Sex" id="sex">
                  <select id="sex" name="sex" value={formData.sex} onChange={handleChange} className="ns-input">
                    <option value="male">Male</option>
                    <option value="female">Female</option>
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
                    placeholder="e.g. 27.5"
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
                    placeholder="e.g. 2"
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
                    <option value="southeast">Southeast</option>
                    <option value="southwest">Southwest</option>
                    <option value="northeast">Northeast</option>
                    <option value="northwest">Northwest</option>
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
                    style={{ border: '3px solid rgba(0,210,255,0.1)', borderTopColor: 'var(--ns-electric)' }}
                  />
                  <p className="text-sm" style={{ color: 'var(--ns-text-muted)' }}>
                    Processing assessment…
                  </p>
                </div>
              ) : result ? (
                <div className="space-y-4">
                  <VerdictBadge verdict={result.prediction as 'APPROVED' | 'REJECTED'} />

                  <ConfidenceGauge
                    confidence={result.confidence}
                    tier={result.confidence_tier || (
                      result.confidence >= 95 ? 'Very High Confidence' :
                      result.confidence >= 80 ? 'High Confidence' :
                      result.confidence >= 60 ? 'Moderate Confidence' : 'Low Confidence'
                    )}
                  />

                  {result.inference_latency_ms != null && (
                    <div className="flex items-center gap-2 text-xs" style={{ color: 'var(--ns-text-muted)' }}>
                      <span>⚡ Inference latency:</span>
                      <span className="mono" style={{ color: 'var(--ns-electric)' }}>
                        {result.inference_latency_ms.toFixed(1)}ms
                      </span>
                    </div>
                  )}

                  {result.explanation && (
                    <div
                      className="p-4 rounded-xl text-sm leading-relaxed"
                      style={{
                        background: 'rgba(0,90,194,0.04)',
                        border: '1px solid rgba(0,90,194,0.10)',
                        color: 'var(--ns-text-secondary)',
                      }}
                    >
                      {result.explanation}
                    </div>
                  )}
                </div>
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
                    Complete the patient profile and submit
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
