'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { motion } from 'framer-motion'
import { api, SystemInfoResponse, formatPercent } from '@/lib/api'
import TrainingStatus from '@/components/TrainingStatus'

function NavBar() {
  return (
    <nav
      className="fixed top-0 left-0 right-0 z-40"
      style={{
        background: 'var(--ns-bg-overlay)',
        backdropFilter: 'blur(20px)',
        WebkitBackdropFilter: 'blur(20px)',
        borderBottom: '1px solid var(--ns-border)',
      }}
    >
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        <div className="flex items-center gap-3">
          {/* Logo mark */}
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center"
            style={{ background: 'var(--ns-primary-grad)' }}
          >
            <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
            </svg>
          </div>
          <div>
            <span className="font-bold text-sm" style={{ color: 'var(--ns-text-primary)' }}>
              Nexsure
            </span>
            <span
              className="ml-2 text-xs uppercase tracking-widest font-bold"
              style={{ color: 'var(--ns-electric)', opacity: 0.7 }}
            >
              Intelligence
            </span>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <Link
            href="/dashboard"
            className="text-sm font-medium transition-colors"
            style={{ color: 'var(--ns-text-muted)' }}
          >
            Observatory
          </Link>
          <Link href="/predict" className="btn-primary text-sm">
            Assess Risk →
          </Link>
        </div>
      </div>
    </nav>
  )
}

function HealthDot({ health }: { health: string }) {
  const color =
    health === 'healthy' ? 'var(--ns-success)' :
    health === 'training' ? 'var(--ns-electric)' :
    health === 'degraded' ? 'var(--ns-warning)' : '#64748b'

  return (
    <span className="inline-flex items-center gap-1.5">
      <span
        className="w-1.5 h-1.5 rounded-full animate-status-pulse"
        style={{ background: color }}
      />
      <span className="text-xs uppercase tracking-widest font-bold" style={{ color }}>
        {health}
      </span>
    </span>
  )
}

function StatChip({ label, value }: { label: string; value: string }) {
  return (
    <div
      className="flex flex-col items-center px-5 py-3 rounded-xl animate-count-up"
      style={{
        background: 'rgba(0,90,194,0.04)',
        border: '1px solid rgba(0,90,194,0.10)',
      }}
    >
      <span className="text-2xl font-bold mono gradient-text">{value}</span>
      <span className="text-xs uppercase tracking-wider mt-0.5" style={{ color: 'var(--ns-text-muted)' }}>
        {label}
      </span>
    </div>
  )
}

const FEATURES = [
  {
    icon: (
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={1.5}
        d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
      />
    ),
    title: 'Autonomous ML Lifecycle',
    desc: 'Self-initializing pipeline detects missing artifacts and auto-trains on startup. No manual intervention required.',
  },
  {
    icon: (
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={1.5}
        d="M13 10V3L4 14h7v7l9-11h-7z"
      />
    ),
    title: 'Real-Time Explainability',
    desc: 'SHAP-powered feature attribution adapts dynamically per prediction — not static templates.',
  },
  {
    icon: (
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={1.5}
        d="M15 12a3 3 0 11-6 0 3 3 0 016 0z M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
      />
    ),
    title: 'Full Model Observability',
    desc: 'Hidden Nexsure Observatory exposes accuracy, precision, recall, F1, confusion matrix, and inference latency live.',
  },
  {
    icon: (
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={1.5}
        d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
      />
    ),
    title: 'Artifact Integrity',
    desc: 'Version history, metadata validation, and graceful fallback ensure zero downtime after restarts.',
  },
]

export default function HomePage() {
  const [systemInfo, setSystemInfo] = useState<SystemInfoResponse | null>(null)
  const [infoLoading, setInfoLoading] = useState(true)

  useEffect(() => {
    api.getSystemInfo()
      .then(setSystemInfo)
      .catch(() => setSystemInfo(null))
      .finally(() => setInfoLoading(false))
  }, [])

  return (
    <>
      <NavBar />
      <TrainingStatus />

      <main className="relative pt-24 pb-20">
        {/* Radial glow & ambient shimmer */}
        <div
          className="pointer-events-none absolute inset-0 overflow-hidden"
          aria-hidden="true"
        >
          <motion.div
            initial={{ opacity: 0.6 }}
            animate={{ opacity: [0.6, 0.85, 0.6] }}
            transition={{ duration: 8, repeat: Infinity, ease: 'easeInOut' }}
            style={{
              position: 'absolute',
              top: '-10%',
              left: '50%',
              transform: 'translateX(-50%)',
              width: '100vw',
              height: '80vh',
              background: 'radial-gradient(ellipse at top center, rgba(59, 130, 246, 0.12) 0%, rgba(99, 102, 241, 0.05) 40%, transparent 70%)',
            }}
          />
        </div>

        <div className="max-w-7xl mx-auto px-6 relative z-10">

          {/* ── Hero Section ── */}
          <motion.section 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
            className="text-center max-w-4xl mx-auto mb-16"
          >
            {/* System health badge */}
            <div className="flex items-center justify-center gap-2 mb-6">
              <div
                className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-sm font-medium"
                style={{
                  background: 'rgba(0,90,194,0.08)',
                  border: '1px solid rgba(0,90,194,0.20)',
                }}
              >
                <span style={{ color: 'var(--ns-electric)' }}>Engine Status:</span>
                {infoLoading ? (
                  <span className="text-xs animate-shimmer w-16 h-3 rounded" />
                ) : (
                  <HealthDot health={systemInfo?.model_health ?? 'unavailable'} />
                )}
              </div>
            </div>

            <h1 className="text-5xl md:text-6xl font-bold leading-tight mb-6 tracking-tight">
              <span style={{ color: 'var(--ns-text-primary)' }}>Enterprise AI</span>
              <br />
              <span className="gradient-text">Underwriting Intelligence</span>
            </h1>

            <p className="text-lg max-w-2xl mx-auto mb-8 leading-relaxed" style={{ color: 'var(--ns-text-secondary)' }}>
              Production-grade autonomous ML platform for health insurance risk assessment.
              Powered by SHAP explainability, real-time model observability, and a
              self-initializing training lifecycle.
            </p>

            <div className="flex flex-wrap items-center justify-center gap-4">
              <Link href="/predict" className="btn-primary px-8 py-3 text-base">
                Assess a Claim →
              </Link>
              <Link href="/dashboard" className="btn-ghost px-8 py-3 text-base">
                View Observatory
              </Link>
            </div>
          </motion.section>

          {/* ── Live Stats Strip ── */}
          {!infoLoading && systemInfo && (
            <motion.section
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
              className="glass-card p-6 mb-12"
            >
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <p
                    className="text-xs uppercase tracking-widest font-bold mb-0.5"
                    style={{ color: 'var(--ns-text-muted)' }}
                  >
                    Active Model
                  </p>
                  <p className="text-base font-semibold" style={{ color: 'var(--ns-text-primary)' }}>
                    {systemInfo.model_name?.replace('_', ' ') ?? 'Unavailable'}
                    {systemInfo.model_version && (
                      <span
                        className="ml-2 text-xs px-2 py-0.5 rounded mono"
                        style={{
                          background: 'rgba(0,90,194,0.10)',
                          color: 'var(--ns-electric)',
                        }}
                      >
                        {systemInfo.model_version}
                      </span>
                    )}
                  </p>
                </div>

                <div className="flex flex-wrap gap-4">
                  <StatChip label="Accuracy"  value={formatPercent(systemInfo.accuracy)} />
                  <StatChip label="Precision" value={formatPercent(systemInfo.precision)} />
                  <StatChip label="Recall"    value={formatPercent(systemInfo.recall)} />
                  <StatChip label="F1 Score"  value={formatPercent(systemInfo.f1_score)} />
                </div>

                <div className="text-right">
                  <p className="text-xs" style={{ color: 'var(--ns-text-muted)' }}>Latency</p>
                  <p className="text-lg font-bold mono" style={{ color: 'var(--ns-electric)' }}>
                    {systemInfo.inference_latency_ms != null
                      ? `${systemInfo.inference_latency_ms.toFixed(1)}ms`
                      : '—'}
                  </p>
                </div>
              </div>
            </motion.section>
          )}

          {/* ── Feature Cards ── */}
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
            className="grid md:grid-cols-2 lg:grid-cols-4 gap-5 mb-16"
          >
            {FEATURES.map((feat, i) => (
              <motion.div 
                key={i} 
                className="glass-card p-6 relative overflow-hidden group"
                whileHover={{ y: -4 }}
                transition={{ duration: 0.3, ease: 'easeOut' }}
              >
                {/* Subtle Hover Glow inside card */}
                <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 bg-gradient-to-b from-[rgba(59,130,246,0.03)] to-transparent pointer-events-none" />
                <div
                  className="w-10 h-10 rounded-xl flex items-center justify-center mb-4"
                  style={{ background: 'rgba(0,90,194,0.10)', border: '1px solid rgba(0,90,194,0.20)' }}
                >
                  <svg
                    className="w-5 h-5"
                    style={{ color: 'var(--ns-electric)' }}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    {feat.icon}
                  </svg>
                </div>
                <h3 className="text-sm font-semibold mb-2" style={{ color: 'var(--ns-text-primary)' }}>
                  {feat.title}
                </h3>
                <p className="text-sm leading-relaxed" style={{ color: 'var(--ns-text-muted)' }}>
                  {feat.desc}
                </p>
              </motion.div>
            ))}
          </motion.section>

          {/* ── How It Works ── */}
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.3, ease: [0.16, 1, 0.3, 1] }}
            className="glass-card p-8"
          >
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold mb-2" style={{ color: 'var(--ns-text-primary)' }}>
                Clinical Decision Flow
              </h2>
              <p className="text-sm" style={{ color: 'var(--ns-text-muted)' }}>
                From patient profile to intelligent risk verdict in milliseconds
              </p>
            </div>

            <div className="grid md:grid-cols-4 gap-6">
              {[
                { n: '01', title: 'Input Profile', desc: '6 patient features: age, BMI, sex, smoker, region, dependents' },
                { n: '02', title: 'ML Inference', desc: 'Preprocessed through sklearn pipeline, scored by best-selected model' },
                { n: '03', title: 'Risk Verdict', desc: 'APPROVED / REJECTED with confidence tier: Low · Medium · High' },
                { n: '04', title: 'SHAP Analysis', desc: 'Feature attribution explains exactly which factors drove the decision' },
              ].map(step => (
                <div key={step.n} className="text-center">
                  <div
                    className="w-12 h-12 rounded-xl flex items-center justify-center mx-auto mb-3 font-bold mono text-sm"
                    style={{
                      background: 'linear-gradient(135deg, rgba(0,90,194,0.12), rgba(70,72,212,0.12))',
                      border: '1px solid rgba(0,90,194,0.20)',
                      color: 'var(--ns-electric)',
                    }}
                  >
                    {step.n}
                  </div>
                  <h4 className="text-sm font-semibold mb-1" style={{ color: 'var(--ns-text-primary)' }}>
                    {step.title}
                  </h4>
                  <p className="text-xs leading-relaxed" style={{ color: 'var(--ns-text-muted)' }}>
                    {step.desc}
                  </p>
                </div>
              ))}
            </div>
          </motion.section>
        </div>
      </main>

      {/* Footer */}
      <footer
        className="text-center py-6 text-xs"
        style={{ color: 'var(--ns-text-muted)', borderTop: '1px solid var(--ns-border)' }}
      >
        Nexsure Intelligence Platform · v2.0 · All metrics real-time from backend
      </footer>
    </>
  )
}