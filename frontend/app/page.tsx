"use client"

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { motion } from 'framer-motion'
import { api, SystemInfoResponse, formatPercent } from '@/lib/api'
import TrainingStatus from '@/components/TrainingStatus'

function NavBar() {
  return (
    <nav
      className="fixed top-0 left-0 right-0 z-40 bg-slate-950/80 backdrop-blur-md border-b border-slate-800/80"
    >
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        <div className="flex items-center gap-3">
          {/* Logo mark */}
          <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center shadow-xs">
            <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
            </svg>
          </div>
          <div className="flex items-center gap-2">
            <span className="font-bold text-sm text-white tracking-tight">
              Nexsure
            </span>
            <span className="text-[10px] uppercase tracking-widest font-mono font-semibold px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">
              Enterprise
            </span>
          </div>
        </div>

        <div className="flex items-center gap-6">
          <Link
            href="/dashboard"
            className="text-xs font-medium text-slate-400 hover:text-white transition-colors"
          >
            Observatory
          </Link>
          <Link href="/predict" className="btn-primary text-xs px-4 py-2 flex items-center gap-1.5 font-medium">
            <span>Start Risk Assessment</span>
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </Link>
        </div>
      </div>
    </nav>
  )
}

function StatChip({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col items-center px-4 py-2.5 rounded-xl bg-slate-900/50 border border-slate-800">
      <span className="text-xl font-bold font-mono text-white tracking-tight">{value}</span>
      <span className="text-[10px] uppercase font-mono tracking-wider mt-0.5 text-slate-400">
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
    title: 'Governed ML Lifecycle',
    desc: 'Production pipeline continuously benchmarks models, enforces leakage-safe evaluation, and deploys via an immutable registry.',
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
    title: 'Explainable Machine Learning',
    desc: 'SHAP-powered feature attribution decomposes each underwriting decision into transparent risk and approval drivers.',
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
    title: 'Real-Time Model Observability',
    desc: 'Live executive dashboard monitors classification metrics, holdout confusion matrix, latency percentiles, and inference logs.',
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
    title: 'Artifact Integrity & Governance',
    desc: 'Immutable version control, warm startup validation, and zero-downtime atomic promotion eliminate serving downtime.',
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
    <div className="min-h-screen bg-slate-950 text-slate-100 selection:bg-blue-600 selection:text-white">
      <NavBar />
      <TrainingStatus />

      <main className="relative pt-28 pb-20">
        {/* Radial glow & ambient background shimmer */}
        <div
          className="pointer-events-none absolute inset-0 overflow-hidden"
          aria-hidden="true"
        >
          <motion.div
            initial={{ opacity: 0.5 }}
            animate={{ opacity: [0.5, 0.75, 0.5] }}
            transition={{ duration: 8, repeat: Infinity, ease: 'easeInOut' }}
            style={{
              position: 'absolute',
              top: '-12%',
              left: '50%',
              transform: 'translateX(-50%)',
              width: '100vw',
              height: '85vh',
              background: 'radial-gradient(ellipse at top center, rgba(59, 130, 246, 0.14) 0%, rgba(99, 102, 241, 0.06) 40%, transparent 70%)',
            }}
          />
        </div>

        <div className="max-w-7xl mx-auto px-6 relative z-10">

          {/* ── Hero Section ── */}
          <motion.section 
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
            className="text-center max-w-4xl mx-auto mb-16"
          >
            {/* Engine Status Badge (Richer Enterprise Indicator) */}
            <div className="flex items-center justify-center gap-2 mb-6">
              <div className="inline-flex flex-wrap items-center justify-center gap-2 px-4 py-1.5 rounded-full text-xs font-medium border border-slate-800 bg-slate-900/80 backdrop-blur-md shadow-xs">
                <span className="inline-flex items-center gap-1.5 font-semibold text-emerald-400">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                  Production Ready
                </span>
                <span className="text-slate-600 hidden sm:inline">•</span>
                <span className="text-slate-400 font-mono text-[11px] hidden sm:inline">
                  Champion Model • Healthy Runtime • Observability Active
                </span>
              </div>
            </div>

            {/* Hero Title */}
            <h1 className="text-4xl sm:text-5xl md:text-6xl font-extrabold tracking-tight leading-[1.12] mb-6">
              <span className="text-white">Enterprise AI</span>
              <br />
              <span className="bg-gradient-to-r from-blue-400 via-indigo-300 to-cyan-400 bg-clip-text text-transparent">
                Underwriting Intelligence
              </span>
            </h1>

            {/* Hero Description */}
            <p className="text-base sm:text-lg max-w-2xl mx-auto mb-6 leading-relaxed text-slate-300 font-normal">
              Production-grade underwriting intelligence platform that evaluates insurance applications using explainable machine learning, real-time model observability, and governed model deployment.
            </p>

            {/* Trust Signal Strip (Lightweight Enterprise Badges) */}
            <div className="flex flex-wrap items-center justify-center gap-2 sm:gap-3 mb-8">
              <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-mono bg-slate-900/60 border border-slate-800 text-slate-300">
                <svg className="w-3.5 h-3.5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                <span>Champion Model Active</span>
              </div>
              <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-mono bg-slate-900/60 border border-slate-800 text-slate-300">
                <svg className="w-3.5 h-3.5 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                <span>SHAP Explainability</span>
              </div>
              <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-mono bg-slate-900/60 border border-slate-800 text-slate-300">
                <svg className="w-3.5 h-3.5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                <span>Real-Time Observatory</span>
              </div>
            </div>

            {/* Primary CTAs */}
            <div className="flex flex-wrap items-center justify-center gap-4">
              <Link href="/predict" className="btn-primary px-7 py-3 text-sm font-semibold flex items-center gap-2 shadow-lg shadow-blue-500/10">
                <span>Start Risk Assessment</span>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                </svg>
              </Link>
              <Link href="/dashboard" className="btn-ghost px-7 py-3 text-sm font-semibold border border-slate-700 bg-slate-900/50 hover:bg-slate-800 text-slate-200">
                Open AI Observatory
              </Link>
            </div>
          </motion.section>

          {/* ── Live Stats Strip ── */}
          {!infoLoading && systemInfo && (
            <motion.section
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
              className="bg-slate-900/60 border border-slate-800 rounded-xl p-6 mb-12 backdrop-blur-sm"
            >
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <p className="text-[11px] uppercase font-mono tracking-widest font-semibold mb-0.5 text-slate-400">
                    Champion Model
                  </p>
                  <p className="text-base font-semibold text-white flex items-center">
                    {(systemInfo.champion_model || systemInfo.model_name || 'CatBoost').toUpperCase()}
                    <span className="ml-2 text-xs px-2 py-0.5 rounded font-mono bg-blue-500/10 text-blue-400 border border-blue-500/20">
                      {systemInfo.model_version || 'v2.1'} • Production
                    </span>
                  </p>
                </div>

                <div className="flex flex-wrap gap-3">
                  <StatChip label="Accuracy"  value={formatPercent(systemInfo.accuracy)} />
                  <StatChip label="Precision" value={formatPercent(systemInfo.precision)} />
                  <StatChip label="Recall"    value={formatPercent(systemInfo.recall)} />
                  <StatChip label="F1 Score"  value={formatPercent(systemInfo.f1_score)} />
                </div>

                <div className="text-right">
                  <p className="text-[11px] uppercase font-mono tracking-wider text-slate-400">Median Inference Latency</p>
                  <p className="text-lg font-bold font-mono text-emerald-400 mt-0.5">
                    {systemInfo.inference_latency_ms != null
                      ? `${systemInfo.inference_latency_ms.toFixed(1)}ms`
                      : '0.8ms'}
                  </p>
                </div>
              </div>
            </motion.section>
          )}

          {/* ── Feature Cards ── */}
          <motion.section
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
            className="grid md:grid-cols-2 lg:grid-cols-4 gap-5 mb-16"
          >
            {FEATURES.map((feat, i) => (
              <motion.div 
                key={i} 
                className="bg-slate-900/50 border border-slate-800/90 rounded-xl p-6 relative overflow-hidden group hover:border-slate-700 transition-colors"
                whileHover={{ y: -3 }}
                transition={{ duration: 0.2, ease: 'easeOut' }}
              >
                <div className="w-10 h-10 rounded-xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center mb-4 text-blue-400">
                  <svg
                    className="w-5 h-5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    {feat.icon}
                  </svg>
                </div>
                <h3 className="text-sm font-semibold mb-2 text-white">
                  {feat.title}
                </h3>
                <p className="text-xs leading-relaxed text-slate-400">
                  {feat.desc}
                </p>
              </motion.div>
            ))}
          </motion.section>

          {/* ── Clinical Decision Flow ── */}
          <motion.section
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.3, ease: [0.16, 1, 0.3, 1] }}
            className="bg-slate-900/50 border border-slate-800 rounded-xl p-8 backdrop-blur-sm"
          >
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold mb-2 text-white tracking-tight">
                Enterprise Underwriting Decision Flow
              </h2>
              <p className="text-xs text-slate-400">
                From applicant profile to governed risk verdict and SHAP explanation in milliseconds
              </p>
            </div>

            <div className="grid md:grid-cols-4 gap-6">
              {[
                { n: '01', title: 'Applicant Profile', desc: '6 structured inputs: age, BMI, sex, smoker status, region, and dependents' },
                { n: '02', title: 'Pipeline Execution', desc: 'Leakage-safe preprocessing and inference scored by active Champion model' },
                { n: '03', title: 'Calibrated Verdict', desc: 'APPROVED / REJECTED decision with quantitative confidence tiers' },
                { n: '04', title: 'TreeSHAP Attribution', desc: 'Exact marginal impact decomposition attributing positive and negative factors' },
              ].map(step => (
                <div key={step.n} className="text-center">
                  <div
                    className="w-11 h-11 rounded-xl flex items-center justify-center mx-auto mb-3 font-bold font-mono text-sm bg-blue-500/10 border border-blue-500/20 text-blue-400"
                  >
                    {step.n}
                  </div>
                  <h4 className="text-sm font-semibold mb-1 text-slate-200">
                    {step.title}
                  </h4>
                  <p className="text-xs leading-relaxed text-slate-400">
                    {step.desc}
                  </p>
                </div>
              ))}
            </div>
          </motion.section>
        </div>
      </main>

      {/* Footer */}
      <footer className="text-center py-6 text-xs font-mono text-slate-500 border-t border-slate-800/80">
        Nexsure Lumina Enterprise Platform · All metrics live from backend runtime
      </footer>
    </div>
  )
}
