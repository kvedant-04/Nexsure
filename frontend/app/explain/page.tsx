'use client'

import { useState } from 'react'
import Link from 'next/link'
import { api, formatErrorMessage } from '@/lib/api'

interface ExplanationData {
  top_features?: unknown[]
  feature_importance?: Record<string, number>
}

export default function ExplainPage() {
  const [explanation, setExplanation] = useState<ExplanationData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchGlobalExplanation = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await api.getExplanation()
      setExplanation(data)
    } catch (err) {
      const errorMsg = formatErrorMessage(err)
      setError(errorMsg)
      console.error('Explanation fetch error:', err)
    } finally {
      setLoading(false)
    }
  }

  const getTopFeatures = (features: Record<string, number>, limit = 15) => {
    return Object.entries(features)
      .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
      .slice(0, limit)
  }

  const renderFeatureBar = (feature: string, importance: number) => {
    const absImportance = Math.abs(importance)
    const percentage = (absImportance * 100).toFixed(1)

    return (
      <div key={feature} className="flex items-center py-2.5 border-b last:border-0 border-gray-100">
        <div className="w-64 text-sm text-gray-700 truncate capitalize pr-4 font-medium">
          {feature.replace(/_/g, ' ')}
        </div>
        <div className="flex-1 mx-4">
          <div className="w-full bg-gray-100 rounded-full h-3 relative overflow-hidden">
            <div
              className={`h-3 rounded-full transition-all duration-500 ${
                importance >= 0 ? 'bg-blue-600' : 'bg-red-500'
              }`}
              style={{
                width: `${Math.min(absImportance * 100, 100)}%`,
              }}
            />
          </div>
        </div>
        <div className={`w-20 text-right text-sm font-semibold mono ${
          importance >= 0 ? 'text-blue-600' : 'text-red-600'
        }`}>
          {importance >= 0 ? '+' : ''}{percentage}%
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-slate-50 py-10">
      <div className="container mx-auto px-4 max-w-5xl">
        <div className="mb-8">
          <Link href="/" className="text-blue-600 hover:text-blue-800 text-sm font-medium mb-3 inline-flex items-center gap-1">
            ← Back to Nexsure Platform
          </Link>
          <h1 className="text-3xl font-bold text-slate-900 tracking-tight">AI Explainability &amp; SHAP Telemetry</h1>
          <p className="text-slate-600 text-sm mt-1">
            Global feature attribution for the currently active production Champion model.
          </p>
        </div>

        {/* Controls */}
        <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200/80 mb-8">
          <div className="flex gap-4 items-center flex-wrap">
            <button
              onClick={fetchGlobalExplanation}
              disabled={loading}
              className="bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white px-5 py-2.5 rounded-xl font-semibold text-sm transition-all shadow-sm flex items-center gap-2"
            >
              {loading && (
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white" />
              )}
              {loading ? 'Computing Attributions...' : 'Load Global Feature Importance'}
            </button>
            <Link
              href="/predict"
              className="bg-slate-900 hover:bg-slate-800 text-white px-5 py-2.5 rounded-xl font-semibold text-sm transition-all shadow-sm"
            >
              Run Patient Risk Assessment
            </Link>
          </div>
          <p className="text-xs text-slate-500 mt-4">
            Global importance aggregates SHAP Shapley values across the full underwriting baseline.
            Local instance explanations are automatically calculated per live prediction.
          </p>
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 rounded-2xl p-5 mb-8">
            <p className="text-red-800 text-sm">{error}</p>
          </div>
        )}

        {/* Global Feature Importance */}
        {explanation?.feature_importance && (
          <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200/80 mb-8">
            <div className="mb-6">
              <h2 className="text-xl font-bold text-slate-900 mb-1">
                Global Feature Attribution
              </h2>
              <p className="text-slate-500 text-xs">
                Relative influence of each feature across all historical claims data.
              </p>
            </div>

            <div className="space-y-1">
              {getTopFeatures(explanation.feature_importance).map(([feature, importance]) =>
                renderFeatureBar(feature, importance)
              )}
            </div>

            <div className="mt-8 p-4 bg-slate-50 rounded-xl border border-slate-100">
              <h3 className="font-semibold text-slate-800 text-xs uppercase tracking-wider mb-2">Attribution Legend</h3>
              <div className="grid md:grid-cols-2 gap-4 text-xs text-slate-600">
                <div className="flex items-start gap-2">
                  <div className="w-2.5 h-2.5 rounded-full bg-blue-600 mt-0.5 flex-shrink-0" />
                  <p><strong>Positive influence (+)</strong>: Shifts underwriting assessment toward higher approval probability.</p>
                </div>
                <div className="flex items-start gap-2">
                  <div className="w-2.5 h-2.5 rounded-full bg-red-500 mt-0.5 flex-shrink-0" />
                  <p><strong>Negative influence (-)</strong>: Increases risk assessment, pushing outcome toward denial or elevated premium.</p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}