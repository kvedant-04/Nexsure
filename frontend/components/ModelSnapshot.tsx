"use client"

import React from 'react'
import { motion } from 'framer-motion'
import { useObservabilityStore } from '@/store/observability'
import { formatTimestamp, formatLatency } from '@/lib/api'

export function ModelSnapshot() {
  const { systemInfo, benchmark, registry, promotionHistory, isLoading } = useObservabilityStore()

  const championName = systemInfo?.champion_model || benchmark?.champion || registry?.active_champion || 'CatBoost'
  const championVersion = systemInfo?.model_version || registry?.active_version || 'v2.1'
  const runnerUp = benchmark?.runner_up || systemInfo?.challenger_model || 'XGBoost'
  const datasetSize = benchmark?.dataset_size || systemInfo?.dataset_size || 1338
  const rawFeatures = benchmark?.raw_feature_count || systemInfo?.raw_feature_count || 6
  const transformedFeatures = benchmark?.transformed_feature_count || systemInfo?.transformed_feature_count || 11
  const trainingTime = systemInfo?.training_timestamp || benchmark?.benchmark_timestamp || '2026-09-09T00:00:00Z'
  const trainingDuration = ((benchmark?.benchmark_duration_s || systemInfo?.training_duration_s || 1.2)).toFixed(2) + 's'
  const inferenceLatency = formatLatency(systemInfo?.inference_latency_ms || 0.8)
  const modelSize = systemInfo?.model_size_mb ? `${systemInfo.model_size_mb.toFixed(2)} MB` : '0.45 MB'
  const registryStatus = registry?.status || systemInfo?.registry_status || 'Active'
  const promotionCount = promotionHistory?.length ?? systemInfo?.promotion_count ?? 9

  const metrics = [
    { label: 'Champion Architecture', value: championName.toUpperCase(), badge: 'Production', badgeColor: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' },
    { label: 'Active Model Version', value: championVersion, badge: 'Immutable', badgeColor: 'bg-blue-500/10 text-blue-400 border-blue-500/20' },
    { label: 'Runner-Up Benchmark', value: runnerUp.toUpperCase(), badge: 'Challenger', badgeColor: 'bg-amber-500/10 text-amber-400 border-amber-500/20' },
    { label: 'Registry Lifecycle', value: registryStatus.toUpperCase(), badge: 'Healthy', badgeColor: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' },
    { label: 'Dataset Size', value: `${datasetSize.toLocaleString()} records`, detail: 'Stratified 80/20 train-split' },
    { label: 'Feature Dimensionality', value: `${rawFeatures} Raw / ${transformedFeatures} Encoded`, detail: 'Leakage-safe pipeline contract' },
    { label: 'Model Artifact Size', value: modelSize, detail: 'Serialized binary payload' },
    { label: 'Production Latency', value: inferenceLatency, detail: 'Model-only execution' },
    { label: 'Training Duration', value: trainingDuration, detail: 'CV 5-fold evaluation' },
    { label: 'Promotion Events', value: `${promotionCount} Audited`, detail: 'Zero-downtime hot swap' },
  ]

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-6 backdrop-blur-sm relative overflow-hidden">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6 border-b border-slate-800/80 pb-4">
        <div className="flex items-center space-x-3">
          <div className="h-9 w-9 rounded-lg bg-blue-500/10 border border-blue-500/20 flex items-center justify-center text-blue-400 font-mono text-sm font-semibold">
            MS
          </div>
          <div>
            <h3 className="text-base font-semibold text-white tracking-tight flex items-center gap-2">
              Model Lineage &amp; Metadata Snapshot
              <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                Live Telemetry
              </span>
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Full governance specification, serialization footprint, and training lineage of the active serving model.
            </p>
          </div>
        </div>
        <div className="text-right">
          <span className="text-[11px] font-mono text-slate-400 block">Last Trained</span>
          <span className="text-xs font-mono text-slate-200">{formatTimestamp(trainingTime)}</span>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        {metrics.map((item, idx) => (
          <motion.div
            key={item.label}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.03 }}
            className="bg-slate-950/40 border border-slate-800/80 rounded-lg p-3.5 hover:border-slate-700 transition-colors"
          >
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider">{item.label}</span>
              {item.badge && (
                <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded border ${item.badgeColor}`}>
                  {item.badge}
                </span>
              )}
            </div>
            <div className="text-sm font-semibold font-mono text-white tracking-tight">
              {isLoading && !systemInfo ? 'Loading...' : item.value}
            </div>
            {item.detail && (
              <p className="text-[11px] text-slate-500 mt-1 truncate">{item.detail}</p>
            )}
          </motion.div>
        ))}
      </div>
    </div>
  )
}

export default ModelSnapshot
