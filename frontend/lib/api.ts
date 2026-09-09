/**
 * Nexsure API Client â€” v2.0
 * All endpoints use port 8000. No mixed ports, no 8001 references.
 */

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000/api'

// â”€â”€â”€ Response Interfaces â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

export interface SystemInfoResponse {
  model_name:           string | null
  model_version:        string | null
  model_health:         'healthy' | 'training' | 'degraded' | 'unavailable'
  training_timestamp:   string | null
  dataset_size:         number | null
  feature_count:        number | null
  training_duration_s:  number | null
  inference_latency_ms: number | null
  accuracy:             number | null
  precision:            number | null
  recall:               number | null
  f1_score:             number | null
  roc_auc:              number | null
  confusion_matrix:     number[][] | null
  feature_columns:      string[] | null
  trained_models:       string[] | null
  // Registry & Governance Extensions (Phase 2)
  champion_model?:      string | null
  challenger_model?:    string | null
  deployment_timestamp?: string | null
  registry_status?:     string
  promotion_count?:     number
  selection_policy?:    Record<string, string>
  raw_feature_count?:   number
  transformed_feature_count?: number
  model_size_mb?:       number
  uptime_seconds?:      number
  p50_latency_ms?:      number
  p95_latency_ms?:      number
  requests_per_second?: number
  target_definition?:   string
  target_threshold?:    number
  target_source_split?: string
}

export interface ModelRegistryCandidate {
  name:                  string
  display_name:          string
  version:               string
  role:                  'CHAMPION' | 'CHALLENGER' | 'CANDIDATE'
  benchmark_rank:        number
  is_serving:            boolean
  f1_score:              number
  accuracy:              number
  precision:             number
  recall:                number
  roc_auc:               number
  cv_f1_mean:            number
  latency_ms:            number
  end_to_end_latency_ms: number
  size_mb:               number
}

export interface ModelRegistryResponse {
  registry_schema_version: string
  active_champion:          string
  active_version:           string
  deployment_timestamp:     string
  status:                   string
  champion:                 ModelRegistryCandidate
  challenger:               ModelRegistryCandidate | null
  candidates:               Record<string, ModelRegistryCandidate>
  ranking:                  string[]
  selection_policy:         Record<string, string>
}

export interface PromotionHistoryItem {
  event_id:          string
  timestamp:         string
  action:            'INITIAL_DEPLOYMENT' | 'PROMOTION' | 'ROLLBACK' | 'BENCHMARK_PROMOTION'
  from_model:        string | null
  from_version:      string | null
  to_model:          string
  to_version:        string
  actor:             string
  reason:            string
  provenance_status: string
  metrics_snapshot?: Record<string, any>
}


export interface LatencyPercentiles {
  mean_ms: number
  median_ms: number
  p50_ms: number
  p90_ms: number
  p95_ms: number
  p99_ms: number
  min_ms: number
  max_ms: number
  std_dev_ms: number
  sample_count: number
}

export interface ThroughputMetrics {
  total_predictions: number
  successful_predictions: number
  failed_predictions: number
  requests_per_second: number
  requests_per_minute: number
  uptime_seconds: number
  requests_last_minute: number
  cache_hit_rate: number
  cache_miss_rate: number
  average_requests_per_second_since_boot: number
}

export interface StageLatencyStat {
  mean_ms: number
  p50_ms: number
  p95_ms: number
}

export interface LatencyTimelinePoint {
  prediction_id: number
  total_latency_ms: number
  inference_latency_ms: number
  shap_latency_ms: number
  model_name: string
  timestamp: string
}


export interface ShapDiagnostics {
  current_shap_average_ms: number
  current_shap_p50_ms: number
  current_shap_p95_ms: number
  baseline_shap_latency_ms: number
  shap_latency_reduction_pct: number
  current_total_average_ms: number
  baseline_total_latency_ms: number
  total_latency_reduction_pct: number
  shap_cache_hit_rate: number
  shap_cache_hits: number
  shap_cache_misses: number
  sample_count: number
}

export interface PerformanceTelemetryResponse {
  throughput: ThroughputMetrics
  percentiles: LatencyPercentiles
  stage_latencies_ms: Record<string, StageLatencyStat>
  shap_diagnostics?: ShapDiagnostics
  timeline: LatencyTimelinePoint[]
}

export interface CacheLayerStatus {
  name: string
  state: 'READY' | 'MISS' | 'INVALIDATED' | 'REBUILDING'
  cache_hits: number
  cache_misses: number
  hit_rate_pct: number
  initialized_at: string
  last_accessed_at: string
  model_name?: string
  model_version?: string
  artifact_hash?: string
}

export interface CacheStatusResponse {
  readiness_stage: string
  is_warm: boolean
  overall_status: string
  explainer_initialized?: boolean
  background_cached?: boolean
  feature_mapping_cached?: boolean
  model_hash?: string
  average_shap_latency_ms?: number
  shap_latency_reduction_pct?: number
  shap_diagnostics?: ShapDiagnostics
  layers: CacheLayerStatus[]
  timestamp: string
}

export interface TrainingStatusResponse {
  status: 'idle' | 'training' | 'ready' | 'degraded'
  stage:  'idle' | 'initializing' | 'preprocessing' | 'training' |
          'evaluating' | 'selecting' | 'saving'
}

export interface ShapFeature {
  feature: string
  shap_value: number
  impact_pct: number
  impact_dir: string
  impact_level: string
  insight: string
}


export interface BenchmarkModelResult {
  accuracy: number
  precision: number
  recall: number
  f1_score: number
  roc_auc: number
  confusion_matrix: number[][]
  model_only_latency_ms: number
  end_to_end_latency_ms: number
  model_size_mb: number
  cv_f1_mean?: number
  cv_f1_std?: number
}

export interface ModelBenchmarkResponse {
  benchmark_timestamp: string
  benchmark_duration_s: number
  dataset_size: number
  raw_feature_count: number
  transformed_feature_count: number
  champion: string
  runner_up: string
  ranking: string[]
  results: Record<string, BenchmarkModelResult>
  target_definition?: any
  selection_policy?: any
}

export interface GlobalExplanationResponse {
  model_name: string
  model_version?: string
  feature_importance: Record<string, number>
  top_features: Array<{ feature: string; importance: number; impact_pct?: number }>
  executive_summary?: string
  timestamp: string
}

export interface PredictionResponse {
  model_name:           string
  model_version?:       string
  verdict:              'APPROVED' | 'REJECTED'
  prediction:           'APPROVED' | 'REJECTED'
  confidence:           number
  confidence_tier:      string
  approval_probability: number
  rejection_probability: number
  predicted_class?:     number
  positive_class?:      number
  negative_class?:      number
  positive_class_name?: string
  negative_class_name?: string
  label:                string
  explanation:          string
  top_decision_drivers?: string[]
  feature_importance?:  Record<string, number>
  top_features?:        Array<{ feature: string; shap_value: number; importance?: number; impact?: string; impact_pct?: number; impact_dir?: string; impact_level?: string; insight?: string }>
  inference_latency_ms?: number
  shap?: {
    top_positive_drivers: ShapFeature[]
    top_negative_drivers: ShapFeature[]
    top_decision_drivers?: string[]
    global_importance?: any[]
    feature_summary?: any[]
    executive_summary?: string
  }
}

export interface MetricsResponse {
  evaluations: Record<string, ModelMetrics>
}

export interface ModelMetrics {
  accuracy:             number
  precision:            number
  recall:               number
  f1_score:             number
  confusion_matrix:     number[][]
  inference_latency_ms: number
  roc_auc?:             number
  average_method?:      string
}

export interface ModelInsightsResponse {
  top_features:      Array<{ feature: string; importance: number; shap_value?: number }>
  feature_importance: Record<string, number>
  all_model_metrics?: Record<string, ModelMetrics>
}

export interface VersionHistoryEntry {
  version:             string
  timestamp:           string
  model:               string
  accuracy:            number
  f1_score:            number
  dataset_size:        number
  training_duration_s: number
}

export interface PredictionLog {
  id?:                 number
  timestamp:           string
  features?:           Record<string, any>
  age?:                number
  smoker?:             string
  prediction?:         'APPROVED' | 'REJECTED'
  verdict?:            'APPROVED' | 'REJECTED'
  result?:             'APPROVED' | 'REJECTED'
  confidence:          number
  confidence_tier?:    string
  approval_probability?: number
  rejection_probability?: number
  latency_ms?:         number
  inference_latency_ms?: number
  model_name?:         string
  model_used?:         string
}

export interface HealthResponse {
  status:       string
  model_health: string
  engine:       string
}

export interface TrainResponse {
  models_trained:       Record<string, string>
  best_model:           string
  dataset_rows:         number
  dataset_columns:      number
  training_duration_s?: number
  inference_latency_ms?: number
}

// â”€â”€â”€ Error Class â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

export class APIError extends Error {
  constructor(
    public statusCode: number,
    public details: string,
    message: string
  ) {
    super(message)
    this.name = 'APIError'
  }
}

// â”€â”€â”€ Core Request Helper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

async function apiRequest<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`
  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    })

    const contentType = response.headers.get('content-type')
    let data: unknown

    if (contentType?.includes('application/json')) {
      data = await response.json()
    } else {
      data = await response.text()
    }

    if (!response.ok) {
      const msg =
        typeof data === 'object' && data !== null && 'detail' in data
          ? String((data as Record<string, unknown>).detail)
          : JSON.stringify(data)
      throw new APIError(
        response.status,
        msg,
        `API ${response.status}: ${response.statusText}`
      )
    }

    return data as T
  } catch (err) {
    if (err instanceof APIError) throw err
    if (err instanceof TypeError) {
      throw new APIError(
        0,
        err.message,
        'Network error: Unable to reach Nexsure backend on port 8000.'
      )
    }
    throw new APIError(500, String(err), 'Unexpected error')
  }
}

// â”€â”€â”€ API Methods â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

export const api = {
  /** Real-time latency percentiles, throughput & timeline (Phase 3) */
  getPerformance: () =>
    apiRequest<PerformanceTelemetryResponse>('/performance'),

  /** Layered cache readiness and hit rate metrics (Phase 3) */
  getCacheStatus: () =>
    apiRequest<CacheStatusResponse>('/cache-status'),

  /** Liveness probe */
  health: () =>
    apiRequest<HealthResponse>('/health'),

  /** Full model metadata + real metrics */
  getSystemInfo: () =>
    apiRequest<SystemInfoResponse>('/system-info'),

  /** Polling endpoint for training stage */
  getTrainingStatus: () =>
    apiRequest<TrainingStatusResponse>('/training-status'),

  /** Per-model evaluation metrics */
  getMetrics: () =>
    apiRequest<MetricsResponse>('/metrics'),

  /** Feature importance + model comparison */
  getModelInsights: () =>
    apiRequest<ModelInsightsResponse>('/model-insights'),

  /** Retraining audit trail */
  getVersionHistory: () =>
    apiRequest<VersionHistoryEntry[]>('/version-history'),

  /** Recent prediction logs */
  getLogs: () =>
    apiRequest<PredictionLog[]>('/logs'),

  /** Model Registry (Champion/Challenger metadata) */
  getModelBenchmark: () =>
    apiRequest<ModelBenchmarkResponse>('/model-benchmark'),

  getModelRegistry: () =>
    apiRequest<ModelRegistryResponse>('/model-registry'),

  /** Model Promotion & Rollback Audit History */
  getPromotionHistory: () =>
    apiRequest<PromotionHistoryItem[]>('/promotion-history'),

  /** SHAP global explanations */
  getExplanation: () =>
    apiRequest<GlobalExplanationResponse>('/explain'),

  /** Risk assessment prediction */
  predict: (features: Record<string, unknown>, modelName = 'best_model') =>
    apiRequest<PredictionResponse>('/predict', {
      method: 'POST',
      body: JSON.stringify({ features, model_name: modelName }),
    }),

  /** Emergency manual retrain */
  train: (datasetFilename = 'insurance3r2.csv') =>
    apiRequest<TrainResponse>('/train', {
      method: 'POST',
      body: JSON.stringify({
        dataset_filename: datasetFilename,
        target_column: 'insuranceclaim',
        test_size: 0.2,
        random_state: 42,
      }),
    }),
}

// â”€â”€â”€ Utility â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

export function formatErrorMessage(error: unknown): string {
  if (error instanceof APIError) {
    if (error.statusCode === 0) {
      return 'Connection failed â€” ensure the Nexsure backend is running on port 8000.'
    }
    return error.details || error.message
  }
  if (error instanceof Error) return error.message
  return 'An unexpected error occurred.'
}

export function formatPercent(value: number | null | undefined, decimals = 1): string {
  if (value == null || Number.isNaN(value)) return 'Unavailable'
  return `${(value * 100).toFixed(decimals)}%`
}

export function formatLatency(ms: number | null | undefined): string {
  if (ms == null) return 'Unavailable'
  return `${ms.toFixed(1)}ms`
}

export function formatTimestamp(iso: string | null | undefined): string {
  if (!iso) return 'Unavailable'
  try {
    return new Date(iso).toLocaleString('en-US', {
      year: 'numeric', month: 'short', day: 'numeric',
      hour: '2-digit', minute: '2-digit', timeZoneName: 'short',
    })
  } catch {
    return iso
  }
}