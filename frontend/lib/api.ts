/**
 * Nexsure API Client — v2.0
 * All endpoints use port 8000. No mixed ports, no 8001 references.
 */

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000/api'

// ─── Response Interfaces ──────────────────────────────────────────────────────

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

export interface PredictionResponse {
  model_name:           string
  prediction:           'APPROVED' | 'REJECTED'
  confidence:           number
  confidence_tier:      string
  label:                string
  explanation:          string
  feature_importance?:  Record<string, number>
  top_features?:        Array<{ feature: string; shap_value: number; importance?: number; impact?: string }>
  inference_latency_ms?: number
  shap?: {
    top_positive_drivers: ShapFeature[]
    top_negative_drivers: ShapFeature[]
    global_importance: any[]
    feature_summary: any[]
    executive_summary: string
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
  id:                  number
  timestamp:           string
  age:                 number
  smoker:              string
  result:              'APPROVED' | 'REJECTED'
  confidence:          number
  confidence_tier:     string
  model_used:          string
  inference_latency_ms: number
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

// ─── Error Class ──────────────────────────────────────────────────────────────

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

// ─── Core Request Helper ──────────────────────────────────────────────────────

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

// ─── API Methods ──────────────────────────────────────────────────────────────

export const api = {
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

  /** SHAP global explanations */
  getExplanation: () =>
    apiRequest<{ top_features: unknown[]; feature_importance: Record<string, number> }>('/explain'),

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

// ─── Utility ──────────────────────────────────────────────────────────────────

export function formatErrorMessage(error: unknown): string {
  if (error instanceof APIError) {
    if (error.statusCode === 0) {
      return 'Connection failed — ensure the Nexsure backend is running on port 8000.'
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