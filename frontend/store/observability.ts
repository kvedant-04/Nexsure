import { create } from 'zustand'
import {
  api,
  SystemInfoResponse,
  ModelBenchmarkResponse,
  PerformanceTelemetryResponse,
  ModelRegistryResponse,
  PromotionHistoryItem,
  PredictionLog,
  GlobalExplanationResponse,
  PredictionResponse,
} from '@/lib/api'

export interface ShapDriver {
  feature: string
  shap_value: number
  impact_pct?: number
  impact_dir?: string
  impact_level?: string
  insight?: string
}

export interface LiveAnalysisPayload {
  prediction_id?: string
  verdict?: 'APPROVED' | 'REJECTED' | string
  confidence?: number
  executive_summary?: string
  top_positive_drivers: any[]
  top_negative_drivers: any[]
  feature_summary?: any[]
  timestamp?: string
}

export interface ObservabilityStoreState {
  // Live Telemetry States
  systemInfo: SystemInfoResponse | null
  benchmark: ModelBenchmarkResponse | null
  performance: PerformanceTelemetryResponse | null
  registry: ModelRegistryResponse | null
  promotionHistory: PromotionHistoryItem[]
  logs: PredictionLog[]
  globalShap: GlobalExplanationResponse | null
  globalExplanation: GlobalExplanationResponse | null
  liveAnalysis: LiveAnalysisPayload | null

  // Mode and Meta
  mode: 'global' | 'live'
  loading: boolean
  isLoading: boolean
  isRefreshing: boolean
  isLoadingPerformance: boolean
  isLoadingLogs: boolean
  error: string | null
  lastRefresh: string
  lastRefreshed: string

  // Prediction History
  latestPrediction: PredictionResponse | null
  recentPredictions: PredictionResponse[]

  // Actions
  setObservabilityMode: (mode: 'global' | 'live') => void
  setLatestPrediction: (prediction: PredictionResponse) => void
  setInferenceAnalysis: (payload: LiveAnalysisPayload) => void
  clearTelemetry: () => void
  fetchObservabilityData: () => Promise<void>
  fetchPerformanceOnly: () => Promise<void>
  fetchLogsOnly: () => Promise<void>
  fetchRegistryOnly: () => Promise<void>
}

export const useObservabilityStore = create<ObservabilityStoreState>((set, get) => ({
  systemInfo: null,
  benchmark: null,
  performance: null,
  registry: null,
  promotionHistory: [],
  logs: [],
  globalShap: null,
  globalExplanation: null,
  liveAnalysis: null,

  mode: 'global',
  loading: false,
  isLoading: false,
  isRefreshing: false,
  isLoadingPerformance: false,
  isLoadingLogs: false,
  error: null,
  lastRefresh: new Date().toISOString(),
  lastRefreshed: new Date().toISOString(),

  latestPrediction: null,
  recentPredictions: [],

  setObservabilityMode: (mode) => set({ mode }),

  setLatestPrediction: (prediction) => {
    set((state) => {
      const topPos: ShapDriver[] = []
      const topNeg: ShapDriver[] = []

      if (prediction.top_features && Array.isArray(prediction.top_features)) {
        for (const feat of prediction.top_features) {
          const item: ShapDriver = {
            feature: feat.feature,
            shap_value: feat.shap_value,
            impact_pct: feat.impact_pct,
            impact_dir: feat.impact_dir,
            impact_level: feat.impact_level,
            insight: feat.insight,
          }
          if (feat.impact_dir === 'favors_approval') {
            topPos.push(item)
          } else {
            topNeg.push(item)
          }
        }
      }

      const liveAnalysis: LiveAnalysisPayload = {
        verdict: (prediction.verdict || prediction.prediction || 'APPROVED') as any,
        confidence: prediction.confidence || 95.0,
        executive_summary: prediction.explanation,
        top_positive_drivers: topPos,
        top_negative_drivers: topNeg,
        timestamp: new Date().toISOString(),
      }

      return {
        latestPrediction: prediction,
        recentPredictions: [prediction, ...state.recentPredictions].slice(0, 10),
        liveAnalysis,
        mode: 'live',
      }
    })
  },

  setInferenceAnalysis: (payload) => set({ liveAnalysis: payload, mode: 'live' }),

  clearTelemetry: () =>
    set({
      systemInfo: null,
      benchmark: null,
      performance: null,
      registry: null,
      promotionHistory: [],
      logs: [],
      globalShap: null,
      globalExplanation: null,
      liveAnalysis: null,
      error: null,
      loading: false,
      isLoading: false,
      isRefreshing: false,
    }),

  fetchObservabilityData: async () => {
    try {
      set({ loading: true, isLoading: true, isRefreshing: true, error: null })

      const [sysRes, benchRes, perfRes, regRes, promRes, logsRes, shapRes] = await Promise.allSettled([
        api.getSystemInfo(),
        api.getModelBenchmark(),
        api.getPerformance(),
        api.getModelRegistry(),
        api.getPromotionHistory(),
        api.getLogs(),
        api.getExplanation(),
      ])

      const now = new Date().toISOString()

      const updates: Partial<ObservabilityStoreState> = {
        loading: false,
        isLoading: false,
        isRefreshing: false,
        lastRefresh: now,
        lastRefreshed: now,
      }

      if (sysRes.status === 'fulfilled') updates.systemInfo = sysRes.value
      if (benchRes.status === 'fulfilled') updates.benchmark = benchRes.value
      if (perfRes.status === 'fulfilled') updates.performance = perfRes.value
      if (regRes.status === 'fulfilled') updates.registry = regRes.value
      if (promRes.status === 'fulfilled') updates.promotionHistory = promRes.value
      if (logsRes.status === 'fulfilled') updates.logs = logsRes.value
      if (shapRes.status === 'fulfilled') {
        updates.globalShap = shapRes.value
        updates.globalExplanation = shapRes.value
      }

      set(updates)
    } catch (err: any) {
      set({
        error: err?.message || 'Failed to load observability telemetry',
        loading: false,
        isLoading: false,
        isRefreshing: false,
      })
    }
  },

  fetchPerformanceOnly: async () => {
    try {
      set({ isLoadingPerformance: true })
      const perf = await api.getPerformance()
      set({ performance: perf, isLoadingPerformance: false })
    } catch {
      set({ isLoadingPerformance: false })
    }
  },

  fetchLogsOnly: async () => {
    try {
      set({ isLoadingLogs: true })
      const logs = await api.getLogs()
      set({ logs, isLoadingLogs: false })
    } catch {
      set({ isLoadingLogs: false })
    }
  },

  fetchRegistryOnly: async () => {
    try {
      const reg = await api.getModelRegistry()
      set({ registry: reg })
    } catch {
      // silent background refresh
    }
  },
}))
