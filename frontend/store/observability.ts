import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { PredictionResponse } from '@/lib/api'

interface ObservabilityState {
  latestPrediction: PredictionResponse | null
  recentPredictions: PredictionResponse[]
  setLatestPrediction: (prediction: PredictionResponse) => void
  clearTelemetry: () => void
}

export const useObservabilityStore = create<ObservabilityState>()(
  persist(
    (set) => ({
      latestPrediction: null,
      recentPredictions: [],
      setLatestPrediction: (prediction) =>
        set((state) => {
          const updatedRecent = [prediction, ...state.recentPredictions].slice(0, 5)
          return {
            latestPrediction: prediction,
            recentPredictions: updatedRecent,
          }
        }),
      clearTelemetry: () => set({ latestPrediction: null, recentPredictions: [] }),
    }),
    {
      name: 'nexsure-observability',
    }
  )
)
