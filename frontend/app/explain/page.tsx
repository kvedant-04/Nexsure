'use client'

import { useState } from 'react'
import Link from 'next/link'
import { api, ExplanationResponse, formatErrorMessage } from '@/lib/api'

export default function ExplainPage() {
  const [explanation, setExplanation] = useState<ExplanationResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchGlobalExplanation = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await api.getExplanation({})
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
      .sort(([,a], [,b]) => Math.abs(b) - Math.abs(a))
      .slice(0, limit)
  }

  const renderFeatureBar = (feature: string, importance: number) => {
    const absImportance = Math.abs(importance)
    const percentage = (absImportance * 100).toFixed(1)

    return (
      <div key={feature} className="flex items-center py-2">
        <div className="w-64 text-sm text-gray-700 truncate capitalize pr-4 font-medium">
          {feature.replace(/_/g, ' ')}
        </div>
        <div className="flex-1 mx-4">
          <div className="w-full bg-gray-200 rounded-full h-4 relative">
            <div
              className={`h-4 rounded-full transition-all duration-500 ${
                importance > 0 ? 'bg-blue-500' : 'bg-red-500'
              }`}
              style={{
                width: `${absImportance * 100}%`,
              }}
            ></div>
          </div>
        </div>
        <div className={`w-20 text-right text-sm font-medium ${
          importance > 0 ? 'text-blue-600' : 'text-red-600'
        }`}>
          {importance > 0 ? '+' : ''}{percentage}%
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="container mx-auto px-4 max-w-6xl">
        <div className="mb-8">
          <Link href="/" className="text-blue-600 hover:text-blue-800 mb-4 inline-block">
            ← Back to Home
          </Link>
          <h1 className="text-3xl font-bold text-gray-900">AI Explainability Panel</h1>
          <p className="text-gray-600 mt-2">Understand how the AI model makes decisions using SHAP explanations</p>
        </div>

        {/* Controls */}
        <div className="bg-white p-6 rounded-lg shadow-lg mb-8">
          <div className="flex gap-4 items-center flex-wrap">
            <button
              onClick={fetchGlobalExplanation}
              disabled={loading}
              className="bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white px-6 py-2 rounded-lg font-semibold transition-colors flex items-center gap-2"
            >
              {loading && (
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
              )}
              {loading ? 'Loading...' : 'Show Global Feature Importance'}
            </button>
            <Link
              href="/predict"
              className="bg-green-600 hover:bg-green-700 text-white px-6 py-2 rounded-lg font-semibold transition-colors"
            >
              Get Local Explanation
            </Link>
          </div>
          <p className="text-sm text-gray-600 mt-4">
            Global importance shows which features generally influence predictions across all cases.
            Local explanations are available on the prediction page for specific claims.
          </p>
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-8">
            <div className="flex gap-3">
              <svg className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
              <p className="text-red-800 text-sm">{error}</p>
            </div>
          </div>
        )}

        {/* Global Feature Importance */}
        {explanation && (
          <div className="bg-white p-6 rounded-lg shadow-lg mb-8 animate-fade-in">
            <div className="mb-6">
              <h2 className="text-2xl font-semibold text-gray-900 mb-2">
                Global Feature Importance
              </h2>
              <p className="text-gray-600 text-sm">
                This chart shows which features have the most impact on the model's predictions across all claims.
                Higher absolute values indicate stronger influence on the decision.
              </p>
            </div>

            <div className="space-y-1 mb-6">
              {getTopFeatures(explanation.global_feature_importance).map(([feature, importance]) =>
                renderFeatureBar(feature, importance)
              )}
            </div>

            <div className="p-4 bg-gray-50 rounded-lg border border-gray-200 mb-6">
              <h3 className="font-semibold text-gray-900 mb-3">Understanding the Chart</h3>
              <div className="grid md:grid-cols-2 gap-4 text-sm text-gray-600">
                <div className="flex gap-3">
                  <div className="w-3 h-3 bg-blue-500 rounded-full flex-shrink-0 mt-0.5"></div>
                  <div>
                    <p className="font-medium text-blue-600 mb-1">Blue bars (positive impact)</p>
                    <p>Features that increase the likelihood of claim approval when their values are higher.</p>
                  </div>
                </div>
                <div className="flex gap-3">
                  <div className="w-3 h-3 bg-red-500 rounded-full flex-shrink-0 mt-0.5"></div>
                  <div>
                    <p className="font-medium text-red-600 mb-1">Red bars (negative impact)</p>
                    <p>Features that decrease the likelihood of claim approval when their values are higher.</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
              <h3 className="font-semibold text-gray-900 mb-2">SHAP Values Explanation</h3>
              <p className="text-sm text-gray-700 leading-relaxed">
                SHAP (SHapley Additive exPlanations) values show the contribution of each feature to the prediction.
                Positive values push the prediction toward "approved", while negative values push toward "rejected".
                The magnitude indicates the strength of influence. This mathematically grounded approach ensures fair
                and consistent explainability across all predictions.
              </p>
            </div>
          </div>
        )}

        {/* Local Explanation Info */}
        <div className="bg-white p-6 rounded-lg shadow-lg mb-8">
          <h2 className="text-xl font-semibold mb-4">Local Explanations</h2>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-medium text-gray-900 mb-2">What are Local Explanations?</h3>
              <p className="text-sm text-gray-600 mb-4 leading-relaxed">
                Local explanations show exactly why a specific claim was predicted as approved or rejected.
                They provide personalized insights for individual cases, helping you understand the precise
                decision factors for that particular claim.
              </p>
              <ul className="text-sm text-gray-600 space-y-2">
                <li className="flex items-start gap-2">
                  <span className="text-blue-600 font-bold mt-0.5">•</span>
                  <span>Feature contributions for that specific claim</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-600 font-bold mt-0.5">•</span>
                  <span>Which factors pushed toward approval/rejection</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-600 font-bold mt-0.5">•</span>
                  <span>Confidence level of the prediction</span>
                </li>
              </ul>
            </div>
            <div>
              <h3 className="font-medium text-gray-900 mb-2">How to Get Local Explanations</h3>
              <p className="text-sm text-gray-600 mb-4 leading-relaxed">
                Local explanations are automatically generated when you make a prediction on the prediction page.
                Simply enter claim details and submit the form to see the explanation.
              </p>
              <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                <p className="text-sm font-medium text-gray-900 mb-2">Steps:</p>
                <ol className="text-sm text-gray-600 list-decimal list-inside space-y-1">
                  <li>Go to the Prediction page</li>
                  <li>Fill in the claim details</li>
                  <li>Click "Get Prediction"</li>
                  <li>View the explanation below the result</li>
                </ol>
              </div>
            </div>
          </div>
        </div>

        {/* SHAP Information */}
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <h2 className="text-xl font-semibold mb-4">About SHAP (SHapley Additive exPlanations)</h2>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-medium text-gray-900 mb-2">What is SHAP?</h3>
              <p className="text-sm text-gray-600 leading-relaxed">
                SHAP is a game theory approach to explain the output of any machine learning model.
                It connects optimal credit allocation with local explanations using the classic Shapley values
                from game theory and their related extensions. SHAP values provide a unified measure of feature 
                importance that is theoretically sound and practically useful.
              </p>
            </div>
            <div>
              <h3 className="font-medium text-gray-900 mb-2">Why Use SHAP?</h3>
              <ul className="text-sm text-gray-600 space-y-2">
                <li className="flex items-start gap-2">
                  <span className="text-green-600 font-bold mt-0.5">•</span>
                  <span>Model-agnostic explanations work with any ML model</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-600 font-bold mt-0.5">•</span>
                  <span>Theoretically grounded in game theory</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-600 font-bold mt-0.5">•</span>
                  <span>Provides consistent and accurate explanations</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-600 font-bold mt-0.5">•</span>
                  <span>Enables regulatory compliance and transparency</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

  const fetchGlobalExplanation = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch('http://localhost:8000/explain', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          features: {} // Empty features for global importance
        }),
      })

      if (!response.ok) {
        throw new Error('Failed to fetch explanation')
      }

      const data = await response.json()
      setExplanation(data)
      setShowLocal(false)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  const getTopFeatures = (features: Record<string, number>, limit = 15) => {
    return Object.entries(features)
      .sort(([,a], [,b]) => Math.abs(b) - Math.abs(a))
      .slice(0, limit)
  }

  const renderFeatureBar = (feature: string, importance: number, isGlobal = true) => {
    const absImportance = Math.abs(importance)
    const percentage = (absImportance * 100).toFixed(1)

    return (
      <div key={feature} className="flex items-center py-2">
        <div className="w-64 text-sm text-gray-700 truncate capitalize pr-4">
          {feature.replace(/_/g, ' ')}
        </div>
        <div className="flex-1 mx-4">
          <div className="w-full bg-gray-200 rounded-full h-4 relative">
            <div
              className={`h-4 rounded-full ${
                importance > 0 ? 'bg-blue-500' : 'bg-red-500'
              }`}
              style={{
                width: `${absImportance * 100}%`,
                marginLeft: importance < 0 ? `${(1 - absImportance) * 100}%` : '0'
              }}
            ></div>
          </div>
        </div>
        <div className={`w-20 text-right text-sm font-medium ${
          importance > 0 ? 'text-blue-600' : 'text-red-600'
        }`}>
          {importance > 0 ? '+' : ''}{percentage}%
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="container mx-auto px-4 max-w-6xl">
        <div className="mb-8">
          <Link href="/" className="text-blue-600 hover:text-blue-800 mb-4 inline-block">
            ← Back to Home
          </Link>
          <h1 className="text-3xl font-bold text-gray-900">AI Explainability Panel</h1>
          <p className="text-gray-600 mt-2">Understand how the AI model makes decisions using SHAP explanations</p>
        </div>

        {/* Controls */}
        <div className="bg-white p-6 rounded-lg shadow-lg mb-8">
          <div className="flex gap-4 items-center">
            <button
              onClick={fetchGlobalExplanation}
              disabled={loading}
              className="bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white px-6 py-2 rounded-lg font-semibold transition-colors"
            >
              {loading ? 'Loading...' : 'Show Global Feature Importance'}
            </button>
            <Link
              href="/predict"
              className="bg-green-600 hover:bg-green-700 text-white px-6 py-2 rounded-lg font-semibold transition-colors"
            >
              Get Local Explanation
            </Link>
          </div>
          <p className="text-sm text-gray-600 mt-4">
            Global importance shows which features generally influence predictions across all cases.
            Local explanations are available on the prediction page for specific claims.
          </p>
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-8">
            <p className="text-red-800">{error}</p>
          </div>
        )}

        {/* Global Feature Importance */}
        {explanation && (
          <div className="bg-white p-6 rounded-lg shadow-lg">
            <div className="mb-6">
              <h2 className="text-2xl font-semibold text-gray-900 mb-2">
                Global Feature Importance
              </h2>
              <p className="text-gray-600">
                This chart shows which features have the most impact on the model's predictions across all claims.
                Higher absolute values indicate stronger influence on the decision.
              </p>
            </div>

            <div className="space-y-1">
              {getTopFeatures(explanation.global_feature_importance).map(([feature, importance]) =>
                renderFeatureBar(feature, importance, true)
              )}
            </div>

            <div className="mt-8 p-4 bg-gray-50 rounded-lg">
              <h3 className="font-semibold text-gray-900 mb-2">Understanding the Chart</h3>
              <div className="grid md:grid-cols-2 gap-4 text-sm text-gray-600">
                <div>
                  <p className="font-medium text-blue-600 mb-1">Blue bars (positive impact)</p>
                  <p>Features that increase the likelihood of claim approval when their values are higher.</p>
                </div>
                <div>
                  <p className="font-medium text-red-600 mb-1">Red bars (negative impact)</p>
                  <p>Features that decrease the likelihood of claim approval when their values are higher.</p>
                </div>
              </div>
            </div>

            <div className="mt-6 p-4 bg-blue-50 rounded-lg">
              <h3 className="font-semibold text-gray-900 mb-2">SHAP Values Explanation</h3>
              <p className="text-sm text-gray-700">
                SHAP (SHapley Additive exPlanations) values show the contribution of each feature to the prediction.
                Positive values push the prediction toward "approved", while negative values push toward "rejected".
                The magnitude indicates the strength of influence.
              </p>
            </div>
          </div>
        )}

        {/* Local Explanation Info */}
        <div className="mt-8 bg-white p-6 rounded-lg shadow-lg">
          <h2 className="text-xl font-semibold mb-4">Local Explanations</h2>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-medium text-gray-900 mb-2">What are Local Explanations?</h3>
              <p className="text-sm text-gray-600 mb-4">
                Local explanations show exactly why a specific claim was predicted as approved or rejected.
                They provide personalized insights for individual cases.
              </p>
              <ul className="text-sm text-gray-600 space-y-1">
                <li>• Feature contributions for that specific claim</li>
                <li>• Which factors pushed toward approval/rejection</li>
                <li>• Confidence level of the prediction</li>
              </ul>
            </div>
            <div>
              <h3 className="font-medium text-gray-900 mb-2">How to Get Local Explanations</h3>
              <p className="text-sm text-gray-600 mb-4">
                Local explanations are automatically generated when you make a prediction on the prediction page.
              </p>
              <div className="bg-gray-50 p-4 rounded-lg">
                <p className="text-sm font-medium text-gray-900 mb-2">Steps:</p>
                <ol className="text-sm text-gray-600 list-decimal list-inside space-y-1">
                  <li>Go to the Prediction page</li>
                  <li>Fill in the claim details</li>
                  <li>Click "Get Prediction"</li>
                  <li>View the explanation below the result</li>
                </ol>
              </div>
            </div>
          </div>
        </div>

        {/* SHAP Information */}
        <div className="mt-8 bg-white p-6 rounded-lg shadow-lg">
          <h2 className="text-xl font-semibold mb-4">About SHAP (SHapley Additive exPlanations)</h2>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-medium text-gray-900 mb-2">What is SHAP?</h3>
              <p className="text-sm text-gray-600">
                SHAP is a game theory approach to explain the output of any machine learning model.
                It connects optimal credit allocation with local explanations using the classic Shapley values
                from game theory and their related extensions.
              </p>
            </div>
            <div>
              <h3 className="font-medium text-gray-900 mb-2">Why Use SHAP?</h3>
              <ul className="text-sm text-gray-600 space-y-1">
                <li>• Model-agnostic explanations</li>
                <li>• Consistent and accurate</li>
                <li>• Mathematically grounded</li>
                <li>• Works with any ML model</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}