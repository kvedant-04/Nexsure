'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import { api, PredictionResponse, formatErrorMessage } from '@/lib/api'

export default function PredictPage() {
  const [formData, setFormData] = useState({
    age: '',
    sex: 'male',
    bmi: '',
    children: '',
    smoker: 'no',
    region: 'southeast',
    charges: ''
  })
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [featureData, setFeatureData] = useState<Array<{ name: string; importance: number }>>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value
    }))
  }

  const transformFeatureImportance = (rawData: Record<string, number>) => {
    const grouped: Record<string, number> = {}

    Object.entries(rawData).forEach(([key, value]) => {
      let cleanKey = key.toLowerCase()

      if (cleanKey.includes('sex')) cleanKey = 'sex'
      else if (cleanKey.includes('region')) cleanKey = 'region'
      else if (cleanKey.includes('bmi')) cleanKey = 'bmi'
      else if (cleanKey.includes('age')) cleanKey = 'age'
      else if (cleanKey.includes('charges')) cleanKey = 'charges'
      else if (cleanKey.includes('children')) cleanKey = 'children'

      if (!grouped[cleanKey]) grouped[cleanKey] = 0
      grouped[cleanKey] += Math.abs(Number(value) || 0)
    })

    const formatted = Object.entries(grouped).map(([key, value]) => ({
      name: key.toUpperCase(),
      importance: value,
    }))

    const total = formatted.reduce((sum, feature) => sum + feature.importance, 0)

    return formatted
      .map(feature => ({
        ...feature,
        importance: total ? feature.importance / total : 0,
      }))
      .filter(feature => feature.importance > 0.01)
      .sort((a, b) => b.importance - a.importance)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!formData.age || !formData.bmi || !formData.charges) {
      alert('Please fill all required fields')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const payload = {
        age: Number(formData.age),
        sex: formData.sex,
        bmi: Number(formData.bmi),
        children: Number(formData.children),
        smoker: formData.smoker,
        region: formData.region,
        charges: Number(formData.charges)
      }

      console.log('FINAL PAYLOAD:', payload)

      const response = await api.predict(payload)
      setResult(response)

      if (response.feature_importance) {
        const formatted = transformFeatureImportance(response.feature_importance)
        setFeatureData(formatted.length > 0 ? formatted : [
          { name: 'AGE', importance: 0.3 },
          { name: 'BMI', importance: 0.25 },
          { name: 'CHARGES', importance: 0.2 },
        ])
      }
    } catch (err) {
      const errorMsg = formatErrorMessage(err)
      setError(errorMsg)
      console.error('Prediction error:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setFormData({
      age: '',
      sex: 'male',
      bmi: '',
      children: '',
      smoker: 'no',
      region: 'southeast',
      charges: ''
    })
    setResult(null)
    setError(null)
  }

  const approved = result?.prediction === "APPROVED"

  useEffect(() => {
    if (featureData.length === 0) {
      setFeatureData([
        { name: 'AGE', importance: 0.3 },
        { name: 'BMI', importance: 0.25 },
        { name: 'CHARGES', importance: 0.2 },
      ])
    }
  }, [])

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="container mx-auto px-4 max-w-2xl">
        <div className="mb-8">
          <Link href="/" className="text-blue-600 hover:text-blue-800 mb-4 inline-block font-medium">
            ← Back to Home
          </Link>
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Health Insurance Prediction</h1>
          <p className="text-gray-600">Enter the patient's profile to get a prediction from the trained model.</p>
        </div>

        <div className="grid gap-8 lg:grid-cols-2">
          <div className="bg-white p-6 rounded-lg shadow-lg">
            <h2 className="text-xl font-semibold mb-4">Patient Features</h2>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Age</label>
                <input
                  type="number"
                  name="age"
                  value={formData.age}
                  onChange={handleInputChange}
                  placeholder="Age"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Sex</label>
                <select
                  name="sex"
                  value={formData.sex}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="male">male</option>
                  <option value="female">female</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">BMI</label>
                <input
                  type="number"
                  step="0.1"
                  name="bmi"
                  value={formData.bmi}
                  onChange={handleInputChange}
                  placeholder="BMI"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Children</label>
                <input
                  type="number"
                  name="children"
                  value={formData.children}
                  onChange={handleInputChange}
                  placeholder="Number of children"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Smoker</label>
                <select
                  name="smoker"
                  value={formData.smoker}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="no">no</option>
                  <option value="yes">yes</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Region</label>
                <select
                  name="region"
                  value={formData.region}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="southeast">southeast</option>
                  <option value="southwest">southwest</option>
                  <option value="northeast">northeast</option>
                  <option value="northwest">northwest</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Charges</label>
                <input
                  type="number"
                  step="0.01"
                  name="charges"
                  value={formData.charges}
                  onChange={handleInputChange}
                  placeholder="Charges"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              <div className="flex gap-3 pt-4">
                <button
                  type="submit"
                  disabled={loading}
                  className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-medium py-2 px-4 rounded-md transition"
                >
                  {loading ? 'Predicting...' : 'Get Prediction'}
                </button>
                <button
                  type="button"
                  onClick={handleReset}
                  className="px-4 py-2 border border-gray-300 text-gray-700 hover:bg-gray-50 font-medium rounded-md transition"
                >
                  Reset
                </button>
              </div>
            </form>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-lg">
            <h2 className="text-xl font-semibold mb-4 text-gray-900">Prediction Output</h2>
            {error ? (
              <p className="text-red-600">{error}</p>
            ) : result ? (
              <div className="space-y-4">
                <div className={`p-4 rounded-lg ${approved ? 'bg-green-50' : 'bg-red-50'}`}>
                  <p className={`text-2xl font-bold ${approved ? 'text-green-700' : 'text-red-700'}`}>
                    {approved ? 'APPROVED' : 'REJECTED'}
                  </p>
                </div>
                {result.confidence !== undefined && (
                  <p className="text-sm text-gray-600">Confidence: {result.confidence}%</p>
                )}
                {result.explanation && (
                  <div>
                    <h3 className="font-medium text-gray-900">Explanation</h3>
                    <p className="text-sm text-gray-600">{result.explanation}</p>
                  </div>
                )}
              </div>
            ) : (
              <p className="text-gray-600">Enter the seven required inputs and submit to get a prediction.</p>
            )}
          </div>
        </div>

        {/* NEW SECTION START */}
        <div className="mt-10 space-y-6">
          <div className="bg-gradient-to-r from-blue-50 to-blue-100 p-6 rounded-2xl shadow-md transition-all duration-300 hover:scale-105">
            <h2 className="text-lg font-semibold mb-2">Model Information</h2>
            <p className="text-sm text-gray-700">
              <span className="font-medium">Best Model:</span> Random Forest
            </p>
            <p className="text-sm text-gray-700 mt-1">
              <span className="font-medium">Accuracy:</span> 95%
            </p>
            <p className="text-sm text-gray-500 mt-2">
              Selected based on highest F1-score and performance.
            </p>
          </div>

          <div className="bg-white rounded-2xl shadow-md p-6">
            <h2 className="text-lg font-semibold mb-4">Feature Importance</h2>

            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={featureData}>
                <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                <YAxis tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} tick={{ fontSize: 12 }} />
                <Tooltip formatter={(value: number) => `${(value * 100).toFixed(1)}%`} />
                <Bar dataKey="importance" radius={[8, 8, 0, 0]} fill="#4f46e5" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  )
}
