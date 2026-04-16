/**
 * API Client Utility
 * Handles all communication with the backend API with proper error handling and loading states
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';

// Response types
export interface PredictionResponse {
  model_name: string;
  prediction: string;
  confidence: number;
  label: string;
  explanation: string;
  feature_importance?: Record<string, number>;
}

export interface MetricsResponse {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  confusion_matrix: number[][];
}

export interface ExplanationResponse {
  global_feature_importance: Record<string, number>;
}

export interface TrainResponse {
  models_trained: Record<string, string>;
  best_model: string;
  dataset_rows: number;
  dataset_columns: number;
}

export interface HealthResponse {
  status: string;
}

// Error handling
export class APIError extends Error {
  constructor(
    public statusCode: number,
    public details: string,
    message: string
  ) {
    super(message);
    this.name = 'APIError';
  }
}

/**
 * Make a request to the API with error handling
 */
async function apiRequest<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;

  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    // Handle non-JSON responses
    const contentType = response.headers.get('content-type');
    let data;

    if (contentType && contentType.includes('application/json')) {
      data = await response.json();
    } else {
      data = await response.text();
    }

    if (!response.ok) {
      const errorText = typeof data === 'string' ? data : JSON.stringify(data)
      console.error('Backend Error:', errorText)
      throw new APIError(
        response.status,
        errorText,
        `API Error: ${response.status} - ${response.statusText}`
      );
    }

    return data as T;
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }

    // Network or parsing errors
    if (error instanceof TypeError) {
      throw new APIError(
        0,
        error.message,
        'Network error: Unable to connect to the backend. Make sure the server is running.'
      );
    }

    throw new APIError(
      500,
      String(error),
      'An unexpected error occurred'
    );
  }
}

/**
 * API Methods
 */
export const api = {
  /**
   * Check if API is healthy
   */
  async health(): Promise<HealthResponse> {
    return apiRequest<HealthResponse>('/health');
  },

  /**
   * Train the ML model
   */
  async train(datasetFilename: string = 'insurance_claims.csv'): Promise<TrainResponse> {
    return apiRequest<TrainResponse>('/train', {
      method: 'POST',
      body: JSON.stringify({
        dataset_filename: datasetFilename,
        target_column: 'fraud',
        test_size: 0.2,
        random_state: 42,
      }),
    });
  },

  /**
   * Make a prediction
   */
  async predict(features: Record<string, any>, modelName: string = 'logistic_regression'): Promise<PredictionResponse> {
    return apiRequest<PredictionResponse>('/predict', {
      method: 'POST',
      body: JSON.stringify({
        features,
        model_name: modelName,
      }),
    });
  },

  /**
   * Get model metrics
   */
  async getMetrics(): Promise<MetricsResponse> {
    return apiRequest<MetricsResponse>('/metrics', {
      method: 'GET',
    });
  },

  /**
   * Get explanations
   */
  async getExplanation(features: Record<string, any> = {}): Promise<ExplanationResponse> {
    return apiRequest<ExplanationResponse>('/explain', {
      method: 'POST',
      body: JSON.stringify({ features }),
    });
  },
};

/**
 * Hook for making API calls with loading and error states
 */
export function useApiCall() {
  return async function <T>(
    apiCall: () => Promise<T>
  ): Promise<{ data?: T; error?: string; isLoading: boolean }> {
    try {
      const data = await apiCall();
      return { data, isLoading: false };
    } catch (error) {
      const errorMessage =
        error instanceof APIError
          ? error.message
          : error instanceof Error
          ? error.message
          : 'An unknown error occurred';

      return { error: errorMessage, isLoading: false };
    }
  };
}

/**
 * Format API error messages for user display
 */
export function formatErrorMessage(error: unknown): string {
  if (error instanceof APIError) {
    if (error.statusCode === 0) {
      return 'Network Error: Unable to connect to the server. Please ensure the backend is running.';
    }
    return error.message;
  }

  if (error instanceof Error) {
    return error.message;
  }

  return 'An unexpected error occurred. Please try again.';
}