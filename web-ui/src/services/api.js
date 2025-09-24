import axios from 'axios';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8001/api/v1',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add timestamp to prevent caching
    config.params = {
      ...config.params,
      _t: Date.now(),
    };
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response.data;
  },
  (error) => {
    const errorMessage = error.response?.data?.detail ||
                        error.response?.data?.message ||
                        error.message ||
                        'An unexpected error occurred';

    throw new Error(errorMessage);
  }
);

// API Functions

/**
 * Send a general query to the RAG system
 */
export const sendQuery = async (query, options = {}) => {
  try {
    const response = await api.post('/query', {
      query,
      include_sources: true,
      max_results: options.maxResults || 10,
      ...options
    });

    return response;
  } catch (error) {
    console.error('Query failed:', error);
    throw error;
  }
};

/**
 * Search ATM logs with filters
 */
export const searchLogs = async (query, filters = {}) => {
  try {
    const response = await api.post('/search', {
      query,
      filters,
      limit: filters.limit || 10,
      include_metadata: true
    });

    return response;
  } catch (error) {
    console.error('Search failed:', error);
    throw error;
  }
};

/**
 * Get troubleshooting guidance
 */
export const getTroubleshootingHelp = async (atmId, errorCode, description) => {
  try {
    const response = await api.post('/troubleshoot', {
      atm_id: atmId,
      error_code: errorCode,
      issue_description: description,
      include_resolution_steps: true
    });

    return response;
  } catch (error) {
    console.error('Troubleshooting request failed:', error);
    throw error;
  }
};

/**
 * Request analysis
 */
export const requestAnalysis = async (analysisType, timeRange, focusArea) => {
  try {
    const response = await api.post('/analyze', {
      analysis_type: analysisType,
      time_range: timeRange,
      focus_area: focusArea,
      include_recommendations: true
    });

    return response;
  } catch (error) {
    console.error('Analysis request failed:', error);
    throw error;
  }
};

/**
 * Check system health
 */
export const checkSystemHealth = async () => {
  try {
    const response = await api.get('/health');
    return response;
  } catch (error) {
    console.error('Health check failed:', error);
    // Return a default status if health check fails
    return {
      status: 'unhealthy',
      timestamp: new Date().toISOString(),
      components: {
        api: { status: 'unhealthy', details: 'API connection failed' },
        database: { status: 'unknown', details: 'Unable to check' },
        embeddings: { status: 'unknown', details: 'Unable to check' },
        rag_engine: { status: 'unknown', details: 'Unable to check' }
      },
      uptime_seconds: 0,
      version: 'Unknown'
    };
  }
};

/**
 * Get system statistics
 */
export const getSystemStats = async () => {
  try {
    const response = await api.get('/stats');
    return response;
  } catch (error) {
    console.error('Stats request failed:', error);
    throw error;
  }
};

/**
 * Process batch queries
 */
export const sendBatchQueries = async (queries) => {
  try {
    const response = await api.post('/batch-query', {
      queries: queries.map(q => ({ query: q }))
    });

    return response;
  } catch (error) {
    console.error('Batch query failed:', error);
    throw error;
  }
};

/**
 * Upload log data
 */
export const uploadLogs = async (logData) => {
  try {
    const response = await api.post('/upload-logs', {
      logs: logData,
      process_immediately: true
    });

    return response;
  } catch (error) {
    console.error('Log upload failed:', error);
    throw error;
  }
};

export default api;