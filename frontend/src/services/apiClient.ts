import type { ApiError } from '../types';

const BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

export class ApiException extends Error {
  error: ApiError;

  constructor(error: ApiError) {
    super(error.message || 'An unexpected API error occurred');
    this.name = 'ApiException';
    this.error = error;
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (response.status === 204) {
    return {} as T;
  }

  const contentType = response.headers.get('content-type');
  const isJson = contentType && contentType.includes('application/json');
  const data = isJson ? await response.json() : await response.text();

  if (!response.ok) {
    const error: ApiError = isJson && typeof data === 'object' ? data : {
      status: response.status,
      code: 'HTTP_ERROR',
      message: typeof data === 'string' ? data : response.statusText,
    };
    throw new ApiException(error);
  }

  return data as T;
}

export const apiClient = {
  async get<T>(endpoint: string, params?: Record<string, string>): Promise<T> {
    const url = new URL(`${BASE_URL}${endpoint}`, window.location.origin);
    if (params) {
      Object.entries(params).forEach(([key, val]) => {
        if (val !== undefined && val !== null) {
          url.searchParams.append(key, val);
        }
      });
    }

    const response = await fetch(url.toString(), {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    });

    return handleResponse<T>(response);
  },

  async post<T>(endpoint: string, body?: any): Promise<T> {
    const response = await fetch(`${BASE_URL}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: body ? JSON.stringify(body) : undefined,
    });

    return handleResponse<T>(response);
  },

  async patch<T>(endpoint: string, body: any): Promise<T> {
    const response = await fetch(`${BASE_URL}${endpoint}`, {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(body),
    });

    return handleResponse<T>(response);
  },

  async delete<T>(endpoint: string): Promise<T> {
    const response = await fetch(`${BASE_URL}${endpoint}`, {
      method: 'DELETE',
      headers: {
        'Accept': 'application/json',
      },
    });

    return handleResponse<T>(response);
  },
};
