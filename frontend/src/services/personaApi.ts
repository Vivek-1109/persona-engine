import { apiClient } from './apiClient';
import type { CreatePersonaInput, Persona, UpdatePersonaInput } from '../types';

export const personaApi = {
  async listPersonas(userId?: string): Promise<Persona[]> {
    const params = userId ? { userId } : undefined;
    return apiClient.get<Persona[]>('/personas', params);
  },

  async getPersonaById(id: string): Promise<Persona> {
    return apiClient.get<Persona>(`/personas/${id}`);
  },

  async createPersona(data: CreatePersonaInput): Promise<Persona> {
    return apiClient.post<Persona>('/personas', data);
  },

  async updatePersona(id: string, data: UpdatePersonaInput): Promise<Persona> {
    return apiClient.patch<Persona>(`/personas/${id}`, data);
  },

  async deletePersona(id: string): Promise<void> {
    return apiClient.delete<void>(`/personas/${id}`);
  },
};
