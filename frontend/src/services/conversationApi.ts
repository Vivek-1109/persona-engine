import { apiClient } from './apiClient';
import type { Conversation, CreateConversationInput, UpdateConversationInput } from '../types';

export const conversationApi = {
  async listConversations(params?: { userId?: string; personaId?: string }): Promise<Conversation[]> {
    const queryParams: Record<string, string> = {};
    if (params?.userId) queryParams.userId = params.userId;
    if (params?.personaId) queryParams.personaId = params.personaId;
    return apiClient.get<Conversation[]>('/conversations', Object.keys(queryParams).length > 0 ? queryParams : undefined);
  },

  async getConversationById(id: string): Promise<Conversation> {
    return apiClient.get<Conversation>(`/conversations/${id}`);
  },

  async createConversation(data: CreateConversationInput): Promise<Conversation> {
    return apiClient.post<Conversation>('/conversations', data);
  },

  async updateConversation(id: string, data: UpdateConversationInput): Promise<Conversation> {
    return apiClient.patch<Conversation>(`/conversations/${id}`, data);
  },

  async deleteConversation(id: string): Promise<void> {
    return apiClient.delete<void>(`/conversations/${id}`);
  },
};
