import { apiClient } from './apiClient';
import type { Message, SendMessageInput, SendMessageResponse } from '../types';

export const messageApi = {
  async getMessages(conversationId: string): Promise<Message[]> {
    return apiClient.get<Message[]>(`/conversations/${conversationId}/messages`);
  },

  async sendMessage(conversationId: string, data: SendMessageInput): Promise<SendMessageResponse> {
    return apiClient.post<SendMessageResponse>(`/conversations/${conversationId}/messages`, data);
  },
};
