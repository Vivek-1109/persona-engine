export type EntityStatus = 'ACTIVE' | 'ARCHIVED' | 'DELETED';
export type SenderType = 'USER' | 'PERSONA' | 'SYSTEM';

export interface User {
  id: string;
  email: string;
  name: string;
  status: EntityStatus;
  createdAt: string;
  updatedAt: string;
}

export interface Persona {
  id: string;
  userId: string;
  name: string;
  description: string | null;
  status: EntityStatus;
  conversationCount: number;
  createdAt: string;
  updatedAt: string;
}

export interface Conversation {
  id: string;
  userId: string;
  personaId: string;
  personaName?: string;
  title: string;
  status: EntityStatus;
  messageCount: number;
  createdAt: string;
  updatedAt: string;
}

export interface Message {
  id: string;
  conversationId: string;
  senderType: SenderType;
  content: string;
  metadata?: Record<string, any>;
  createdAt: string;
}

export interface SendMessageResponse {
  userMessage: Message;
  personaMessage: Message;
}

export interface CreatePersonaInput {
  name: string;
  description?: string;
  userId?: string;
}

export interface UpdatePersonaInput {
  name?: string;
  description?: string;
  status?: EntityStatus;
}

export interface CreateConversationInput {
  personaId: string;
  title: string;
  userId?: string;
}

export interface UpdateConversationInput {
  title?: string;
  status?: EntityStatus;
}

export interface SendMessageInput {
  content: string;
  metadata?: Record<string, any>;
}

export interface ApiError {
  timestamp?: string;
  status: number;
  code: string;
  message: string;
  path?: string;
  details?: string[];
}
