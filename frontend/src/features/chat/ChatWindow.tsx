import React, { useState, useEffect, useRef } from 'react';
import { ArrowLeft, Bot, Sparkles, AlertCircle, RefreshCw, Trash2, Cpu } from 'lucide-react';
import type { Conversation, Message, Persona } from '../../types';
import { messageApi } from '../../services/messageApi';
import { MessageBubble } from './MessageBubble';
import { MessageInput } from './MessageInput';

interface ChatWindowProps {
  conversation: Conversation;
  persona?: Persona;
  onBack: () => void;
  onDeleteConversation: (id: string) => void;
}

export const ChatWindow: React.FC<ChatWindowProps> = ({
  conversation,
  persona,
  onBack,
  onDeleteConversation,
}) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const fetchMessages = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await messageApi.getMessages(conversation.id);
      setMessages(data);
    } catch (err: any) {
      setError(err?.message || 'Failed to load conversation messages');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMessages();
  }, [conversation.id]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, sending]);

  const handleSendMessage = async (content: string) => {
    setSending(true);
    setError(null);

    // Optimistically create temporary user message in UI
    const tempUserMessage: Message = {
      id: 'temp-' + Date.now(),
      conversationId: conversation.id,
      senderType: 'USER',
      content,
      createdAt: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, tempUserMessage]);

    try {
      const response = await messageApi.sendMessage(conversation.id, { content });
      // Replace optimistic message and append persona message
      setMessages((prev) => {
        const withoutTemp = prev.filter((m) => m.id !== tempUserMessage.id);
        return [...withoutTemp, response.userMessage, response.personaMessage];
      });
    } catch (err: any) {
      setError(err?.message || 'Failed to send message. Please retry.');
      // Remove failed optimistic message
      setMessages((prev) => prev.filter((m) => m.id !== tempUserMessage.id));
    } finally {
      setSending(false);
    }
  };

  const personaDisplayName = conversation.personaName || persona?.name || 'Persona';

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)] max-w-5xl mx-auto rounded-3xl bg-slate-900/40 border border-slate-800 shadow-2xl backdrop-blur-md overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800/80 bg-slate-950/70">
        <div className="flex items-center space-x-3.5">
          <button
            onClick={onBack}
            className="p-2 rounded-xl text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
            title="Back to conversations"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>
          <div className="flex items-center space-x-3">
            <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 to-purple-600 text-white shadow-md">
              <Bot className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h2 className="text-base font-bold text-white tracking-tight">{conversation.title}</h2>
                <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-semibold bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                  <Cpu className="w-3 h-3 mr-1" />
                  Mock Orchestrator
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Talking with <span className="text-indigo-300 font-medium">{personaDisplayName}</span>
              </p>
            </div>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={fetchMessages}
            title="Refresh messages"
            className="p-2 rounded-xl text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
          <button
            onClick={() => onDeleteConversation(conversation.id)}
            title="Delete conversation"
            className="p-2 rounded-xl text-slate-400 hover:text-rose-400 hover:bg-rose-500/10 transition-colors"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Error alert */}
      {error && (
        <div className="mx-6 mt-4 p-3 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
          <button
            onClick={() => setError(null)}
            className="text-xs font-semibold underline hover:text-rose-300"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Message list area */}
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-2">
        {loading ? (
          <div className="flex flex-col items-center justify-center h-full text-slate-500 space-y-3">
            <div className="w-8 h-8 rounded-full border-2 border-indigo-500 border-t-transparent animate-spin" />
            <p className="text-xs">Loading conversation history...</p>
          </div>
        ) : messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center px-4">
            <div className="w-14 h-14 rounded-2xl bg-indigo-500/10 text-indigo-400 flex items-center justify-center mb-4">
              <Sparkles className="w-7 h-7" />
            </div>
            <h3 className="text-base font-semibold text-white mb-1">
              Start chatting with {personaDisplayName}
            </h3>
            <p className="text-xs text-slate-400 max-w-sm mb-6">
              Send your first message. The conversation orchestrator and mock model gateway will reply using this persona profile.
            </p>
            <div className="flex flex-wrap justify-center gap-2 max-w-md">
              {[
                `Hey ${personaDisplayName}, tell me about your background!`,
                'How do you approach challenging problems?',
                'What is your favorite topic to discuss?',
              ].map((prompt, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSendMessage(prompt)}
                  className="text-xs px-3 py-1.5 rounded-xl bg-slate-800/80 hover:bg-slate-800 border border-slate-700/60 text-slate-300 hover:text-white transition-colors"
                >
                  "{prompt}"
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {messages.map((message) => (
              <MessageBubble
                key={message.id}
                message={message}
                personaName={personaDisplayName}
              />
            ))}

            {/* Persona thinking indicator */}
            {sending && (
              <div className="flex items-center space-x-3 my-4">
                <div className="flex items-center justify-center w-8 h-8 rounded-xl bg-gradient-to-tr from-purple-600 to-indigo-600 text-white border border-purple-400/30">
                  <Bot className="w-4 h-4" />
                </div>
                <div className="flex items-center space-x-1.5 px-4 py-3 rounded-2xl bg-slate-900 border border-slate-800 rounded-tl-sm text-slate-400 text-xs">
                  <span className="w-2 h-2 rounded-full bg-indigo-400 animate-bounce" />
                  <span className="w-2 h-2 rounded-full bg-indigo-400 animate-bounce [animation-delay:0.2s]" />
                  <span className="w-2 h-2 rounded-full bg-indigo-400 animate-bounce [animation-delay:0.4s]" />
                  <span className="ml-2 font-mono text-[11px] text-slate-500">Orchestrating response...</span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* Input area */}
      <div className="p-4 border-t border-slate-800/80 bg-slate-950/60">
        <MessageInput onSendMessage={handleSendMessage} loading={sending} />
      </div>
    </div>
  );
};
