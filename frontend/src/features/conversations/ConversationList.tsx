import React, { useState } from 'react';
import { MessageSquare, Plus, Search, Calendar, ChevronRight, Bot, Trash2 } from 'lucide-react';
import type { Conversation, Persona } from '../../types';

interface ConversationListProps {
  conversations: Conversation[];
  personas: Persona[];
  loading: boolean;
  onSelectConversation: (conversation: Conversation) => void;
  onOpenNewConversation: () => void;
  onDeleteConversation: (id: string) => void;
}

export const ConversationList: React.FC<ConversationListProps> = ({
  conversations,
  personas,
  loading,
  onSelectConversation,
  onOpenNewConversation,
  onDeleteConversation,
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedPersonaFilter, setSelectedPersonaFilter] = useState<string>('all');

  const filteredConversations = conversations.filter((c) => {
    const matchesSearch =
      c.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      (c.personaName && c.personaName.toLowerCase().includes(searchTerm.toLowerCase()));
    const matchesPersona = selectedPersonaFilter === 'all' || c.personaId === selectedPersonaFilter;
    return matchesSearch && matchesPersona;
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight flex items-center space-x-2.5">
            <MessageSquare className="w-6 h-6 text-indigo-400 inline" />
            <span>Conversations</span>
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Active and archived conversation sessions across your personas.
          </p>
        </div>

        <button
          onClick={onOpenNewConversation}
          className="flex items-center space-x-2 px-4 py-2.5 text-sm font-semibold rounded-xl text-white bg-indigo-600 hover:bg-indigo-500 shadow-lg shadow-indigo-600/25 transition-all hover:scale-[1.02] active:scale-[0.98]"
        >
          <Plus className="w-4 h-4" />
          <span>New Conversation</span>
        </button>
      </div>

      {/* Filter and Search Bar */}
      <div className="flex flex-col sm:flex-row gap-3">
        <div className="relative flex-1">
          <Search className="w-4 h-4 absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400" />
          <input
            type="text"
            placeholder="Search conversations by title or persona..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-slate-900 border border-slate-800 rounded-xl text-slate-200 placeholder-slate-500 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all"
          />
        </div>

        <select
          value={selectedPersonaFilter}
          onChange={(e) => setSelectedPersonaFilter(e.target.value)}
          className="px-3.5 py-2 bg-slate-900 border border-slate-800 rounded-xl text-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all sm:w-56"
        >
          <option value="all">All Personas</option>
          {personas.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}
            </option>
          ))}
        </select>
      </div>

      {/* List */}
      {loading ? (
        <div className="space-y-3 animate-pulse">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="h-20 rounded-2xl bg-slate-900/40 border border-slate-800" />
          ))}
        </div>
      ) : filteredConversations.length === 0 ? (
        <div className="text-center py-16 px-4 rounded-2xl bg-slate-900/30 border border-slate-800/80">
          <div className="w-12 h-12 mx-auto rounded-2xl bg-indigo-500/10 text-indigo-400 flex items-center justify-center mb-4">
            <MessageSquare className="w-6 h-6" />
          </div>
          <h3 className="text-lg font-semibold text-white mb-1">No conversations found</h3>
          <p className="text-sm text-slate-400 max-w-sm mx-auto mb-5">
            Start a conversation with one of your personas to test dialogue orchestration.
          </p>
          <button
            onClick={onOpenNewConversation}
            className="inline-flex items-center space-x-2 px-4 py-2 text-sm font-semibold rounded-xl text-white bg-indigo-600 hover:bg-indigo-500 transition-colors"
          >
            <Plus className="w-4 h-4" />
            <span>Start First Conversation</span>
          </button>
        </div>
      ) : (
        <div className="space-y-3">
          {filteredConversations.map((conv) => {
            const formattedDate = new Date(conv.updatedAt || conv.createdAt).toLocaleDateString([], {
              month: 'short',
              day: 'numeric',
              hour: '2-digit',
              minute: '2-digit',
            });

            return (
              <div
                key={conv.id}
                onClick={() => onSelectConversation(conv)}
                className="group flex items-center justify-between p-4.5 rounded-2xl bg-slate-900/60 border border-slate-800 hover:border-slate-700 hover:bg-slate-900/90 transition-all cursor-pointer shadow-md"
              >
                <div className="flex items-center space-x-4 min-w-0">
                  <div className="flex items-center justify-center w-11 h-11 rounded-xl bg-gradient-to-tr from-indigo-500/20 to-purple-500/20 border border-indigo-500/30 text-indigo-400 shrink-0 group-hover:scale-105 transition-transform">
                    <Bot className="w-5 h-5" />
                  </div>
                  <div className="min-w-0">
                    <h3 className="text-sm font-bold text-white group-hover:text-indigo-300 transition-colors truncate">
                      {conv.title}
                    </h3>
                    <div className="flex items-center space-x-3 text-xs text-slate-400 mt-1">
                      <span className="font-medium text-slate-300">
                        {conv.personaName || 'Persona'}
                      </span>
                      <span>•</span>
                      <span className="font-mono">{conv.messageCount} messages</span>
                      <span>•</span>
                      <span className="flex items-center text-slate-500 text-[11px]">
                        <Calendar className="w-3 h-3 mr-1" />
                        {formattedDate}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="flex items-center space-x-2">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteConversation(conv.id);
                    }}
                    title="Delete conversation"
                    className="p-2 rounded-lg text-slate-500 hover:text-rose-400 hover:bg-rose-500/10 transition-colors opacity-0 group-hover:opacity-100"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                  <ChevronRight className="w-5 h-5 text-slate-500 group-hover:text-white group-hover:translate-x-1 transition-all" />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
