import React from 'react';
import { Users, MessageSquare, Cpu, Database, Plus, Sparkles, ArrowRight, Bot, ShieldCheck } from 'lucide-react';
import type { Conversation, Persona } from '../../types';
import { PersonaCard } from '../personas/PersonaCard';

interface DashboardViewProps {
  personas: Persona[];
  conversations: Conversation[];
  loading: boolean;
  onSelectConversation: (conversation: Conversation) => void;
  onOpenNewPersona: () => void;
  onOpenNewConversation: () => void;
  onEditPersona: (persona: Persona) => void;
  onDeletePersona: (id: string) => void;
  onNavigateTab: (tab: 'personas' | 'chat') => void;
}

export const DashboardView: React.FC<DashboardViewProps> = ({
  personas,
  conversations,
  loading,
  onSelectConversation,
  onOpenNewPersona,
  onOpenNewConversation,
  onEditPersona,
  onDeletePersona,
  onNavigateTab,
}) => {
  const totalMessages = conversations.reduce((acc, c) => acc + (c.messageCount || 0), 0);

  return (
    <div className="space-y-10">
      {/* Hero / Banner */}
      <div className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-indigo-950/60 via-slate-900/80 to-purple-950/40 border border-slate-800 p-8 shadow-2xl">
        <div className="absolute top-0 right-0 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none -mr-20 -mt-20" />
        <div className="relative z-10 max-w-3xl">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-xs font-semibold mb-4">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Persona Engine Architecture — Phase 1</span>
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight leading-tight">
            Personalized Conversational AI Foundation
          </h1>
          <p className="mt-3 text-base text-slate-300 leading-relaxed">
            A modular system designed to model unique human communication styles, humor, tone, and long-term memory. Phase 1 provides the decoupled frontend, Spring Boot API, Flyway migrations, and orchestrator abstraction.
          </p>

          <div className="flex flex-wrap items-center gap-3 mt-6">
            <button
              onClick={onOpenNewConversation}
              className="flex items-center space-x-2 px-5 py-2.5 rounded-xl text-sm font-semibold text-white bg-indigo-600 hover:bg-indigo-500 shadow-lg shadow-indigo-600/25 transition-all hover:scale-[1.02] active:scale-[0.98]"
            >
              <MessageSquare className="w-4 h-4" />
              <span>Start Conversation</span>
            </button>
            <button
              onClick={onOpenNewPersona}
              className="flex items-center space-x-2 px-5 py-2.5 rounded-xl text-sm font-semibold text-slate-200 bg-slate-900/90 hover:bg-slate-800 border border-slate-700/80 transition-colors"
            >
              <Plus className="w-4 h-4" />
              <span>Create Persona</span>
            </button>
          </div>
        </div>
      </div>

      {/* Architecture & Metrics Strip */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="p-5 rounded-2xl bg-slate-900/50 border border-slate-800 flex items-center space-x-4">
          <div className="p-3 rounded-xl bg-indigo-500/10 text-indigo-400">
            <Users className="w-6 h-6" />
          </div>
          <div>
            <div className="text-2xl font-bold text-white">{personas.length}</div>
            <div className="text-xs text-slate-400 font-medium">Active Personas</div>
          </div>
        </div>

        <div className="p-5 rounded-2xl bg-slate-900/50 border border-slate-800 flex items-center space-x-4">
          <div className="p-3 rounded-xl bg-purple-500/10 text-purple-400">
            <MessageSquare className="w-6 h-6" />
          </div>
          <div>
            <div className="text-2xl font-bold text-white">{conversations.length}</div>
            <div className="text-xs text-slate-400 font-medium">Conversations</div>
          </div>
        </div>

        <div className="p-5 rounded-2xl bg-slate-900/50 border border-slate-800 flex items-center space-x-4">
          <div className="p-3 rounded-xl bg-emerald-500/10 text-emerald-400">
            <Cpu className="w-6 h-6" />
          </div>
          <div>
            <div className="text-2xl font-bold text-white">{totalMessages}</div>
            <div className="text-xs text-slate-400 font-medium">Exchanged Messages</div>
          </div>
        </div>

        <div className="p-5 rounded-2xl bg-slate-900/50 border border-slate-800 flex items-center space-x-4">
          <div className="p-3 rounded-xl bg-cyan-500/10 text-cyan-400">
            <Database className="w-6 h-6" />
          </div>
          <div>
            <div className="text-sm font-bold text-white flex items-center space-x-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span>PostgreSQL</span>
            </div>
            <div className="text-xs text-slate-400 font-medium">Flyway Migrations Active</div>
          </div>
        </div>
      </div>

      {/* Target Personas Section */}
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-white tracking-tight">Configured Personas</h2>
            <p className="text-xs text-slate-400">Target communication identities available for dialogue</p>
          </div>
          <button
            onClick={() => onNavigateTab('personas')}
            className="flex items-center space-x-1 text-xs font-semibold text-indigo-400 hover:text-indigo-300 transition-colors"
          >
            <span>View All ({personas.length})</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>

        {loading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 animate-pulse">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-64 rounded-2xl bg-slate-900/40 border border-slate-800" />
            ))}
          </div>
        ) : personas.length === 0 ? (
          <div className="text-center py-10 rounded-2xl bg-slate-900/30 border border-slate-800">
            <p className="text-sm text-slate-400">No personas found. Create one to begin.</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {personas.slice(0, 3).map((persona) => (
              <PersonaCard
                key={persona.id}
                persona={persona}
                onEdit={onEditPersona}
                onDelete={onDeletePersona}
                onStartChat={() => {
                  onOpenNewConversation();
                }}
              />
            ))}
          </div>
        )}
      </section>

      {/* Recent Conversations Section */}
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-white tracking-tight">Recent Conversations</h2>
            <p className="text-xs text-slate-400">Past chat sessions with persona orchestrator</p>
          </div>
          <button
            onClick={() => onNavigateTab('chat')}
            className="flex items-center space-x-1 text-xs font-semibold text-indigo-400 hover:text-indigo-300 transition-colors"
          >
            <span>View All ({conversations.length})</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>

        {conversations.length === 0 ? (
          <div className="text-center py-10 rounded-2xl bg-slate-900/30 border border-slate-800">
            <p className="text-sm text-slate-400 mb-3">No conversations yet.</p>
            <button
              onClick={onOpenNewConversation}
              className="inline-flex items-center space-x-2 px-4 py-2 text-xs font-semibold rounded-xl text-white bg-indigo-600 hover:bg-indigo-500 transition-colors"
            >
              <Plus className="w-3.5 h-3.5" />
              <span>Start First Conversation</span>
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {conversations.slice(0, 4).map((conv) => (
              <div
                key={conv.id}
                onClick={() => onSelectConversation(conv)}
                className="group flex items-center justify-between p-4 rounded-2xl bg-slate-900/60 border border-slate-800 hover:border-slate-700 hover:bg-slate-900/90 transition-all cursor-pointer shadow-md"
              >
                <div className="flex items-center space-x-3.5 min-w-0">
                  <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-500/20 to-purple-500/20 text-indigo-400 shrink-0">
                    <Bot className="w-5 h-5" />
                  </div>
                  <div className="min-w-0">
                    <h4 className="text-sm font-bold text-white group-hover:text-indigo-300 transition-colors truncate">
                      {conv.title}
                    </h4>
                    <p className="text-xs text-slate-400 mt-0.5">
                      Persona: <span className="text-slate-300">{conv.personaName || 'Persona'}</span> • {conv.messageCount} messages
                    </p>
                  </div>
                </div>
                <ArrowRight className="w-4 h-4 text-slate-500 group-hover:text-white group-hover:translate-x-1 transition-all shrink-0" />
              </div>
            ))}
          </div>
        )}
      </section>

      {/* Architectural Guarantees Footer Info */}
      <div className="p-5 rounded-2xl bg-slate-950/60 border border-slate-800/80 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-slate-400">
        <div className="flex items-center space-x-2">
          <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0" />
          <span>
            Loosely Coupled Architecture: Frontend ↔ Spring Boot REST API ↔ PostgreSQL + Flyway ↔ Orchestrator Interface.
          </span>
        </div>
        <div className="text-slate-500 font-mono text-[11px]">
          ML Engine Plug Point Ready
        </div>
      </div>
    </div>
  );
};
