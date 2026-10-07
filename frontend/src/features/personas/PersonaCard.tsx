import React from 'react';
import { Bot, MessageSquare, Edit2, Trash2, ArrowRight, BrainCircuit } from 'lucide-react';
import type { Persona } from '../../types';
import { StatusBadge } from '../../components/StatusBadge';

interface PersonaCardProps {
  persona: Persona;
  onEdit: (persona: Persona) => void;
  onDelete: (id: string) => void;
  onStartChat: (persona: Persona) => void;
}

export const PersonaCard: React.FC<PersonaCardProps> = ({
  persona,
  onEdit,
  onDelete,
  onStartChat,
}) => {
  return (
    <div className="group relative flex flex-col justify-between p-6 rounded-2xl bg-slate-900/60 border border-slate-800 hover:border-slate-700/80 hover:bg-slate-900/90 transition-all duration-200 shadow-lg hover:shadow-xl hover:shadow-indigo-500/5">
      <div>
        {/* Header */}
        <div className="flex items-start justify-between gap-3 mb-3.5">
          <div className="flex items-center space-x-3">
            <div className="flex items-center justify-center w-11 h-11 rounded-xl bg-gradient-to-br from-indigo-500/20 to-purple-500/20 border border-indigo-500/30 text-indigo-400 group-hover:scale-105 transition-transform">
              <Bot className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-white group-hover:text-indigo-300 transition-colors">
                {persona.name}
              </h3>
              <div className="flex items-center space-x-2 mt-0.5">
                <StatusBadge status={persona.status} />
                <span className="text-xs text-slate-400 font-mono flex items-center">
                  <MessageSquare className="w-3 h-3 mr-1 inline" />
                  {persona.conversationCount} {persona.conversationCount === 1 ? 'chat' : 'chats'}
                </span>
              </div>
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center space-x-1 opacity-80 group-hover:opacity-100 transition-opacity">
            <button
              onClick={() => onEdit(persona)}
              title="Edit Persona"
              className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
            >
              <Edit2 className="w-4 h-4" />
            </button>
            <button
              onClick={() => onDelete(persona.id)}
              title="Delete Persona"
              className="p-1.5 rounded-lg text-slate-400 hover:text-rose-400 hover:bg-rose-500/10 transition-colors"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Description */}
        <p className="text-sm text-slate-300 line-clamp-3 mb-4 leading-relaxed">
          {persona.description || 'No personality description provided.'}
        </p>

        {/* Personality Model Placeholder (Per Section 18) */}
        <div className="p-2.5 rounded-xl bg-slate-950/60 border border-slate-800/80 mb-5 flex items-center space-x-2 text-xs text-slate-400">
          <BrainCircuit className="w-4 h-4 text-purple-400 shrink-0" />
          <div className="truncate">
            <span className="font-semibold text-slate-300">Personality Model: </span>
            <span className="text-purple-400 font-mono text-[11px]">Not configured yet (Phase 1)</span>
          </div>
        </div>
      </div>

      {/* Start Chat Button */}
      <button
        onClick={() => onStartChat(persona)}
        className="w-full flex items-center justify-center space-x-2 py-2.5 px-4 rounded-xl text-sm font-semibold text-slate-200 bg-slate-800 hover:bg-indigo-600 hover:text-white transition-all duration-200 group/btn"
      >
        <span>Open Conversation</span>
        <ArrowRight className="w-4 h-4 group-hover/btn:translate-x-0.5 transition-transform" />
      </button>
    </div>
  );
};
