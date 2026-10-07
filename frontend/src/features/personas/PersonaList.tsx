import React, { useState } from 'react';
import { Search, Plus, Users, Sparkles } from 'lucide-react';
import type { Persona } from '../../types';
import { PersonaCard } from './PersonaCard';

interface PersonaListProps {
  personas: Persona[];
  loading: boolean;
  onEdit: (persona: Persona) => void;
  onDelete: (id: string) => void;
  onStartChat: (persona: Persona) => void;
  onOpenNewPersona: () => void;
}

export const PersonaList: React.FC<PersonaListProps> = ({
  personas,
  loading,
  onEdit,
  onDelete,
  onStartChat,
  onOpenNewPersona,
}) => {
  const [searchTerm, setSearchTerm] = useState('');

  const filteredPersonas = personas.filter(
    (p) =>
      p.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      (p.description && p.description.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  return (
    <div className="space-y-6">
      {/* Header Bar */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight flex items-center space-x-2.5">
            <Users className="w-6 h-6 text-indigo-400 inline" />
            <span>Target Personas</span>
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Communication profiles and behavioral identities managed in Persona Engine.
          </p>
        </div>

        <button
          onClick={onOpenNewPersona}
          className="flex items-center space-x-2 px-4 py-2.5 text-sm font-semibold rounded-xl text-white bg-indigo-600 hover:bg-indigo-500 shadow-lg shadow-indigo-600/25 transition-all hover:scale-[1.02] active:scale-[0.98]"
        >
          <Plus className="w-4 h-4" />
          <span>Create Persona</span>
        </button>
      </div>

      {/* Search Input */}
      <div className="relative max-w-md">
        <Search className="w-4 h-4 absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400" />
        <input
          type="text"
          placeholder="Search personas by name or traits..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="w-full pl-10 pr-4 py-2 bg-slate-900 border border-slate-800 rounded-xl text-slate-200 placeholder-slate-500 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all"
        />
      </div>

      {/* Personas Grid */}
      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 animate-pulse">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-64 rounded-2xl bg-slate-900/40 border border-slate-800" />
          ))}
        </div>
      ) : filteredPersonas.length === 0 ? (
        <div className="text-center py-16 px-4 rounded-2xl bg-slate-900/30 border border-slate-800/80">
          <div className="w-12 h-12 mx-auto rounded-2xl bg-indigo-500/10 text-indigo-400 flex items-center justify-center mb-4">
            <Sparkles className="w-6 h-6" />
          </div>
          <h3 className="text-lg font-semibold text-white mb-1">No personas found</h3>
          <p className="text-sm text-slate-400 max-w-sm mx-auto mb-5">
            {searchTerm ? 'Try a different search term.' : 'Get started by creating your first conversational persona profile.'}
          </p>
          <button
            onClick={onOpenNewPersona}
            className="inline-flex items-center space-x-2 px-4 py-2 text-sm font-semibold rounded-xl text-white bg-indigo-600 hover:bg-indigo-500 transition-colors"
          >
            <Plus className="w-4 h-4" />
            <span>Create First Persona</span>
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredPersonas.map((persona) => (
            <PersonaCard
              key={persona.id}
              persona={persona}
              onEdit={onEdit}
              onDelete={onDelete}
              onStartChat={onStartChat}
            />
          ))}
        </div>
      )}
    </div>
  );
};
