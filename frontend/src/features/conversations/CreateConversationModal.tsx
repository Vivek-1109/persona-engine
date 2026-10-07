import React, { useState, useEffect } from 'react';
import { Modal } from '../../components/Modal';
import type { CreateConversationInput, Persona } from '../../types';

interface CreateConversationModalProps {
  isOpen: boolean;
  onClose: () => void;
  personas: Persona[];
  preselectedPersonaId?: string;
  onSubmit: (data: CreateConversationInput) => Promise<void>;
}

export const CreateConversationModal: React.FC<CreateConversationModalProps> = ({
  isOpen,
  onClose,
  personas,
  preselectedPersonaId,
  onSubmit,
}) => {
  const [personaId, setPersonaId] = useState('');
  const [title, setTitle] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (preselectedPersonaId) {
      setPersonaId(preselectedPersonaId);
      const persona = personas.find((p) => p.id === preselectedPersonaId);
      if (persona) {
        setTitle(`Conversation with ${persona.name}`);
      }
    } else if (personas.length > 0) {
      setPersonaId(personas[0].id);
      setTitle(`Conversation with ${personas[0].name}`);
    }
    setError(null);
  }, [preselectedPersonaId, personas, isOpen]);

  const handlePersonaChange = (id: string) => {
    setPersonaId(id);
    const p = personas.find((persona) => persona.id === id);
    if (p) {
      setTitle(`Conversation with ${p.name}`);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!personaId) {
      setError('Please select a target persona');
      return;
    }
    if (!title.trim()) {
      setError('Please provide a conversation title');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      await onSubmit({
        personaId,
        title: title.trim(),
      });
      onClose();
    } catch (err: any) {
      setError(err?.message || 'Failed to start conversation');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Start New Conversation">
      <form onSubmit={handleSubmit} className="space-y-4">
        {error && (
          <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs">
            {error}
          </div>
        )}

        <div>
          <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1.5">
            Select Persona *
          </label>
          <select
            value={personaId}
            onChange={(e) => handlePersonaChange(e.target.value)}
            className="w-full px-3.5 py-2.5 bg-slate-950 border border-slate-800 rounded-xl text-slate-100 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all"
          >
            {personas.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1.5">
            Conversation Title *
          </label>
          <input
            type="text"
            required
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="e.g. Discussing project vision, Sarcastic weekend catchup"
            className="w-full px-3.5 py-2.5 bg-slate-950 border border-slate-800 rounded-xl text-slate-100 placeholder-slate-500 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all"
          />
        </div>

        <div className="flex items-center justify-end space-x-3 pt-3 border-t border-slate-800/80">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 text-sm font-medium text-slate-400 hover:text-white hover:bg-slate-800 rounded-xl transition-colors"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={loading || personas.length === 0}
            className="px-5 py-2 text-sm font-semibold text-white bg-indigo-600 hover:bg-indigo-500 rounded-xl shadow-md shadow-indigo-600/20 disabled:opacity-50 transition-all hover:scale-[1.02] active:scale-[0.98]"
          >
            {loading ? 'Starting...' : 'Start Conversation'}
          </button>
        </div>
      </form>
    </Modal>
  );
};
