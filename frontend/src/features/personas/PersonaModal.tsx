import React, { useState, useEffect } from 'react';
import { Modal } from '../../components/Modal';
import type { CreatePersonaInput, Persona, UpdatePersonaInput } from '../../types';

interface PersonaModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (data: CreatePersonaInput | UpdatePersonaInput) => Promise<void>;
  initialData?: Persona | null;
}

export const PersonaModal: React.FC<PersonaModalProps> = ({
  isOpen,
  onClose,
  onSubmit,
  initialData,
}) => {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (initialData) {
      setName(initialData.name || '');
      setDescription(initialData.description || '');
    } else {
      setName('');
      setDescription('');
    }
    setError(null);
  }, [initialData, isOpen]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) {
      setError('Persona name is required');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      await onSubmit({
        name: name.trim(),
        description: description.trim() || undefined,
      });
      onClose();
    } catch (err: any) {
      setError(err?.message || 'Failed to save persona');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title={initialData ? 'Edit Persona' : 'Create New Persona'}
    >
      <form onSubmit={handleSubmit} className="space-y-4">
        {error && (
          <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs">
            {error}
          </div>
        )}

        <div>
          <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1.5">
            Persona Name *
          </label>
          <input
            type="text"
            required
            placeholder="e.g. Alex (Startup Founder) or Priya (Sarcastic Best Friend)"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="w-full px-3.5 py-2.5 bg-slate-950 border border-slate-800 rounded-xl text-slate-100 placeholder-slate-500 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all"
          />
        </div>

        <div>
          <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1.5">
            Description & Conversational Traits
          </label>
          <textarea
            rows={4}
            placeholder="Describe target personality: vocabulary, sentence length, humor, slang, topics, Hinglish usage, emotional style..."
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            className="w-full px-3.5 py-2.5 bg-slate-950 border border-slate-800 rounded-xl text-slate-100 placeholder-slate-500 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all resize-none"
          />
          <p className="mt-1 text-[11px] text-slate-500">
            In Phase 1, this description guides the mock orchestrator. In future ML phases, this profile will be learned automatically from conversation data.
          </p>
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
            disabled={loading}
            className="px-5 py-2 text-sm font-semibold text-white bg-indigo-600 hover:bg-indigo-500 rounded-xl shadow-md shadow-indigo-600/20 disabled:opacity-50 transition-all hover:scale-[1.02] active:scale-[0.98]"
          >
            {loading ? 'Saving...' : initialData ? 'Update Persona' : 'Create Persona'}
          </button>
        </div>
      </form>
    </Modal>
  );
};
