import { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { DashboardView } from './features/dashboard/DashboardView';
import { PersonaList } from './features/personas/PersonaList';
import { PersonaModal } from './features/personas/PersonaModal';
import { ConversationList } from './features/conversations/ConversationList';
import { CreateConversationModal } from './features/conversations/CreateConversationModal';
import { ChatWindow } from './features/chat/ChatWindow';
import type { Conversation, CreateConversationInput, CreatePersonaInput, Persona, UpdatePersonaInput } from './types';
import { personaApi } from './services/personaApi';
import { conversationApi } from './services/conversationApi';
import { AlertCircle } from 'lucide-react';

export function App() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'personas' | 'chat'>('dashboard');
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversation, setActiveConversation] = useState<Conversation | null>(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Modals state
  const [isPersonaModalOpen, setIsPersonaModalOpen] = useState(false);
  const [editingPersona, setEditingPersona] = useState<Persona | null>(null);
  const [isConversationModalOpen, setIsConversationModalOpen] = useState(false);
  const [preselectedPersonaId, setPreselectedPersonaId] = useState<string | undefined>(undefined);

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [fetchedPersonas, fetchedConversations] = await Promise.all([
        personaApi.listPersonas(),
        conversationApi.listConversations(),
      ]);
      setPersonas(fetchedPersonas);
      setConversations(fetchedConversations);
    } catch (err: any) {
      console.warn('Backend connection note:', err);
      setError(
        err?.message ||
          'Unable to communicate with the Spring Boot backend. Ensure the backend is running on port 8080.'
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  // Persona handlers
  const handleOpenCreatePersona = () => {
    setEditingPersona(null);
    setIsPersonaModalOpen(true);
  };

  const handleOpenEditPersona = (persona: Persona) => {
    setEditingPersona(persona);
    setIsPersonaModalOpen(true);
  };

  const handleSavePersona = async (data: CreatePersonaInput | UpdatePersonaInput) => {
    if (editingPersona) {
      await personaApi.updatePersona(editingPersona.id, data as UpdatePersonaInput);
    } else {
      await personaApi.createPersona(data as CreatePersonaInput);
    }
    await loadData();
  };

  const handleDeletePersona = async (id: string) => {
    if (confirm('Are you sure you want to delete this persona?')) {
      try {
        await personaApi.deletePersona(id);
        await loadData();
      } catch (err: any) {
        alert(err?.message || 'Failed to delete persona');
      }
    }
  };

  // Conversation handlers
  const handleOpenCreateConversation = (personaId?: string) => {
    setPreselectedPersonaId(personaId);
    setIsConversationModalOpen(true);
  };

  const handleSaveConversation = async (data: CreateConversationInput) => {
    const newConv = await conversationApi.createConversation(data);
    await loadData();
    setActiveConversation(newConv);
    setActiveTab('chat');
  };

  const handleDeleteConversation = async (id: string) => {
    if (confirm('Are you sure you want to delete this conversation?')) {
      try {
        await conversationApi.deleteConversation(id);
        if (activeConversation?.id === id) {
          setActiveConversation(null);
        }
        await loadData();
      } catch (err: any) {
        alert(err?.message || 'Failed to delete conversation');
      }
    }
  };

  const handleSelectConversation = (conv: Conversation) => {
    setActiveConversation(conv);
    setActiveTab('chat');
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      <Navbar
        activeTab={activeTab}
        setActiveTab={(tab) => {
          setActiveTab(tab);
          if (tab !== 'chat') {
            setActiveConversation(null);
          }
        }}
        onOpenNewPersona={handleOpenCreatePersona}
        onOpenNewConversation={() => handleOpenCreateConversation()}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {error && (
          <div className="mb-6 p-4 rounded-2xl bg-amber-500/10 border border-amber-500/20 text-amber-300 text-xs sm:text-sm flex items-start justify-between shadow-lg">
            <div className="flex items-start space-x-3">
              <AlertCircle className="w-5 h-5 shrink-0 text-amber-400 mt-0.5" />
              <div>
                <p className="font-semibold text-amber-200">Backend API Notice</p>
                <p className="text-amber-300/90 mt-0.5 leading-relaxed">{error}</p>
              </div>
            </div>
            <button
              onClick={loadData}
              className="px-3 py-1 bg-amber-500/20 hover:bg-amber-500/30 text-amber-200 font-semibold rounded-lg text-xs transition-colors shrink-0 ml-4"
            >
              Retry
            </button>
          </div>
        )}

        {activeTab === 'dashboard' && (
          <DashboardView
            personas={personas}
            conversations={conversations}
            loading={loading}
            onSelectConversation={handleSelectConversation}
            onOpenNewPersona={handleOpenCreatePersona}
            onOpenNewConversation={() => handleOpenCreateConversation()}
            onEditPersona={handleOpenEditPersona}
            onDeletePersona={handleDeletePersona}
            onNavigateTab={(tab) => setActiveTab(tab)}
          />
        )}

        {activeTab === 'personas' && (
          <PersonaList
            personas={personas}
            loading={loading}
            onEdit={handleOpenEditPersona}
            onDelete={handleDeletePersona}
            onStartChat={(p) => handleOpenCreateConversation(p.id)}
            onOpenNewPersona={handleOpenCreatePersona}
          />
        )}

        {activeTab === 'chat' && (
          activeConversation ? (
            <ChatWindow
              conversation={activeConversation}
              persona={personas.find((p) => p.id === activeConversation.personaId)}
              onBack={() => setActiveConversation(null)}
              onDeleteConversation={handleDeleteConversation}
            />
          ) : (
            <ConversationList
              conversations={conversations}
              personas={personas}
              loading={loading}
              onSelectConversation={handleSelectConversation}
              onOpenNewConversation={() => handleOpenCreateConversation()}
              onDeleteConversation={handleDeleteConversation}
            />
          )
        )}
      </main>

      {/* Modals */}
      <PersonaModal
        isOpen={isPersonaModalOpen}
        onClose={() => setIsPersonaModalOpen(false)}
        onSubmit={handleSavePersona}
        initialData={editingPersona}
      />

      <CreateConversationModal
        isOpen={isConversationModalOpen}
        onClose={() => setIsConversationModalOpen(false)}
        personas={personas}
        preselectedPersonaId={preselectedPersonaId}
        onSubmit={handleSaveConversation}
      />
    </div>
  );
}

export default App;
