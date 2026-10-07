import React from 'react';
import { Bot, Sparkles, MessageSquare, Users, LayoutDashboard, Terminal } from 'lucide-react';

interface NavbarProps {
  activeTab: 'dashboard' | 'personas' | 'chat';
  setActiveTab: (tab: 'dashboard' | 'personas' | 'chat') => void;
  onOpenNewPersona: () => void;
  onOpenNewConversation: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  onOpenNewPersona,
  onOpenNewConversation,
}) => {
  return (
    <header className="sticky top-0 z-40 w-full border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Brand */}
          <div className="flex items-center space-x-3 cursor-pointer" onClick={() => setActiveTab('dashboard')}>
            <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-purple-500 shadow-lg shadow-indigo-500/20">
              <Bot className="w-5 h-5 text-white" />
              <div className="absolute -top-1 -right-1 w-3 h-3 bg-emerald-500 rounded-full border-2 border-slate-950 animate-pulse" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-bold text-lg text-white tracking-tight">Persona Engine</span>
                <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-semibold bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                  Phase 1 Foundation
                </span>
              </div>
              <p className="text-xs text-slate-400 font-mono">Personalized Conversational AI</p>
            </div>
          </div>

          {/* Navigation Items */}
          <nav className="flex items-center space-x-1 sm:space-x-2">
            <button
              onClick={() => setActiveTab('dashboard')}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                activeTab === 'dashboard'
                  ? 'bg-slate-800 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
              }`}
            >
              <LayoutDashboard className="w-4 h-4" />
              <span>Dashboard</span>
            </button>
            <button
              onClick={() => setActiveTab('personas')}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                activeTab === 'personas'
                  ? 'bg-slate-800 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
              }`}
            >
              <Users className="w-4 h-4" />
              <span>Personas</span>
            </button>
            <button
              onClick={() => setActiveTab('chat')}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                activeTab === 'chat'
                  ? 'bg-slate-800 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
              }`}
            >
              <MessageSquare className="w-4 h-4" />
              <span>Chat / Conversations</span>
            </button>
          </nav>

          {/* Quick Actions */}
          <div className="flex items-center space-x-2.5">
            <button
              onClick={onOpenNewPersona}
              className="hidden sm:inline-flex items-center space-x-1.5 px-3 py-1.5 text-xs font-medium rounded-lg text-slate-300 bg-slate-900 hover:bg-slate-800 border border-slate-800 transition-colors"
            >
              <Sparkles className="w-3.5 h-3.5 text-purple-400" />
              <span>+ New Persona</span>
            </button>
            <button
              onClick={onOpenNewConversation}
              className="inline-flex items-center space-x-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg text-white bg-indigo-600 hover:bg-indigo-500 shadow-md shadow-indigo-600/20 transition-all hover:scale-[1.02] active:scale-[0.98]"
            >
              <Terminal className="w-3.5 h-3.5" />
              <span>+ Start Chat</span>
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};
