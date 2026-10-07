import React, { useState } from 'react';
import { Bot, User as UserIcon, Info, ChevronDown, ChevronUp } from 'lucide-react';
import type { Message } from '../../types';

interface MessageBubbleProps {
  message: Message;
  personaName?: string;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({ message, personaName = 'Persona' }) => {
  const [showMetadata, setShowMetadata] = useState(false);
  const isUser = message.senderType === 'USER';
  const isSystem = message.senderType === 'SYSTEM';

  const formattedTime = new Date(message.createdAt).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  });

  if (isSystem) {
    return (
      <div className="flex justify-center my-3">
        <div className="px-3 py-1 rounded-full bg-slate-900 border border-slate-800 text-[11px] text-slate-400 font-mono">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className={`flex items-start gap-3 my-4 group ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
      {/* Avatar */}
      <div
        className={`flex items-center justify-center w-8 h-8 rounded-xl shrink-0 mt-0.5 shadow-md ${
          isUser
            ? 'bg-gradient-to-tr from-indigo-600 to-indigo-500 text-white'
            : 'bg-gradient-to-tr from-purple-600 to-indigo-600 text-white border border-purple-400/30'
        }`}
      >
        {isUser ? <UserIcon className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
      </div>

      {/* Bubble Container */}
      <div className={`flex flex-col max-w-[80%] sm:max-w-[70%] ${isUser ? 'items-end' : 'items-start'}`}>
        <div className="flex items-center space-x-2 mb-1 px-1">
          <span className="text-xs font-semibold text-slate-300">
            {isUser ? 'You' : personaName}
          </span>
          <span className="text-[10px] text-slate-500 font-mono">{formattedTime}</span>
        </div>

        <div
          className={`px-4 py-3 rounded-2xl text-sm leading-relaxed shadow-sm break-words ${
            isUser
              ? 'bg-indigo-600 text-white rounded-tr-sm'
              : 'bg-slate-900 border border-slate-800 text-slate-100 rounded-tl-sm'
          }`}
        >
          {message.content}
        </div>

        {/* Metadata Inspector for Future ML Details */}
        {message.metadata && Object.keys(message.metadata).length > 0 && !isUser && (
          <div className="mt-1.5 px-1">
            <button
              onClick={() => setShowMetadata(!showMetadata)}
              className="inline-flex items-center space-x-1 text-[11px] text-slate-500 hover:text-indigo-400 font-mono transition-colors"
            >
              <Info className="w-3 h-3" />
              <span>Orchestration & ML Context</span>
              {showMetadata ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            </button>

            {showMetadata && (
              <div className="mt-2 p-2.5 rounded-xl bg-slate-950/80 border border-slate-800 text-[11px] font-mono text-slate-300 space-y-1 shadow-inner animate-in fade-in duration-150">
                <div className="text-indigo-400 font-semibold mb-1">Architecture Metadata:</div>
                {Object.entries(message.metadata).map(([key, val]) => (
                  <div key={key} className="flex justify-between gap-4">
                    <span className="text-slate-500">{key}:</span>
                    <span className="text-slate-200 truncate max-w-[200px]">
                      {typeof val === 'object' ? JSON.stringify(val) : String(val)}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
