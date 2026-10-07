import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2 } from 'lucide-react';

interface MessageInputProps {
  onSendMessage: (content: string) => Promise<void>;
  loading: boolean;
  disabled?: boolean;
}

export const MessageInput: React.FC<MessageInputProps> = ({
  onSendMessage,
  loading,
  disabled = false,
}) => {
  const [content, setContent] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 140)}px`;
    }
  }, [content]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!content.trim() || loading || disabled) return;

    const text = content.trim();
    setContent('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }

    await onSendMessage(text);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="relative flex items-end gap-2 p-3 bg-slate-900/90 border border-slate-800 rounded-2xl shadow-xl backdrop-blur-md">
      <textarea
        ref={textareaRef}
        rows={1}
        value={content}
        onChange={(e) => setContent(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Type a message... (Press Enter to send, Shift+Enter for new line)"
        disabled={disabled || loading}
        className="flex-1 max-h-36 py-2 px-3 bg-transparent text-slate-100 placeholder-slate-500 text-sm focus:outline-none resize-none leading-relaxed disabled:opacity-50"
      />

      <button
        type="submit"
        disabled={!content.trim() || loading || disabled}
        className="flex items-center justify-center w-10 h-10 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white shadow-md shadow-indigo-600/30 disabled:opacity-40 disabled:hover:bg-indigo-600 transition-all shrink-0 hover:scale-105 active:scale-95"
      >
        {loading ? (
          <Loader2 className="w-4 h-4 animate-spin text-white" />
        ) : (
          <Send className="w-4 h-4 text-white" />
        )}
      </button>
    </form>
  );
};
