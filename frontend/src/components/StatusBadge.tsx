import type { EntityStatus } from '../types';

interface StatusBadgeProps {
  status: EntityStatus | string;
  className?: string;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status, className = '' }) => {
  const getBadgeStyle = () => {
    switch (status) {
      case 'ACTIVE':
        return 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20';
      case 'ARCHIVED':
        return 'bg-amber-500/10 text-amber-400 border-amber-500/20';
      case 'DELETED':
        return 'bg-rose-500/10 text-rose-400 border-rose-500/20';
      case 'MOCK_ENGINE':
        return 'bg-indigo-500/10 text-indigo-400 border-indigo-500/20';
      default:
        return 'bg-slate-800 text-slate-300 border-slate-700';
    }
  };

  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium border ${getBadgeStyle()} ${className}`}
    >
      <span className="w-1.5 h-1.5 rounded-full bg-current mr-1.5 opacity-80" />
      {status}
    </span>
  );
};
