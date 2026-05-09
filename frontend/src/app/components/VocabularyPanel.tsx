import { VOCABULARY } from '../hooks/useHandSignSimulation';
import { motion } from 'motion/react';
import { CheckCircle2 } from 'lucide-react';

interface VocabularyPanelProps {
  activeWord: string | null;
}

export function VocabularyPanel({ activeWord }: VocabularyPanelProps) {
  return (
    <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 flex flex-col h-full shadow-xl overflow-hidden transition-colors">
      <div className="mb-6">
        <h2 className="text-lg font-semibold text-slate-900 dark:text-white tracking-tight flex items-center justify-between">
          <span>수어 단어장</span>
          <span className="text-xs font-normal text-slate-600 dark:text-slate-400 bg-slate-100 dark:bg-slate-800 px-2 py-1 rounded-full">
            {VOCABULARY.length}개 단어
          </span>
        </h2>
        <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">현재 AI 모델이 인식할 수 있는 한국 수어 목록입니다.</p>
      </div>

      <div className="flex-1 overflow-y-auto pr-2 custom-scrollbar space-y-2">
        {VOCABULARY.map((word) => {
          const isActive = activeWord === word;
          return (
            <motion.div
              key={word}
              animate={{
                scale: isActive ? 1.02 : 1,
              }}
              className={`p-3 rounded-xl border flex items-center justify-between transition-all duration-300 ${
                isActive 
                  ? 'bg-blue-500/15 border-blue-500/50 text-blue-500 dark:text-blue-400 font-semibold' 
                  : 'bg-slate-100 dark:bg-slate-800/50 border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 font-medium'
              }`}
            >
              <span>{word}</span>
              {isActive && (
                <motion.div
                  initial={{ scale: 0, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ type: "spring", stiffness: 400, damping: 20 }}
                >
                  <CheckCircle2 className="w-5 h-5 text-blue-500" />
                </motion.div>
              )}
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
