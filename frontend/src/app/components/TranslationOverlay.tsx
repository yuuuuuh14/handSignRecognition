import { motion, AnimatePresence } from 'motion/react';

interface TranslationOverlayProps {
  activeWord: string | null;
  confidence: number;
}

export function TranslationOverlay({ activeWord, confidence }: TranslationOverlayProps) {
  return (
    <div className="absolute bottom-6 left-0 right-0 flex justify-center pointer-events-none z-30 px-4">
      <AnimatePresence mode="wait">
        {activeWord ? (
          <motion.div
            key={activeWord}
            initial={{ y: 20, opacity: 0, scale: 0.95 }}
            animate={{ y: 0, opacity: 1, scale: 1 }}
            exit={{ y: -10, opacity: 0, scale: 0.95 }}
            transition={{ type: "spring", stiffness: 300, damping: 25 }}
            className="bg-white/90 dark:bg-black/70 backdrop-blur-xl border border-slate-200 dark:border-white/10 shadow-2xl rounded-2xl p-6 flex flex-col items-center min-w-[280px] transition-colors"
          >
            <p className="text-sm font-medium text-slate-500 dark:text-slate-400 mb-1 uppercase tracking-widest">번역 결과</p>
            <h2 className="text-5xl md:text-6xl font-bold text-slate-900 dark:text-white tracking-tight font-outfit text-center">
              {activeWord}
            </h2>
            <div className="mt-4 w-full flex items-center justify-between text-xs font-medium text-slate-600 dark:text-slate-400">
               <span>정확도</span>
               <span className="text-green-600 dark:text-green-400">{confidence}%</span>
            </div>
            <div className="mt-1 w-full bg-slate-200 dark:bg-slate-800 h-1.5 rounded-full overflow-hidden transition-colors">
               <motion.div 
                 initial={{ width: 0 }}
                 animate={{ width: `${confidence}%` }}
                 className="h-full bg-green-500 rounded-full"
               />
            </div>
          </motion.div>
        ) : (
          <motion.div
            key="empty"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="bg-white/70 dark:bg-black/40 backdrop-blur-md border border-slate-200 dark:border-white/5 shadow-xl rounded-2xl py-4 px-8 transition-colors"
          >
            <p className="text-slate-600 dark:text-slate-400 font-medium">수어 동작을 기다리는 중...</p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
