import { motion, AnimatePresence } from 'motion/react';
import { X, Moon, Sun, Camera, Sliders } from 'lucide-react';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  theme: 'dark' | 'light';
  setTheme: (theme: 'dark' | 'light') => void;
  threshold: number;
  setThreshold: (val: number) => void;
}

export function SettingsModal({ isOpen, onClose, theme, setTheme, threshold, setThreshold }: SettingsModalProps) {
  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-slate-900/40 dark:bg-black/60 backdrop-blur-sm z-40"
          />
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-md bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl shadow-2xl z-50 overflow-hidden transition-colors"
          >
            <div className="flex items-center justify-between p-4 border-b border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/50 transition-colors">
              <h2 className="text-lg font-semibold text-slate-900 dark:text-white flex items-center gap-2">
                <Sliders className="w-5 h-5" /> 설정
              </h2>
              <button 
                onClick={onClose} 
                className="p-1.5 text-slate-500 hover:text-slate-900 dark:text-slate-400 dark:hover:text-white hover:bg-slate-200 dark:hover:bg-slate-800 rounded-lg transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div className="p-6 space-y-6">
              {/* Theme Toggle */}
              <div>
                <label className="text-sm font-medium text-slate-600 dark:text-slate-400 block mb-3">테마</label>
                <div className="flex bg-slate-100 dark:bg-slate-950 p-1.5 rounded-xl border border-slate-200 dark:border-slate-800 transition-colors">
                  <button 
                    onClick={() => setTheme('dark')}
                    className={`flex-1 flex items-center justify-center gap-2 py-2.5 rounded-lg shadow-sm font-medium text-sm transition-all ${
                      theme === 'dark' ? 'bg-slate-800 text-white' : 'text-slate-500 hover:text-slate-900'
                    }`}
                  >
                    <Moon className="w-4 h-4" /> 다크 모드
                  </button>
                  <button 
                    onClick={() => setTheme('light')}
                    className={`flex-1 flex items-center justify-center gap-2 py-2.5 rounded-lg shadow-sm font-medium text-sm transition-all ${
                      theme === 'light' ? 'bg-white text-slate-900' : 'text-slate-500 hover:text-slate-900 dark:text-slate-400 dark:hover:text-white'
                    }`}
                  >
                    <Sun className="w-4 h-4" /> 라이트 모드
                  </button>
                </div>
              </div>

              {/* Camera Selection */}
              <div>
                <label className="text-sm font-medium text-slate-600 dark:text-slate-400 block mb-3">카메라 장치</label>
                <div className="relative">
                  <Camera className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400 pointer-events-none" />
                  <select className="w-full bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded-xl pl-10 pr-4 py-3 text-slate-900 dark:text-white text-sm appearance-none outline-none focus:border-blue-500 transition-colors">
                    <option>기본 웹 카메라</option>
                    <option>외부 USB 카메라</option>
                    <option>가상 카메라</option>
                  </select>
                </div>
              </div>

              {/* Detection Threshold */}
              <div>
                <div className="flex justify-between mb-3">
                  <label className="text-sm font-medium text-slate-600 dark:text-slate-400">최소 인식 정확도</label>
                  <span className="text-sm font-semibold text-blue-500 dark:text-blue-400">{threshold}%</span>
                </div>
                <input 
                  type="range" 
                  min="50" 
                  max="99" 
                  value={threshold}
                  onChange={(e) => setThreshold(Number(e.target.value))}
                  className="w-full accent-blue-500 h-2 bg-slate-200 dark:bg-slate-800 rounded-lg appearance-none cursor-pointer" 
                />
                <p className="text-xs text-slate-500 mt-2">
                  값이 높을수록 오작동은 줄어들지만, 더 정확한 수어 동작이 필요합니다.
                </p>
              </div>
            </div>
            
            <div className="p-4 border-t border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/50 flex justify-end transition-colors">
              <button 
                onClick={onClose}
                className="px-5 py-2 bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium rounded-xl transition-colors"
              >
                확인
              </button>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
