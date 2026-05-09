import { RefObject } from 'react';
import { CameraStatus } from '../hooks/useWebcam';
import { CameraOff, Loader2, AlertCircle, ScanLine } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

interface CameraViewportProps {
  videoRef: RefObject<HTMLVideoElement | null>;
  status: CameraStatus;
  errorMsg: string;
  activeWord: string | null;
  boundingBox: { x: number, y: number, width: number, height: number } | null;
}

export function CameraViewport({ videoRef, status, errorMsg, activeWord, boundingBox }: CameraViewportProps) {
  return (
    <div className="relative w-full h-full bg-slate-100 dark:bg-slate-950 rounded-2xl overflow-hidden border border-slate-200 dark:border-slate-800 shadow-2xl flex flex-col items-center justify-center isolate transition-colors">
      {/* Video Element */}
      <video
        ref={videoRef}
        className={`absolute inset-0 w-full h-full object-cover transition-opacity duration-700 ${status === 'ready' ? 'opacity-100' : 'opacity-0'} scale-x-[-1]`}
        playsInline
        muted
      />

      {/* Overlays Based on Status */}
      <AnimatePresence mode="wait">
        {status === 'offline' && (
          <motion.div 
            key="offline"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex flex-col items-center justify-center text-slate-500 z-10"
          >
            <div className="w-20 h-20 bg-slate-200 dark:bg-slate-900 rounded-full flex items-center justify-center mb-4 transition-colors">
              <CameraOff className="w-8 h-8 text-slate-400" />
            </div>
            <p className="text-lg font-medium text-slate-600 dark:text-slate-300">카메라 오프라인</p>
            <p className="text-sm text-slate-500 mt-1 max-w-sm text-center">"카메라 켜기" 버튼을 눌러 실시간 번역을 시작하세요</p>
          </motion.div>
        )}

        {status === 'initializing' && (
          <motion.div 
            key="initializing"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex flex-col items-center justify-center text-blue-400 z-10"
          >
            <Loader2 className="w-10 h-10 animate-spin mb-4" />
            <p className="text-lg font-medium animate-pulse">AI 모델 및 카메라 초기화 중...</p>
          </motion.div>
        )}

        {status === 'error' && (
          <motion.div 
            key="error"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex flex-col items-center justify-center text-red-400 z-10 px-6 text-center"
          >
            <AlertCircle className="w-12 h-12 mb-4" />
            <p className="text-lg font-medium">카메라 오류</p>
            <p className="text-sm text-red-500/80 mt-2">{errorMsg}</p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Detection Overlays (Only when ready) */}
      {status === 'ready' && (
        <>
          {/* Bounding Box */}
          <AnimatePresence>
            {boundingBox && (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ 
                  opacity: 1, 
                  scale: 1,
                  left: `${boundingBox.x}%`,
                  top: `${boundingBox.y}%`,
                  width: `${boundingBox.width}%`,
                  height: `${boundingBox.height}%`
                }}
                exit={{ opacity: 0, scale: 0.95 }}
                transition={{ type: "spring", stiffness: 300, damping: 25 }}
                className="absolute border-2 border-green-400/80 bg-green-400/10 rounded-lg shadow-[0_0_15px_rgba(74,222,128,0.3)] z-10 pointer-events-none"
              >
                <div className="absolute -top-3 -right-3">
                   <span className="relative flex h-3 w-3">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
                  </span>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Scanning Overlay Effect */}
          {!activeWord && (
             <div className="absolute inset-0 pointer-events-none z-10 flex flex-col items-center justify-center">
                <div className="w-3/4 max-w-md h-64 border border-dashed border-slate-400/50 dark:border-white/20 rounded-xl relative overflow-hidden transition-colors">
                    <motion.div 
                      animate={{ top: ['-10%', '110%'] }}
                      transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                      className="absolute left-0 right-0 h-1 bg-gradient-to-r from-transparent via-blue-500/50 to-transparent shadow-[0_0_10px_rgba(59,130,246,0.5)]"
                    />
                </div>
                <p className="text-slate-700 dark:text-white/80 mt-4 flex items-center gap-2 font-medium bg-white/70 dark:bg-black/40 px-4 py-2 rounded-full backdrop-blur-md shadow-lg dark:shadow-none border border-slate-200/50 dark:border-transparent transition-colors">
                   <ScanLine className="w-4 h-4" /> 수어 동작을 카메라에 보여주세요
                </p>
             </div>
          )}
        </>
      )}

      {/* Status Badge */}
      <div className="absolute top-4 left-4 z-20 flex items-center gap-2 bg-black/50 backdrop-blur-md px-3 py-1.5 rounded-full border border-white/10">
        <div className={`w-2.5 h-2.5 rounded-full ${
          status === 'ready' ? 'bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.8)]' : 
          status === 'initializing' ? 'bg-yellow-500 animate-pulse' : 
          status === 'error' ? 'bg-red-500' : 'bg-slate-500'
        }`} />
        <span className="text-xs font-semibold text-white uppercase tracking-wider">
          {status === 'ready' ? '라이브' : 
           status === 'initializing' ? '준비중' : 
           status === 'error' ? '오류' : '오프라인'}
        </span>
      </div>
    </div>
  );
}
