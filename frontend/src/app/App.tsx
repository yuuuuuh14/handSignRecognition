import { useState, useRef } from 'react';
import { Header } from './components/Header';
import { CameraViewport } from './components/CameraViewport';
import { VocabularyPanel } from './components/VocabularyPanel';
import { TranslationOverlay } from './components/TranslationOverlay';
import { SettingsModal } from './components/SettingsModal';
import { useWebcam } from './hooks/useWebcam';
import { useHandSignRecognition } from './hooks/useHandSignRecognition';

export default function App() {
  const [cameraEnabled, setCameraEnabled] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [theme, setTheme] = useState<'dark' | 'light'>('dark');
  const [threshold, setThreshold] = useState(80);
  const videoRef = useRef<HTMLVideoElement>(null);

  const { status, errorMsg } = useWebcam(videoRef, cameraEnabled);
  const isReady = status === 'ready';
  
  const { activeWord, confidence, boundingBox } = useHandSignRecognition(videoRef, isReady, threshold);

  const handleToggleCamera = () => {
    setCameraEnabled(prev => !prev);
  };

  return (
    <div className={`min-h-screen font-outfit flex flex-col overflow-hidden selection:bg-blue-500/30 transition-colors ${theme === 'dark' ? 'dark bg-slate-950 text-slate-100' : 'bg-slate-50 text-slate-900'}`}>
      <Header 
        cameraEnabled={cameraEnabled} 
        onToggleCamera={handleToggleCamera} 
        onOpenSettings={() => setIsSettingsOpen(true)}
      />
      
      <main className="flex-1 p-4 md:p-6 flex flex-col lg:flex-row gap-6 max-w-[1600px] mx-auto w-full lg:h-[calc(100vh-73px)] overflow-y-auto lg:overflow-hidden">
        
        {/* Main Camera Viewport Area */}
        <section className="flex-1 relative flex flex-col min-h-[400px] lg:min-h-0">
          <CameraViewport 
            videoRef={videoRef}
            status={status}
            errorMsg={errorMsg}
            activeWord={activeWord}
            boundingBox={boundingBox}
          />
          
          {/* Real-Time Translation Display (Overlay) */}
          {status === 'ready' && (
            <TranslationOverlay activeWord={activeWord} confidence={confidence} />
          )}
        </section>

        {/* Sidebar Vocabulary Panel */}
        <aside className="w-full lg:w-[320px] xl:w-[380px] h-[300px] lg:h-full shrink-0">
          <VocabularyPanel activeWord={activeWord} />
        </aside>

      </main>

      <SettingsModal 
        isOpen={isSettingsOpen} 
        onClose={() => setIsSettingsOpen(false)} 
        theme={theme}
        setTheme={setTheme}
        threshold={threshold}
        setThreshold={setThreshold}
      />
    </div>
  );
}
